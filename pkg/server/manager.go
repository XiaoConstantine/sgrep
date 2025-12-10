package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"time"
)

const (
	DefaultPort    = 8080
	DefaultHost    = "localhost"
	StartupTimeout = 15 * time.Second
	HealthInterval = 500 * time.Millisecond
)

// Manager handles llama.cpp server lifecycle.
type Manager struct {
	sgrepHome string
	port      int
	host      string
}

// NewManager creates a server manager.
func NewManager() (*Manager, error) {
	home, err := getSgrepHome()
	if err != nil {
		return nil, err
	}

	port := DefaultPort
	if p := os.Getenv("SGREP_PORT"); p != "" {
		if parsed, err := strconv.Atoi(p); err == nil {
			port = parsed
		}
	}

	return &Manager{
		sgrepHome: home,
		port:      port,
		host:      DefaultHost,
	}, nil
}

// IsRunning checks if the embedding server is responding.
func (m *Manager) IsRunning() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", m.healthURL(), nil)
	if err != nil {
		return false
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	return resp.StatusCode == http.StatusOK
}

// Start starts the llama.cpp server if not already running.
func (m *Manager) Start() error {
	if m.IsRunning() {
		return nil
	}

	// Check if llama-server binary exists
	llamaPath, err := m.findLlamaServer()
	if err != nil {
		return err
	}

	// Check if model exists
	modelPath := m.ModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model not found at %s. Run 'sgrep setup' first", modelPath)
	}

	// Clean up stale PID file
	m.cleanStalePID()

	// Start the server
	logPath := filepath.Join(m.sgrepHome, "server.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	// Calculate optimal settings based on CPU
	// Reference: https://github.com/ggml-org/llama.cpp/discussions/4130
	numCPU := runtime.NumCPU()

	// Threads: use most CPUs for embedding workload
	threads := numCPU
	if threads > 16 {
		threads = 16
	}

	// Parallel slots: more slots = more parallelism but each gets less context
	// Formula: n_slot_ctx = n_ctx / parallel (each slot gets portion of context)
	// Tested: 32 slots optimal for Apple Silicon, smaller context = faster attention
	parallelSlots := 32
	if numCPU < 8 {
		parallelSlots = 16
	}

	// Context size: 256 tokens per slot - speed optimized
	// Benchmarks show identical quality (NDCG@10=0.6218) with 2.5x faster indexing
	contextSize := parallelSlots * 256

	// Build command with GPU support if available
	args := []string{
		"-m", modelPath,
		"--embedding",
		"--port", strconv.Itoa(m.port),
		"--host", m.host,
		"-c", strconv.Itoa(contextSize),
		"-b", "2048",  // batch size (match typical input)
		"-ub", "2048", // microbatch (equal to -b for embeddings)
		"--threads", strconv.Itoa(threads),
		"-ngl", "99",  // Use GPU (Metal on Mac, CUDA on Linux) - offload all layers
		"-np", strconv.Itoa(parallelSlots),
		"-cb", // Continuous batching - CRITICAL for parallel to work!
	}

	cmd := exec.Command(llamaPath, args...)

	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Detach from parent process group
	}

	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Write PID file
	pidPath := m.pidPath()
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0644); err != nil {
		// Non-fatal, just log
		fmt.Fprintf(os.Stderr, "warning: failed to write PID file: %v\n", err)
	}

	// Wait for server to be ready
	if err := m.waitForReady(); err != nil {
		// Try to kill the server if it didn't start properly
		_ = m.Stop()
		return err
	}

	return nil
}

// Stop stops the llama.cpp server.
func (m *Manager) Stop() error {
	pid, err := m.readPID()
	if err != nil {
		// No PID file, check if something is running on the port
		if m.IsRunning() {
			return fmt.Errorf("server running but no PID file found; kill manually on port %d", m.port)
		}
		return nil
	}

	// Find the process
	proc, err := os.FindProcess(pid)
	if err != nil {
		m.removePIDFile()
		return nil
	}

	// Send SIGTERM
	if err := proc.Signal(syscall.SIGTERM); err != nil {
		// Process might already be dead
		m.removePIDFile()
		return nil
	}

	// Wait briefly for graceful shutdown
	time.Sleep(500 * time.Millisecond)

	// Force kill if still running
	if m.IsRunning() {
		_ = proc.Signal(syscall.SIGKILL)
	}

	m.removePIDFile()
	return nil
}

// Status returns server status info.
func (m *Manager) Status() (running bool, pid int, port int) {
	port = m.port
	running = m.IsRunning()
	pid, _ = m.readPID()
	return
}

// ModelPath returns the path to the embedding model.
func (m *Manager) ModelPath() string {
	return filepath.Join(m.sgrepHome, "models", "nomic-embed-text-v1.5.Q8_0.gguf")
}

// ModelsDir returns the models directory.
func (m *Manager) ModelsDir() string {
	return filepath.Join(m.sgrepHome, "models")
}

// Endpoint returns the server endpoint URL.
func (m *Manager) Endpoint() string {
	return fmt.Sprintf("http://%s:%d", m.host, m.port)
}

// EnsureRunning starts the server if not running, used by embedder.
func (m *Manager) EnsureRunning() error {
	if m.IsRunning() {
		return nil
	}
	return m.Start()
}

func (m *Manager) healthURL() string {
	return fmt.Sprintf("http://%s:%d/health", m.host, m.port)
}

func (m *Manager) pidPath() string {
	return filepath.Join(m.sgrepHome, "server.pid")
}

func (m *Manager) readPID() (int, error) {
	data, err := os.ReadFile(m.pidPath())
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(string(data))
}

func (m *Manager) removePIDFile() {
	_ = os.Remove(m.pidPath())
}

func (m *Manager) cleanStalePID() {
	pid, err := m.readPID()
	if err != nil {
		return
	}

	// Check if process is still alive
	proc, err := os.FindProcess(pid)
	if err != nil {
		m.removePIDFile()
		return
	}

	// Signal 0 checks if process exists
	if err := proc.Signal(syscall.Signal(0)); err != nil {
		m.removePIDFile()
	}
}

func (m *Manager) waitForReady() error {
	deadline := time.Now().Add(StartupTimeout)
	for time.Now().Before(deadline) {
		if m.IsRunning() {
			return nil
		}
		time.Sleep(HealthInterval)
	}
	return fmt.Errorf("server failed to start within %v", StartupTimeout)
}

func (m *Manager) findLlamaServer() (string, error) {
	// Check common names
	names := []string{"llama-server", "llama-server-metal", "server"}

	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}

	// Check in homebrew paths
	brewPaths := []string{
		"/opt/homebrew/bin/llama-server",
		"/usr/local/bin/llama-server",
	}
	for _, p := range brewPaths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	return "", fmt.Errorf("llama-server not found. Install with: brew install llama.cpp")
}

func getSgrepHome() (string, error) {
	if home := os.Getenv("SGREP_HOME"); home != "" {
		return home, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(homeDir, ".sgrep"), nil
}
