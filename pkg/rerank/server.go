package rerank

import (
	"context"
	"fmt"
	"io"
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
	DefaultRerankerPort    = 8081
	DefaultHost            = "localhost"
	RerankerModelURL       = "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf"
	RerankerModelName      = "bge-reranker-v2-m3-Q8_0.gguf"
	RerankerModelSize      = 636_000_000 // ~636MB
	RerankerStartupTimeout = 30 * time.Second
	RerankerHealthInterval = 500 * time.Millisecond
)

// RerankerManager handles llama.cpp reranker server lifecycle.
type RerankerManager struct {
	sgrepHome string
	port      int
	host      string
}

// NewRerankerManager creates a reranker server manager.
func NewRerankerManager() (*RerankerManager, error) {
	home, err := getSgrepHome()
	if err != nil {
		return nil, err
	}

	port := DefaultRerankerPort
	if p := os.Getenv("SGREP_RERANKER_PORT"); p != "" {
		if parsed, err := strconv.Atoi(p); err == nil {
			port = parsed
		}
	}

	return &RerankerManager{
		sgrepHome: home,
		port:      port,
		host:      DefaultHost,
	}, nil
}

// IsRunning checks if the reranker server is responding.
func (m *RerankerManager) IsRunning() bool {
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

// Start starts the llama.cpp reranker server if not already running.
func (m *RerankerManager) Start() error {
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
		return fmt.Errorf("reranker model not found at %s. Run 'sgrep setup --with-rerank' first", modelPath)
	}

	// Clean up stale PID file
	m.cleanStalePID()

	// Start the server
	logPath := filepath.Join(m.sgrepHome, "reranker.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	// Calculate optimal settings based on CPU
	numCPU := runtime.NumCPU()

	threads := numCPU
	if threads > 16 {
		threads = 16
	}

	// For reranking, we need fewer parallel slots since each request has multiple documents
	parallelSlots := 4
	if numCPU >= 8 {
		parallelSlots = 8
	}

	// Context size: 2048 tokens per slot
	contextSize := parallelSlots * 2048

	// Build command with reranking-specific flags
	// Key difference from embedding server: --pooling rank enables reranking mode
	args := []string{
		"-m", modelPath,
		"--embedding",
		"--pooling", "rank", // CRITICAL: This enables reranking mode
		"--port", strconv.Itoa(m.port),
		"--host", m.host,
		"-c", strconv.Itoa(contextSize),
		"-b", "2048",
		"-ub", "2048",
		"--threads", strconv.Itoa(threads),
		"-ngl", "99", // Use GPU if available
		"-np", strconv.Itoa(parallelSlots),
		"-cb", // Continuous batching
	}

	cmd := exec.Command(llamaPath, args...)

	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Detach from parent process group
	}

	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return fmt.Errorf("failed to start reranker server: %w", err)
	}

	// Write PID file
	pidPath := m.pidPath()
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to write reranker PID file: %v\n", err)
	}

	// Wait for server to be ready
	if err := m.waitForReady(); err != nil {
		_ = m.Stop()
		return err
	}

	return nil
}

// Stop stops the reranker server.
func (m *RerankerManager) Stop() error {
	pid, err := m.readPID()
	if err != nil {
		if m.IsRunning() {
			return fmt.Errorf("reranker running but no PID file found; kill manually on port %d", m.port)
		}
		return nil
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		m.removePIDFile()
		return nil
	}

	if err := proc.Signal(syscall.SIGTERM); err != nil {
		m.removePIDFile()
		return nil
	}

	time.Sleep(500 * time.Millisecond)

	if m.IsRunning() {
		_ = proc.Signal(syscall.SIGKILL)
	}

	m.removePIDFile()
	return nil
}

// Status returns reranker server status info.
func (m *RerankerManager) Status() (running bool, pid int, port int) {
	port = m.port
	running = m.IsRunning()
	pid, _ = m.readPID()
	return
}

// ModelPath returns the path to the reranker model.
func (m *RerankerManager) ModelPath() string {
	if customPath := os.Getenv("SGREP_RERANK_MODEL"); customPath != "" {
		return customPath
	}
	return filepath.Join(m.sgrepHome, "models", RerankerModelName)
}

// ModelsDir returns the models directory.
func (m *RerankerManager) ModelsDir() string {
	return filepath.Join(m.sgrepHome, "models")
}

// Endpoint returns the reranker server endpoint URL.
func (m *RerankerManager) Endpoint() string {
	return fmt.Sprintf("http://%s:%d", m.host, m.port)
}

// EnsureRunning starts the reranker server if not running.
func (m *RerankerManager) EnsureRunning() error {
	if m.IsRunning() {
		return nil
	}
	return m.Start()
}

// ModelExists checks if the reranker model is already downloaded.
func (m *RerankerManager) ModelExists() bool {
	info, err := os.Stat(m.ModelPath())
	if err != nil {
		return false
	}
	return info.Size() > 500_000_000 // Should be > 500MB
}

// DownloadModel downloads the reranker model if not present.
func (m *RerankerManager) DownloadModel(progress func(downloaded, total int64)) error {
	modelPath := m.ModelPath()

	// Check if already exists
	if info, err := os.Stat(modelPath); err == nil {
		if info.Size() > 500_000_000 { // Sanity check: should be > 500MB
			return nil
		}
		// File exists but seems incomplete, remove it
		_ = os.Remove(modelPath)
	}

	// Create models directory
	modelsDir := m.ModelsDir()
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	// Download to temp file first
	tmpPath := modelPath + ".tmp"
	defer func() { _ = os.Remove(tmpPath) }()

	if err := downloadFile(tmpPath, RerankerModelURL, progress); err != nil {
		return fmt.Errorf("reranker model download failed: %w", err)
	}

	// Rename to final location
	if err := os.Rename(tmpPath, modelPath); err != nil {
		return fmt.Errorf("failed to save reranker model: %w", err)
	}

	return nil
}

func (m *RerankerManager) healthURL() string {
	return fmt.Sprintf("http://%s:%d/health", m.host, m.port)
}

func (m *RerankerManager) pidPath() string {
	return filepath.Join(m.sgrepHome, "reranker.pid")
}

func (m *RerankerManager) readPID() (int, error) {
	data, err := os.ReadFile(m.pidPath())
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(string(data))
}

func (m *RerankerManager) removePIDFile() {
	_ = os.Remove(m.pidPath())
}

func (m *RerankerManager) cleanStalePID() {
	pid, err := m.readPID()
	if err != nil {
		return
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		m.removePIDFile()
		return
	}

	if err := proc.Signal(syscall.Signal(0)); err != nil {
		m.removePIDFile()
	}
}

func (m *RerankerManager) waitForReady() error {
	deadline := time.Now().Add(RerankerStartupTimeout)
	for time.Now().Before(deadline) {
		if m.IsRunning() {
			return nil
		}
		time.Sleep(RerankerHealthInterval)
	}
	return fmt.Errorf("reranker server failed to start within %v", RerankerStartupTimeout)
}

func (m *RerankerManager) findLlamaServer() (string, error) {
	names := []string{"llama-server", "llama-server-metal", "server"}

	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}

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

func downloadFile(filepath string, url string, progress func(downloaded, total int64)) error {
	client := &http.Client{
		Timeout: 30 * time.Minute,
	}

	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned status %d", resp.StatusCode)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer func() { _ = out.Close() }()

	total := resp.ContentLength
	if total <= 0 {
		total = RerankerModelSize
	}

	var downloaded int64
	buf := make([]byte, 32*1024)

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			_, writeErr := out.Write(buf[:n])
			if writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)
			if progress != nil {
				progress(downloaded, total)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	return nil
}
