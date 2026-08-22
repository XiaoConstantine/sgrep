package server

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/modelcfg"
)

const (
	DefaultPort    = 8080
	DefaultHost    = "localhost"
	StartupTimeout = 15 * time.Second
	HealthInterval = 500 * time.Millisecond
)

// Manager handles llama.cpp server lifecycle.
type Manager struct {
	sgrepHome    string
	port         int
	host         string
	processCheck func(pid int) bool // test hook; nil uses OS process inspection
	launchCheck  func(pid int) bool // test hook; nil inspects managed process arguments
}

type launchConfig struct {
	device    string
	gpuLayers string
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

// IsRunning reports whether the PID owned by this manager is alive and serves
// the llama.cpp embedding endpoint. A generic HTTP 200 from another service is
// deliberately insufficient.
func (m *Manager) IsRunning() bool {
	return m.isRunning(false)
}

// IsReady reports whether the managed server is running with launch arguments
// compatible with the current embedding configuration.
func (m *Manager) IsReady() bool {
	return m.isRunning(true)
}

func (m *Manager) isRunning(requireCompatible bool) bool {
	pid, err := m.readPID()
	if err != nil || !processAlive(pid) || !m.ownsProcess(pid) {
		return false
	}
	if requireCompatible && !m.launchCompatible(pid) {
		return false
	}
	return m.embeddingReady()
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func (m *Manager) ownsProcess(pid int) bool {
	if m.processCheck != nil {
		return m.processCheck(pid)
	}
	expectedStart, err := m.readProcessStartIdentity()
	if err != nil || expectedStart == "" || processStartIdentity(pid) != expectedStart {
		return false
	}
	output, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	if err != nil {
		return false
	}
	command := strings.TrimSpace(string(output))
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return false
	}
	executable := filepath.Base(fields[0])
	if executable != "llama-server" && executable != "llama-server-metal" {
		return false
	}
	return strings.Contains(command, "--port "+strconv.Itoa(m.port)) ||
		strings.Contains(command, "--port="+strconv.Itoa(m.port))
}

func (m *Manager) embeddingReady() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.Endpoint()+"/embedding", bytes.NewBufferString(`{"content":"sgrep health check"}`))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

func (m *Manager) portResponding() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.healthURL(), nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return true
}

func (m *Manager) launchCompatible(pid int) bool {
	if m.launchCheck != nil {
		return m.launchCheck(pid)
	}
	// processCheck replaces OS process inspection in tests. Unless a test also
	// supplies launchCheck, preserve the previous assumption that it is ready.
	if m.processCheck != nil {
		return true
	}
	output, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	if err != nil {
		return false
	}
	_, _, contextSize := launchResources(runtime.NumCPU())
	return containsArgValue(strings.Fields(strings.TrimSpace(string(output))), "-c", strconv.Itoa(contextSize))
}

func containsArgValue(args []string, key, value string) bool {
	for i, arg := range args {
		if arg == key && i+1 < len(args) && args[i+1] == value {
			return true
		}
		if arg == key+"="+value {
			return true
		}
	}
	return false
}

// Start starts the llama.cpp server if not already running.
func (m *Manager) Start() error {
	m.cleanStalePID()
	if m.IsReady() {
		return nil
	}
	if pid, err := m.readPID(); err == nil && processAlive(pid) && m.ownsProcess(pid) {
		if err := m.Stop(); err != nil {
			return err
		}
	}
	if m.portResponding() {
		return fmt.Errorf("port %d is occupied by a process that is not the managed embedding server; set SGREP_PORT or SGREP_ENDPOINT", m.port)
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

	threads, parallelSlots, contextSize := launchResources(runtime.NumCPU())

	var startErrs []string
	for _, cfg := range m.launchConfigs() {
		if err := m.startWithConfig(llamaPath, modelPath, threads, parallelSlots, contextSize, cfg); err == nil {
			return nil
		} else {
			startErrs = append(startErrs, err.Error())
			_ = m.Stop()
		}
	}

	return fmt.Errorf("failed to start llama-server: %s", strings.Join(startErrs, "; "))
}

func launchResources(numCPU int) (threads, parallelSlots, contextSize int) {
	// Reference: https://github.com/ggml-org/llama.cpp/discussions/4130
	threads = min(numCPU, 16)
	parallelSlots = 16
	if numCPU < 8 {
		parallelSlots = 8
	}
	contextSize = parallelSlots * modelcfg.ContextTokens()
	return
}

func (m *Manager) startWithConfig(llamaPath, modelPath string, threads, parallelSlots, contextSize int, cfg launchConfig) error {
	logPath := filepath.Join(m.sgrepHome, "server.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer func() { _ = logFile.Close() }()

	args := m.buildArgs(modelPath, threads, parallelSlots, contextSize, cfg)
	cmd := exec.Command(llamaPath, args...)

	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Detach from parent process group
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	startIdentity := processStartIdentity(cmd.Process.Pid)
	if startIdentity == "" {
		_ = cmd.Process.Signal(syscall.SIGKILL)
		_, _ = cmd.Process.Wait()
		return fmt.Errorf("identify managed server process")
	}
	pidState := fmt.Sprintf("%d\n%s\n", cmd.Process.Pid, startIdentity)
	if err := os.WriteFile(m.pidPath(), []byte(pidState), 0644); err != nil {
		_ = cmd.Process.Signal(syscall.SIGKILL)
		_, _ = cmd.Process.Wait()
		return fmt.Errorf("persist managed server PID: %w", err)
	}

	if err := m.waitForReady(); err != nil {
		// Use the direct child handle: ownership probes may be unavailable or
		// truncated on this platform during failed startup.
		_ = cmd.Process.Signal(syscall.SIGKILL)
		_, _ = cmd.Process.Wait()
		m.removePIDFile()
		return fmt.Errorf("device=%q gpu_layers=%s: %w", cfg.device, cfg.gpuLayers, err)
	}

	return nil
}

func (m *Manager) buildArgs(modelPath string, threads, parallelSlots, contextSize int, cfg launchConfig) []string {
	args := []string{
		"-m", modelPath,
		"--embedding",
		"--port", strconv.Itoa(m.port),
		"--host", m.host,
		"-c", strconv.Itoa(contextSize),
		"-b", "2048", // batch size (match typical input)
		"-ub", "2048", // microbatch (equal to -b for embeddings)
		"--threads", strconv.Itoa(threads),
		"-ngl", cfg.gpuLayers,
		"-np", strconv.Itoa(parallelSlots),
		"-cb", // Continuous batching - CRITICAL for parallel to work!
	}

	if cfg.device != "" {
		args = append(args, "--device", cfg.device)
	}

	return args
}

func (m *Manager) launchConfigs() []launchConfig {
	device := strings.TrimSpace(serverDevice())
	gpuLayers := strings.TrimSpace(serverGPULayers(device))
	if device != "" {
		return []launchConfig{{device: device, gpuLayers: gpuLayers}}
	}

	return []launchConfig{
		{gpuLayers: gpuLayers},
		{device: "none", gpuLayers: "0"},
	}
}

func serverDevice() string {
	if device := strings.TrimSpace(os.Getenv("SGREP_DEVICE")); device != "" {
		return device
	}
	return strings.TrimSpace(os.Getenv("LLAMA_ARG_DEVICE"))
}

func serverGPULayers(device string) string {
	if layers := strings.TrimSpace(os.Getenv("SGREP_N_GPU_LAYERS")); layers != "" {
		return layers
	}
	if layers := strings.TrimSpace(os.Getenv("LLAMA_ARG_N_GPU_LAYERS")); layers != "" {
		return layers
	}
	if device == "none" {
		return "0"
	}
	return "99"
}

// Stop stops the llama.cpp server.
func (m *Manager) Stop() error {
	pid, err := m.readPID()
	if err != nil {
		if m.portResponding() {
			return fmt.Errorf("port %d is occupied but is not owned by sgrep", m.port)
		}
		return nil
	}

	if !processAlive(pid) {
		m.removePIDFile()
		return nil
	}
	if !m.ownsProcess(pid) {
		return fmt.Errorf("refusing to stop PID %d because it is not the managed llama-server on port %d", pid, m.port)
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

	// Wait briefly for graceful shutdown. Capability may disappear before the
	// owned process exits, so termination checks process liveness rather than
	// the embedding endpoint.
	time.Sleep(500 * time.Millisecond)
	if processAlive(pid) && m.ownsProcess(pid) {
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
	if !running && (!processAlive(pid) || !m.ownsProcess(pid)) {
		pid = 0
	}
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

// EnsureRunning starts or restarts the server until it is ready for embedding.
func (m *Manager) EnsureRunning() error {
	if m.IsReady() {
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
	lines := strings.SplitN(strings.TrimSpace(string(data)), "\n", 2)
	return strconv.Atoi(lines[0])
}

func (m *Manager) readProcessStartIdentity() (string, error) {
	data, err := os.ReadFile(m.pidPath())
	if err != nil {
		return "", err
	}
	lines := strings.SplitN(strings.TrimSpace(string(data)), "\n", 2)
	if len(lines) != 2 {
		return "", fmt.Errorf("managed server PID state has no process identity")
	}
	return strings.TrimSpace(lines[1]), nil
}

func processStartIdentity(pid int) string {
	output, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "lstart=").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(output))
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
		if m.IsReady() {
			return nil
		}
		time.Sleep(HealthInterval)
	}
	return fmt.Errorf("server failed to start within %v", StartupTimeout)
}

func (m *Manager) findLlamaServer() (string, error) {
	// Only accept unambiguous llama.cpp executable names. A generic "server"
	// cannot be validated safely from a persisted PID after this process exits.
	names := []string{"llama-server", "llama-server-metal"}

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
