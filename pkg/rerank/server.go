package rerank

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const (
	DefaultRerankerPort    = 8081
	DefaultHost            = "localhost"
	RerankerModelURL       = "https://huggingface.co/gpustack/jina-reranker-v2-base-multilingual-GGUF/resolve/main/jina-reranker-v2-base-multilingual-Q8_0.gguf"
	RerankerModelName      = "jina-reranker-v2-base-multilingual-Q8_0.gguf"
	RerankerModelSize      = 305_000_000 // ~305MB
	RerankerStartupTimeout = 30 * time.Second
	RerankerHealthInterval = 500 * time.Millisecond
	rerankerInstanceEnv    = "SGREP_RERANKER_INSTANCE"
)

type rerankerPIDState struct {
	Protocol          int    `json:"protocol"`
	PID               int    `json:"pid"`
	PGID              int    `json:"pgid"`
	Started           string `json:"started"`
	Executable        string `json:"executable"`
	ExecutableDevice  uint64 `json:"executable_device"`
	ExecutableInode   uint64 `json:"executable_inode"`
	Instance          string `json:"instance"`
	Control           string `json:"control"`
	Port              int    `json:"port"`
	SupervisorPID     int    `json:"supervisor_pid"`
	SupervisorStarted string `json:"supervisor_started"`
	GuardianPID       int    `json:"guardian_pid,omitempty"`
}

// RerankerManager handles llama.cpp reranker server lifecycle.
type RerankerManager struct {
	sgrepHome      string
	port           int
	host           string
	startupTimeout time.Duration      // test hook; zero uses RerankerStartupTimeout
	processCheck   func(pid int) bool // test hook; nil uses OS process inspection
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

// IsRunning reports whether the PID owned by this manager is alive and serves
// the reranker endpoint. A generic HTTP 200 from another service is not enough.
func (m *RerankerManager) IsRunning() bool {
	state, err := m.readPIDState()
	if err != nil || !rerankerProcessAlive(state.PID) || !m.ownsProcess(state.PID) {
		return false
	}
	if m.processCheck == nil {
		response, err := requestRerankerSupervisor(state, "status")
		if err != nil || !response.Running || response.PID != state.PID {
			return false
		}
	}
	return m.rerankerReady()
}

func (m *RerankerManager) rerankerReady() bool {
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

func (m *RerankerManager) portResponding() bool {
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

// Start starts the llama.cpp reranker server if not already running.
func (m *RerankerManager) Start() error {
	return m.withLifecycleLock("start", m.start)
}

func (m *RerankerManager) start(lifecycleLock *os.File) error {
	if err := m.cleanStalePID(); err != nil {
		return err
	}
	if adopted, err := m.adoptExistingSupervisor(); adopted || err != nil {
		return err
	}
	if m.portResponding() {
		return fmt.Errorf("port %d is occupied by a process that is not the managed reranker server; set SGREP_RERANKER_PORT", m.port)
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

	// Start the server
	logPath := filepath.Join(m.sgrepHome, "reranker.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer func() { _ = logFile.Close() }()

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

	instance, err := newRerankerInstanceToken()
	if err != nil {
		return fmt.Errorf("create reranker process identity: %w", err)
	}
	control, err := createRerankerControlPath(instance)
	if err != nil {
		return err
	}
	supervisorExecutable, err := m.findSupervisorExecutable()
	if err != nil {
		cleanupAbandonedControlPath(control, instance)
		return err
	}
	supervisor, supervisorDone, err := launchRerankerSupervisorWithLifecycleLock(supervisorExecutable, m.sgrepHome, control, instance, m.port, llamaPath, args, logFile, lifecycleLock)
	if err != nil {
		cleanupAbandonedControlPath(control, instance)
		return fmt.Errorf("start reranker supervisor: %w", err)
	}
	var state rerankerPIDState
	abortStart := func() {
		if state.Instance == instance {
			if _, stopErr := requestRerankerSupervisor(state, "stop"); stopErr == nil {
				return
			}
		}
		_ = supervisor.Process.Signal(syscall.SIGTERM)
	}

	state, err = waitForPublishedRerankerState(m, control, instance, supervisor.Process.Pid, supervisorDone)
	if err != nil {
		abortStart()
		return err
	}
	if !m.ownsProcess(state.PID) {
		abortStart()
		return fmt.Errorf("published reranker process identity did not validate")
	}
	response, err := requestRerankerSupervisor(state, "adopt")
	if err != nil {
		abortStart()
		return fmt.Errorf("adopt reranker supervisor: %w", err)
	}
	if !response.Running || response.PID != state.PID {
		abortStart()
		return fmt.Errorf("adopt reranker supervisor: unexpected response for PID %d", state.PID)
	}

	// Wait for server to be ready
	if err := m.waitForReady(); err != nil {
		abortStart()
		m.removePIDFileIfOwned(state)
		return err
	}
	response, err = requestRerankerSupervisor(state, "commit")
	if err != nil {
		abortStart()
		return fmt.Errorf("commit reranker supervisor adoption for PID %d: %w", state.PID, err)
	}
	if !response.Running || response.PID != state.PID {
		abortStart()
		return fmt.Errorf("commit reranker supervisor adoption: unexpected response for PID %d", state.PID)
	}

	return nil
}

// Stop stops the reranker server.
func (m *RerankerManager) Stop() error {
	return m.withLifecycleLock("stop", m.stop)
}

func (m *RerankerManager) stop(_ *os.File) error {
	state, err := m.readPIDState()
	if err != nil {
		if m.portResponding() {
			return fmt.Errorf("reranker running but no PID file found; kill manually on port %d", m.port)
		}
		return nil
	}

	rootAlive := rerankerProcessGenerationAlive(state.PID, state.Started)
	supervisorAlive := rerankerProcessGenerationAlive(state.SupervisorPID, state.SupervisorStarted)
	if !rootAlive {
		if supervisorAlive {
			response, err := requestRerankerSupervisor(state, "stop")
			if err != nil {
				return fmt.Errorf("stop stale reranker supervisor PID %d: %w", state.SupervisorPID, err)
			}
			if response.PID != state.PID || response.Running {
				return fmt.Errorf("stop stale reranker supervisor PID %d: unexpected response", state.SupervisorPID)
			}
		}
		m.removePIDFileIfOwned(state)
		cleanupAbandonedControlPath(state.Control, state.Instance)
		return nil
	}
	if !m.ownsProcess(state.PID) {
		return fmt.Errorf("refusing to stop PID %d because it is not the managed reranker server on port %d", state.PID, m.port)
	}
	if state.Control == "" || state.Instance == "" || state.SupervisorPID <= 0 || state.SupervisorStarted == "" {
		return fmt.Errorf("refusing to stop PID %d without its authenticated supervisor; stop it manually", state.PID)
	}
	if !supervisorAlive {
		return fmt.Errorf("refusing to stop PID %d because its authenticated supervisor generation is no longer running; stop it manually", state.PID)
	}
	response, err := requestRerankerSupervisor(state, "stop")
	if err != nil {
		return fmt.Errorf("stop managed reranker PID %d: %w", state.PID, err)
	}
	if response.PID != state.PID || response.Running {
		return fmt.Errorf("stop managed reranker PID %d: unexpected supervisor response", state.PID)
	}
	m.removePIDFileIfOwned(state)
	return nil
}

// Status returns reranker server status info.
func (m *RerankerManager) Status() (running bool, pid int, port int) {
	port = m.port
	running = m.IsRunning()
	pid, _ = m.readPID()
	if !running && (!rerankerProcessAlive(pid) || !m.ownsProcess(pid)) {
		pid = 0
	}
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
	return m.Start()
}

func (m *RerankerManager) adoptExistingSupervisor() (bool, error) {
	state, err := m.readPIDState()
	if err != nil || !rerankerProcessAlive(state.PID) || !m.ownsProcess(state.PID) {
		return false, nil
	}
	response, err := requestRerankerSupervisor(state, "adopt")
	if err != nil {
		return true, fmt.Errorf("adopt existing reranker supervisor: %w", err)
	}
	if !response.Running || response.PID != state.PID {
		return true, fmt.Errorf("adopt existing reranker supervisor: unexpected response for PID %d", state.PID)
	}
	if m.rerankerReady() {
		return true, m.commitSupervisorAdoption(state)
	}
	if readyErr := m.waitForReady(); readyErr != nil {
		_, stopErr := requestRerankerSupervisor(state, "stop")
		m.removePIDFileIfOwned(state)
		return true, errors.Join(readyErr, stopErr)
	}
	return true, m.commitSupervisorAdoption(state)
}

func (m *RerankerManager) commitSupervisorAdoption(state rerankerPIDState) error {
	response, err := requestRerankerSupervisor(state, "commit")
	if err != nil {
		err = fmt.Errorf("commit existing reranker supervisor adoption: %w", err)
	} else if !response.Running || response.PID != state.PID {
		err = fmt.Errorf("commit existing reranker supervisor adoption: unexpected response for PID %d", state.PID)
	}
	if err == nil {
		return nil
	}
	_, stopErr := requestRerankerSupervisor(state, "stop")
	m.removePIDFileIfOwned(state)
	return errors.Join(err, stopErr)
}

func (m *RerankerManager) findSupervisorExecutable() (supervisorExecutableRegistration, error) {
	registeredSupervisorExecutable.RLock()
	registered := registeredSupervisorExecutable.supervisorExecutableRegistration
	registeredSupervisorExecutable.RUnlock()
	if registered.Path == "" || registered.Device == 0 || registered.Inode == 0 || registered.Size <= 0 || registered.File == nil {
		return supervisorExecutableRegistration{}, fmt.Errorf("reranker supervision is unavailable: the host command must call rerank.RunSupervisorCommand before normal command dispatch")
	}
	info, err := registered.File.Stat()
	if err != nil {
		return supervisorExecutableRegistration{}, fmt.Errorf("inspect retained reranker supervisor %q: %w", registered.Path, err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || uint64(stat.Dev) != registered.Device || uint64(stat.Ino) != registered.Inode || info.Size() != registered.Size {
		return supervisorExecutableRegistration{}, fmt.Errorf("retained reranker supervisor %q changed identity", registered.Path)
	}
	return registered, nil
}

// ModelExists checks if the reranker model is already downloaded.
func (m *RerankerManager) ModelExists() bool {
	info, err := os.Stat(m.ModelPath())
	if err != nil {
		return false
	}
	return info.Size() > 100_000_000 // Should be > 100MB (jina-reranker-v2 is ~300MB)
}

// DownloadModel downloads the reranker model if not present.
func (m *RerankerManager) DownloadModel(progress func(downloaded, total int64)) error {
	modelPath := m.ModelPath()

	// Check if already exists
	if info, err := os.Stat(modelPath); err == nil {
		if info.Size() > 100_000_000 { // Sanity check: should be > 100MB
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

func (m *RerankerManager) withLifecycleLock(operation string, run func(*os.File) error) error {
	path := filepath.Join(m.sgrepHome, ".reranker.lock")
	fd, err := unix.Open(path, unix.O_CREAT|unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0600)
	if err != nil {
		return fmt.Errorf("%s reranker: open lifecycle lock: %w", operation, err)
	}
	lock := os.NewFile(uintptr(fd), path)
	defer func() { _ = lock.Close() }()
	info, err := lock.Stat()
	if err != nil {
		return fmt.Errorf("%s reranker: inspect lifecycle lock: %w", operation, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("%s reranker: lifecycle lock is not a regular file", operation)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || int(stat.Uid) != os.Getuid() || stat.Nlink != 1 || info.Mode().Perm()&0077 != 0 {
		return fmt.Errorf("%s reranker: lifecycle lock has unsafe ownership or permissions", operation)
	}
	for {
		err = unix.Flock(fd, unix.LOCK_EX)
		if !errors.Is(err, syscall.EINTR) {
			break
		}
	}
	if err != nil {
		return fmt.Errorf("%s reranker: acquire lifecycle lock: %w", operation, err)
	}
	// Do not explicitly unlock: Start may pass a duplicate of this open file
	// description to the supervisor. Closing lets the supervisor retain the
	// lock if the launcher dies before publishing lifecycle state.
	return run(lock)
}

func (m *RerankerManager) readPID() (int, error) {
	state, err := m.readPIDState()
	return state.PID, err
}

func (m *RerankerManager) readPIDState() (rerankerPIDState, error) {
	data, err := os.ReadFile(m.pidPath())
	if err != nil {
		return rerankerPIDState{}, err
	}
	var state rerankerPIDState
	if json.Unmarshal(data, &state) == nil && state.PID > 0 {
		return state, nil
	}
	// Parse legacy one-line PID files for status and safe cleanup. They have no
	// process-generation identity, so ownsProcess deliberately rejects them.
	lines := strings.SplitN(strings.TrimSpace(string(data)), "\n", 2)
	pid, parseErr := strconv.Atoi(lines[0])
	if parseErr != nil {
		return rerankerPIDState{}, parseErr
	}
	return rerankerPIDState{PID: pid}, nil
}

func (m *RerankerManager) readProcessStartIdentity() (string, error) {
	state, err := m.readPIDState()
	if err != nil {
		return "", err
	}
	if state.Started == "" {
		return "", fmt.Errorf("managed reranker PID state has no process identity")
	}
	return state.Started, nil
}

func (m *RerankerManager) writePIDState(state rerankerPIDState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(m.sgrepHome, ".reranker-pid-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()
	if err := tmp.Chmod(0600); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, m.pidPath())
}

func (m *RerankerManager) removePIDFile() {
	_ = os.Remove(m.pidPath())
}

func (m *RerankerManager) removePIDFileIfOwned(want rerankerPIDState) {
	current, err := m.readPIDState()
	if err != nil {
		return
	}
	if current.PID == want.PID && current.Instance == want.Instance && current.Control == want.Control && current.SupervisorPID == want.SupervisorPID {
		m.removePIDFile()
	}
}

func (m *RerankerManager) cleanStalePID() error {
	state, err := m.readPIDState()
	if err != nil {
		return nil
	}
	if rerankerProcessGenerationAlive(state.PID, state.Started) {
		return nil
	}
	if rerankerProcessGenerationAlive(state.SupervisorPID, state.SupervisorStarted) {
		response, err := requestRerankerSupervisor(state, "stop")
		if err != nil {
			return fmt.Errorf("clean stale reranker supervisor PID %d: %w", state.SupervisorPID, err)
		}
		if response.PID != state.PID || response.Running {
			return fmt.Errorf("clean stale reranker supervisor PID %d: unexpected response", state.SupervisorPID)
		}
	}
	m.removePIDFileIfOwned(state)
	cleanupAbandonedControlPath(state.Control, state.Instance)
	return nil
}

func rerankerProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

func rerankerProcessGenerationAlive(pid int, started string) bool {
	if pid <= 0 || started == "" || !rerankerProcessAlive(pid) {
		return false
	}
	return rerankerProcessStartIdentity(pid) == started
}

func (m *RerankerManager) ownsProcess(pid int) bool {
	if m.processCheck != nil {
		return m.processCheck(pid)
	}
	state, err := m.readPIDState()
	if err != nil || state.Protocol != supervisorProtocolVersion || state.PID != pid || state.Started == "" || state.Executable == "" || state.ExecutableDevice == 0 || state.ExecutableInode == 0 || state.Instance == "" || state.Port != m.port {
		return false
	}
	current, err := inspectRerankerProcess(pid)
	if err != nil || current.Started != state.Started || current.Executable != state.Executable || current.ExecutableDevice != state.ExecutableDevice || current.ExecutableInode != state.ExecutableInode {
		return false
	}
	args, err := rerankerProcessArgs(pid)
	if err != nil || !hasExactArg(args, "--port", strconv.Itoa(m.port)) || !hasExactArg(args, "--pooling", "rank") {
		return false
	}
	return rerankerProcessHasInstance(pid, state.Instance)
}

func rerankerProcessStartIdentity(pid int) string {
	return rerankerNativeProcessStartIdentity(pid)
}

func inspectRerankerProcess(pid int) (rerankerPIDState, error) {
	started := rerankerProcessStartIdentity(pid)
	if started == "" {
		return rerankerPIDState{}, fmt.Errorf("read process start identity")
	}
	executable := rerankerProcessExecutable(pid)
	if executable == "" {
		return rerankerPIDState{}, fmt.Errorf("read process executable")
	}
	pgid, err := syscall.Getpgid(pid)
	if err != nil {
		return rerankerPIDState{}, err
	}
	device, inode, err := rerankerProcessExecutableIdentity(pid)
	if err != nil {
		return rerankerPIDState{}, fmt.Errorf("read process executable identity: %w", err)
	}
	return rerankerPIDState{
		Protocol:         supervisorProtocolVersion,
		PID:              pid,
		PGID:             pgid,
		Started:          started,
		Executable:       executable,
		ExecutableDevice: device,
		ExecutableInode:  inode,
	}, nil
}

func rerankerProcessExecutable(pid int) string {
	if runtime.GOOS == "linux" {
		path, err := os.Readlink(filepath.Join("/proc", strconv.Itoa(pid), "exe"))
		if err == nil {
			path = strings.TrimSuffix(path, " (deleted)")
			return canonicalExecutablePath(path)
		}
	}
	output, err := exec.Command("/bin/ps", "-p", strconv.Itoa(pid), "-o", "comm=").Output()
	if err != nil {
		return ""
	}
	// On Darwin, ps reports the launch path. Keep that stable path rather than
	// re-resolving a Homebrew symlink whose target may change during upgrades.
	return absoluteExecutablePath(strings.TrimSpace(string(output)))
}

func canonicalExecutablePath(path string) string {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	return absoluteExecutablePath(path)
}

func absoluteExecutablePath(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return filepath.Clean(path)
}

func rerankerProcessArgs(pid int) ([]string, error) {
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "cmdline"))
		if err != nil {
			return nil, err
		}
		data = []byte(strings.TrimRight(string(data), "\x00"))
		if len(data) == 0 {
			return nil, fmt.Errorf("process command line is empty")
		}
		return strings.Split(string(data), "\x00"), nil
	}
	output, err := exec.Command("/bin/ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	if err != nil {
		return nil, err
	}
	// The flags used for ownership have whitespace-free values. This fallback
	// deliberately matches only complete fields, never substrings.
	return strings.Fields(strings.TrimSpace(string(output))), nil
}

func hasExactArg(args []string, key, value string) bool {
	for i, field := range args {
		if field == key+"="+value {
			return true
		}
		if field == key && i+1 < len(args) && args[i+1] == value {
			return true
		}
	}
	return false
}

func rerankerProcessHasInstance(pid int, instance string) bool {
	want := rerankerInstanceEnv + "=" + instance
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "environ"))
		if err != nil {
			return false
		}
		for _, entry := range strings.Split(strings.TrimRight(string(data), "\x00"), "\x00") {
			if entry == want {
				return true
			}
		}
		return false
	}
	output, err := exec.Command("/bin/ps", "eww", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	if err != nil {
		return false
	}
	for _, field := range strings.Fields(string(output)) {
		if field == want {
			return true
		}
	}
	return false
}

func newRerankerInstanceToken() (string, error) {
	var token [32]byte
	if _, err := rand.Read(token[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(token[:]), nil
}

func rerankerControlPath(instance string) string {
	suffix := instance
	if len(suffix) > 24 {
		suffix = suffix[:24]
	}
	// Darwin limits Unix socket paths to roughly 100 bytes; SGREP_HOME and the
	// system temp directory can both exceed that. A private, unpredictable
	// per-instance directory also closes the pre-chmod socket exposure window.
	return filepath.Join("/tmp", fmt.Sprintf("sgrep-reranker-%d-%s", os.Getuid(), suffix), "control.sock")
}

func createRerankerControlPath(instance string) (string, error) {
	control := rerankerControlPath(instance)
	dir := filepath.Dir(control)
	if err := os.Mkdir(dir, 0700); err != nil {
		return "", fmt.Errorf("create private reranker control directory: %w", err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		_ = os.Remove(dir)
		return "", fmt.Errorf("secure reranker control directory: %w", err)
	}
	info, err := os.Lstat(dir)
	if err != nil || !info.IsDir() || info.Mode().Perm() != 0700 {
		_ = os.Remove(dir)
		return "", fmt.Errorf("reranker control directory is not a private directory")
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || int(stat.Uid) != os.Getuid() {
		_ = os.Remove(dir)
		return "", fmt.Errorf("reranker control directory has the wrong owner")
	}
	return control, nil
}

func cleanupAbandonedControlPath(control, instance string) {
	if control == "" || instance == "" || filepath.Clean(control) != rerankerControlPath(instance) {
		return
	}
	dir := filepath.Dir(control)
	info, err := os.Lstat(dir)
	if err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || int(stat.Uid) != os.Getuid() {
		return
	}
	if socket, err := os.Lstat(control); err == nil && socket.Mode()&os.ModeSocket != 0 {
		_ = os.Remove(control)
	}
	if staged, err := os.Lstat(stagedSupervisorPath(control)); err == nil && staged.Mode().IsRegular() {
		_ = os.Remove(stagedSupervisorPath(control))
	}
	_ = os.Remove(dir)
}

func waitForPublishedRerankerState(m *RerankerManager, control, instance string, supervisorPID int, done <-chan error) (rerankerPIDState, error) {
	deadline := time.Now().Add(5 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		state, err := m.readPIDState()
		if err == nil && state.Protocol == supervisorProtocolVersion && state.Control == control && state.Instance == instance && state.SupervisorPID == supervisorPID && state.SupervisorStarted != "" {
			return state, nil
		}
		if err != nil {
			lastErr = err
		}
		select {
		case err := <-done:
			if err == nil {
				err = fmt.Errorf("supervisor exited before reporting a reranker process")
			}
			return rerankerPIDState{}, fmt.Errorf("reranker supervisor exited: %w", err)
		case <-time.After(25 * time.Millisecond):
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no status response")
	}
	return rerankerPIDState{}, fmt.Errorf("reranker supervisor did not publish state: %w", lastErr)
}

func (m *RerankerManager) waitForReady() error {
	timeout := m.startupTimeout
	if timeout <= 0 {
		timeout = RerankerStartupTimeout
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if m.IsRunning() {
			return nil
		}
		time.Sleep(RerankerHealthInterval)
	}
	return fmt.Errorf("reranker server failed to start within %v", timeout)
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
