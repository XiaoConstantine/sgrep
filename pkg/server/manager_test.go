package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestNewManager(t *testing.T) {
	t.Setenv("SGREP_HOME", t.TempDir())
	mgr, err := NewManager()
	if err != nil {
		t.Fatal(err)
	}
	if mgr.port != DefaultPort || mgr.host != DefaultHost {
		t.Errorf("got port=%d host=%s", mgr.port, mgr.host)
	}
}

func TestNewManager_CustomPort(t *testing.T) {
	t.Setenv("SGREP_HOME", t.TempDir())
	t.Setenv("SGREP_PORT", "9090")
	mgr, _ := NewManager()
	if mgr.port != 9090 {
		t.Errorf("got %d", mgr.port)
	}
}

func TestNewManager_InvalidPort(t *testing.T) {
	t.Setenv("SGREP_HOME", t.TempDir())
	t.Setenv("SGREP_PORT", "bad")
	mgr, _ := NewManager()
	if mgr.port != DefaultPort {
		t.Errorf("should default to %d, got %d", DefaultPort, mgr.port)
	}
}

func TestManager_IsRunning(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusOK)
	defer closeServer()
	if !mgr.IsRunning() {
		t.Error("should be running")
	}
}

func TestManager_IncompatibleServerIsRunningButNotReady(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusOK)
	defer closeServer()
	mgr.launchCheck = func(int) bool { return false }

	if !mgr.IsRunning() {
		t.Fatal("healthy managed server should still be reported as running")
	}
	if mgr.IsReady() {
		t.Fatal("server with incompatible launch arguments should not be ready")
	}
}

func TestManager_IsRunning_False(t *testing.T) {
	mgr := &Manager{port: 59999, host: "localhost"}
	if mgr.IsRunning() {
		t.Error("should not be running")
	}
}

func TestManager_IsRunning_NonOK(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusInternalServerError)
	defer closeServer()
	if mgr.IsRunning() {
		t.Error("500 should not count as running")
	}
}

func TestManager_Paths(t *testing.T) {
	mgr := &Manager{sgrepHome: "/home/test", port: 8080, host: "localhost"}
	if mgr.ModelPath() != "/home/test/models/nomic-embed-text-v1.5.Q8_0.gguf" {
		t.Error("ModelPath")
	}
	if mgr.ModelsDir() != "/home/test/models" {
		t.Error("ModelsDir")
	}
	if mgr.Endpoint() != "http://localhost:8080" {
		t.Error("Endpoint")
	}
	if mgr.healthURL() != "http://localhost:8080/health" {
		t.Error("healthURL")
	}
	if mgr.pidPath() != "/home/test/server.pid" {
		t.Error("pidPath")
	}
}

func TestManager_Status(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "server.pid"), []byte("12345"), 0644)
	mgr := &Manager{sgrepHome: dir, port: 59998, host: "localhost"}

	running, pid, port := mgr.Status()
	if running || pid != 0 || port != 59998 {
		t.Errorf("got running=%v pid=%d port=%d", running, pid, port)
	}
}

func TestManager_PID(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "server.pid")
	mgr := &Manager{sgrepHome: dir}

	// No file
	if _, err := mgr.readPID(); err == nil {
		t.Error("should error on missing")
	}

	// Valid
	_ = os.WriteFile(pidFile, []byte("999"), 0644)
	pid, err := mgr.readPID()
	if err != nil || pid != 999 {
		t.Errorf("got %d %v", pid, err)
	}

	// Invalid
	_ = os.WriteFile(pidFile, []byte("bad"), 0644)
	if _, err := mgr.readPID(); err == nil {
		t.Error("should error on bad content")
	}

	// Remove
	mgr.removePIDFile()
	if _, err := os.Stat(pidFile); !os.IsNotExist(err) {
		t.Error("file should be removed")
	}
}

func TestManager_CleanStalePID(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "server.pid")
	mgr := &Manager{sgrepHome: dir}

	// No file - no panic
	mgr.cleanStalePID()

	// Dead process
	_ = os.WriteFile(pidFile, []byte("99999999"), 0644)
	mgr.cleanStalePID()
	if _, err := os.Stat(pidFile); !os.IsNotExist(err) {
		t.Error("stale PID should be cleaned")
	}

	// Live process (current)
	_ = os.WriteFile(pidFile, []byte(strconv.Itoa(os.Getpid())), 0644)
	mgr.cleanStalePID()
	if _, err := os.Stat(pidFile); os.IsNotExist(err) {
		t.Error("live PID should not be cleaned")
	}
}

func TestManager_Start_AlreadyRunning(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusOK)
	defer closeServer()
	if err := mgr.Start(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_Start_NoModel(t *testing.T) {
	mgr := &Manager{sgrepHome: t.TempDir(), port: 59997, host: "localhost"}
	err := mgr.Start()
	if err == nil {
		t.Error("should fail without model")
	}
}

func TestManager_Stop_NoServer(t *testing.T) {
	mgr := &Manager{sgrepHome: t.TempDir(), port: 59996, host: "localhost"}
	if err := mgr.Stop(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_Stop_DeadPID(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "server.pid"), []byte("99999999"), 0644)
	mgr := &Manager{sgrepHome: dir, port: 59995, host: "localhost"}
	if err := mgr.Stop(); err != nil {
		t.Errorf("should handle dead: %v", err)
	}
}

func TestManager_Stop_RefusesUnownedLivePID(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "server.pid"), []byte(strconv.Itoa(os.Getpid())), 0644); err != nil {
		t.Fatal(err)
	}
	mgr := &Manager{sgrepHome: dir, port: 59994, host: "localhost", processCheck: func(int) bool { return false }}
	if err := mgr.Stop(); err == nil {
		t.Fatal("Stop accepted an unowned live PID")
	}
}

func TestManager_Stop_RunningNoPID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{sgrepHome: t.TempDir(), port: mustPort(srv.URL), host: "localhost"}
	if err := mgr.Stop(); err == nil {
		t.Error("should error when running but no PID")
	}
}

func TestManager_EnsureRunning_AlreadyUp(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusOK)
	defer closeServer()
	if err := mgr.EnsureRunning(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_WaitForReady_AlreadyUp(t *testing.T) {
	mgr, closeServer := runningTestManager(t, http.StatusOK)
	defer closeServer()
	if err := mgr.waitForReady(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_FindLlamaServer(t *testing.T) {
	mgr := &Manager{sgrepHome: t.TempDir()}
	// Just verify it doesn't panic; result depends on system
	_, _ = mgr.findLlamaServer()
}

func TestManager_LaunchConfigs_DefaultFallback(t *testing.T) {
	t.Setenv("SGREP_DEVICE", "")
	t.Setenv("LLAMA_ARG_DEVICE", "")
	t.Setenv("SGREP_N_GPU_LAYERS", "")
	t.Setenv("LLAMA_ARG_N_GPU_LAYERS", "")

	mgr := &Manager{}
	configs := mgr.launchConfigs()
	if len(configs) != 2 {
		t.Fatalf("got %d configs, want 2", len(configs))
	}
	if configs[0].device != "" || configs[0].gpuLayers != "99" {
		t.Fatalf("unexpected primary config: %+v", configs[0])
	}
	if configs[1].device != "none" || configs[1].gpuLayers != "0" {
		t.Fatalf("unexpected fallback config: %+v", configs[1])
	}
}

func TestManager_LaunchConfigs_ExplicitDevice(t *testing.T) {
	t.Setenv("SGREP_DEVICE", "none")
	t.Setenv("SGREP_N_GPU_LAYERS", "")

	mgr := &Manager{}
	configs := mgr.launchConfigs()
	if len(configs) != 1 {
		t.Fatalf("got %d configs, want 1", len(configs))
	}
	if configs[0].device != "none" || configs[0].gpuLayers != "0" {
		t.Fatalf("unexpected config: %+v", configs[0])
	}
}

func TestManager_BuildArgs_UsesDeviceAndGPULayers(t *testing.T) {
	mgr := &Manager{port: 8080, host: "localhost"}
	args := mgr.buildArgs("/tmp/model.gguf", 8, 32, 8192, launchConfig{device: "none", gpuLayers: "0"})

	if !containsPair(args, "-ngl", "0") {
		t.Fatalf("args missing gpu layers override: %v", args)
	}
	if !containsPair(args, "--device", "none") {
		t.Fatalf("args missing device override: %v", args)
	}
}

func TestManager_BuildArgs_UsesPerSlotContextForBatchGeometry(t *testing.T) {
	mgr := &Manager{port: 8080, host: "localhost"}
	args := mgr.buildArgs("/tmp/model.gguf", 16, 16, 20480, launchConfig{gpuLayers: "0"})

	if !containsPair(args, "-b", "1280") || !containsPair(args, "-ub", "1280") {
		t.Fatalf("args do not size embedding batches to the per-slot context: %v", args)
	}
}

func TestLaunchResourcesUseConfiguredContext(t *testing.T) {
	t.Setenv("SGREP_CONTEXT_TOKENS", "1280")
	threads, slots, contextSize := launchResources(16)
	if threads != 16 || slots != 16 || contextSize != 20480 {
		t.Fatalf("launchResources(16) = %d, %d, %d; want 16, 16, 20480", threads, slots, contextSize)
	}
}

func TestLaunchArgsCompatibleRequiresCurrentBatchGeometry(t *testing.T) {
	current := []string{"llama-server", "-c", "20480", "-b", "1280", "-ub", "1280"}
	if !launchArgsCompatible(current, 16, 20480) {
		t.Fatalf("current launch arguments reported incompatible: %v", current)
	}

	for _, args := range [][]string{
		{"llama-server", "-c", "20480", "-b", "2048", "-ub", "1280"},
		{"llama-server", "-c", "20480", "-b", "1280", "-ub", "2048"},
	} {
		if launchArgsCompatible(args, 16, 20480) {
			t.Fatalf("old batch geometry reported compatible: %v", args)
		}
	}
}

func TestContainsArgValue(t *testing.T) {
	args := []string{"llama-server", "-c", "20480", "--port=8080"}
	if !containsArgValue(args, "-c", "20480") {
		t.Fatalf("expected context argument in %v", args)
	}
	if containsArgValue(args, "-c", "8192") {
		t.Fatalf("unexpected old context argument in %v", args)
	}
	if !containsArgValue(args, "--port", "8080") {
		t.Fatalf("expected equals-form port argument in %v", args)
	}
}

func TestManager_ModelExists(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{sgrepHome: dir}

	if mgr.ModelExists() {
		t.Error("should not exist")
	}

	// Create small file
	modelsDir := filepath.Join(dir, "models")
	_ = os.MkdirAll(modelsDir, 0755)
	_ = os.WriteFile(mgr.ModelPath(), []byte("small"), 0644)
	if mgr.ModelExists() {
		t.Error("small file should not count")
	}
}

func TestGetSgrepHome(t *testing.T) {
	t.Setenv("SGREP_HOME", "/custom")
	h, _ := getSgrepHome()
	if h != "/custom" {
		t.Error("env not used")
	}

	t.Setenv("SGREP_HOME", "")
	h, _ = getSgrepHome()
	home, _ := os.UserHomeDir()
	if h != filepath.Join(home, ".sgrep") {
		t.Error("default wrong")
	}
}

func runningTestManager(t *testing.T, embeddingStatus int) (*Manager, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embedding" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(embeddingStatus)
	}))
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "server.pid"), []byte(strconv.Itoa(os.Getpid())), 0644); err != nil {
		srv.Close()
		t.Fatal(err)
	}
	return &Manager{
		sgrepHome: dir,
		port:      mustPort(srv.URL),
		host:      "localhost",
		processCheck: func(pid int) bool {
			return pid == os.Getpid()
		},
	}, srv.Close
}

func mustPort(url string) int {
	for i := len(url) - 1; i >= 0; i-- {
		if url[i] == ':' {
			p, _ := strconv.Atoi(url[i+1:])
			return p
		}
	}
	return 0
}

func containsPair(args []string, key, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == key && args[i+1] == value {
			return true
		}
	}
	return false
}
