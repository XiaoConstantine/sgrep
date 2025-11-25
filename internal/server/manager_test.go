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
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{port: mustPort(srv.URL), host: "localhost"}
	if !mgr.IsRunning() {
		t.Error("should be running")
	}
}

func TestManager_IsRunning_False(t *testing.T) {
	mgr := &Manager{port: 59999, host: "localhost"}
	if mgr.IsRunning() {
		t.Error("should not be running")
	}
}

func TestManager_IsRunning_NonOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	mgr := &Manager{port: mustPort(srv.URL), host: "localhost"}
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
	if running || pid != 12345 || port != 59998 {
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
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{sgrepHome: t.TempDir(), port: mustPort(srv.URL), host: "localhost"}
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
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{sgrepHome: t.TempDir(), port: mustPort(srv.URL), host: "localhost"}
	if err := mgr.EnsureRunning(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_WaitForReady_AlreadyUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	mgr := &Manager{port: mustPort(srv.URL), host: "localhost"}
	if err := mgr.waitForReady(); err != nil {
		t.Errorf("should succeed: %v", err)
	}
}

func TestManager_FindLlamaServer(t *testing.T) {
	mgr := &Manager{sgrepHome: t.TempDir()}
	// Just verify it doesn't panic; result depends on system
	_, _ = mgr.findLlamaServer()
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

func mustPort(url string) int {
	for i := len(url) - 1; i >= 0; i-- {
		if url[i] == ':' {
			p, _ := strconv.Atoi(url[i+1:])
			return p
		}
	}
	return 0
}
