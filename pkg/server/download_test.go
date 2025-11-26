package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestManager_ModelExists_True(t *testing.T) {
	tmpDir := t.TempDir()
	modelsDir := filepath.Join(tmpDir, "models")
	_ = os.MkdirAll(modelsDir, 0755)

	modelPath := filepath.Join(modelsDir, "nomic-embed-text-v1.5.Q8_0.gguf")
	data := make([]byte, 150_000_000)
	_ = os.WriteFile(modelPath, data, 0644)

	mgr := &Manager{sgrepHome: tmpDir}

	if !mgr.ModelExists() {
		t.Error("expected ModelExists to return true")
	}
}

func TestManager_ModelExists_False(t *testing.T) {
	tmpDir := t.TempDir()
	mgr := &Manager{sgrepHome: tmpDir}

	if mgr.ModelExists() {
		t.Error("expected ModelExists to return false")
	}
}

func TestManager_ModelExists_TooSmall(t *testing.T) {
	tmpDir := t.TempDir()
	modelsDir := filepath.Join(tmpDir, "models")
	_ = os.MkdirAll(modelsDir, 0755)

	modelPath := filepath.Join(modelsDir, "nomic-embed-text-v1.5.Q8_0.gguf")
	_ = os.WriteFile(modelPath, []byte("small file"), 0644)

	mgr := &Manager{sgrepHome: tmpDir}

	if mgr.ModelExists() {
		t.Error("expected ModelExists to return false for small file")
	}
}

func TestManager_LlamaServerInstalled(t *testing.T) {
	tmpDir := t.TempDir()
	mgr := &Manager{sgrepHome: tmpDir}

	_ = mgr.LlamaServerInstalled()
}

func TestDownloadFile_Success(t *testing.T) {
	testData := []byte("test model data for download")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "28")
		_, _ = w.Write(testData)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	var progressCalls int
	var lastDownloaded, lastTotal int64

	err := downloadFile(destPath, server.URL, func(downloaded, total int64) {
		progressCalls++
		lastDownloaded = downloaded
		lastTotal = total
	})

	if err != nil {
		t.Fatalf("downloadFile failed: %v", err)
	}

	content, _ := os.ReadFile(destPath)
	if string(content) != string(testData) {
		t.Errorf("expected %s, got %s", string(testData), string(content))
	}

	if progressCalls == 0 {
		t.Error("expected progress callback to be called")
	}

	if lastDownloaded != int64(len(testData)) {
		t.Errorf("expected lastDownloaded %d, got %d", len(testData), lastDownloaded)
	}

	if lastTotal != 28 {
		t.Errorf("expected lastTotal 28, got %d", lastTotal)
	}
}

func TestDownloadFile_NoContentLength(t *testing.T) {
	testData := []byte("test model data")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(testData)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	err := downloadFile(destPath, server.URL, func(downloaded, total int64) {
		// Content-Length not set, so total should fall back to ModelSize
		_ = total
	})

	if err != nil {
		t.Fatalf("downloadFile failed: %v", err)
	}

	content, _ := os.ReadFile(destPath)
	if string(content) != string(testData) {
		t.Errorf("expected %s, got %s", string(testData), string(content))
	}
}

func TestDownloadFile_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	err := downloadFile(destPath, server.URL, nil)
	if err == nil {
		t.Error("expected error for HTTP 404")
	}
}

func TestDownloadFile_NetworkError(t *testing.T) {
	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	err := downloadFile(destPath, "http://localhost:19999/nonexistent", nil)
	if err == nil {
		t.Error("expected error for network failure")
	}
}

func TestDownloadFile_NilProgress(t *testing.T) {
	testData := []byte("test data")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(testData)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	err := downloadFile(destPath, server.URL, nil)
	if err != nil {
		t.Fatalf("downloadFile failed: %v", err)
	}
}

func TestDownloadFile_LargeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large file test in short mode")
	}

	largeData := make([]byte, 1024*1024)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1048576")
		_, _ = io.Copy(w, &dataReader{data: largeData, offset: 0})
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	destPath := filepath.Join(tmpDir, "test.gguf")

	var totalDownloaded int64
	err := downloadFile(destPath, server.URL, func(downloaded, total int64) {
		totalDownloaded = downloaded
	})

	if err != nil {
		t.Fatalf("downloadFile failed: %v", err)
	}

	if totalDownloaded != int64(len(largeData)) {
		t.Errorf("expected %d bytes downloaded, got %d", len(largeData), totalDownloaded)
	}
}

func TestManager_DownloadModel_AlreadyExists(t *testing.T) {
	tmpDir := t.TempDir()
	modelsDir := filepath.Join(tmpDir, "models")
	_ = os.MkdirAll(modelsDir, 0755)

	modelPath := filepath.Join(modelsDir, "nomic-embed-text-v1.5.Q8_0.gguf")
	data := make([]byte, 150_000_000)
	_ = os.WriteFile(modelPath, data, 0644)

	mgr := &Manager{sgrepHome: tmpDir}

	err := mgr.DownloadModel(nil)
	if err != nil {
		t.Errorf("DownloadModel should not error when model exists: %v", err)
	}
}

func TestManager_Cleanup(t *testing.T) {
	tmpDir := t.TempDir()
	modelsDir := filepath.Join(tmpDir, "models")
	_ = os.MkdirAll(modelsDir, 0755)

	modelPath := filepath.Join(modelsDir, "nomic-embed-text-v1.5.Q8_0.gguf")
	_ = os.WriteFile(modelPath, []byte("model"), 0644)

	logPath := filepath.Join(tmpDir, "server.log")
	_ = os.WriteFile(logPath, []byte("logs"), 0644)

	pidPath := filepath.Join(tmpDir, "server.pid")
	_ = os.WriteFile(pidPath, []byte("12345"), 0644)

	mgr := &Manager{
		sgrepHome: tmpDir,
		port:      19995,
		host:      "localhost",
	}

	err := mgr.Cleanup()
	if err != nil {
		t.Errorf("Cleanup failed: %v", err)
	}

	if _, err := os.Stat(modelsDir); !os.IsNotExist(err) {
		t.Error("expected models directory to be removed")
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Error("expected log file to be removed")
	}
	if _, err := os.Stat(pidPath); !os.IsNotExist(err) {
		t.Error("expected PID file to be removed")
	}
}

type dataReader struct {
	data   []byte
	offset int
}

func (r *dataReader) Read(p []byte) (n int, err error) {
	if r.offset >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.offset:])
	r.offset += n
	return n, nil
}
