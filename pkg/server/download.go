package server

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	ModelURL  = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
	ModelName = "nomic-embed-text-v1.5.Q8_0.gguf"
	ModelSize = 137_000_000 // ~130MB approximate
)

// DownloadModel downloads the embedding model if not present.
func (m *Manager) DownloadModel(progress func(downloaded, total int64)) error {
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

	if err := downloadFile(tmpPath, ModelURL, progress); err != nil {
		return fmt.Errorf("download failed: %w", err)
	}

	// Rename to final location
	if err := os.Rename(tmpPath, modelPath); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	return nil
}

// ModelExists checks if the model is already downloaded.
func (m *Manager) ModelExists() bool {
	info, err := os.Stat(m.ModelPath())
	if err != nil {
		return false
	}
	return info.Size() > 100_000_000
}

// LlamaServerInstalled checks if llama-server is available.
func (m *Manager) LlamaServerInstalled() bool {
	_, err := m.findLlamaServer()
	return err == nil
}

func downloadFile(filepath string, url string, progress func(downloaded, total int64)) error {
	client := &http.Client{
		Timeout: 30 * time.Minute, // Large file, long timeout
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
		total = ModelSize
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

// Setup performs full setup: checks llama-server and downloads model.
func (m *Manager) Setup(verbose bool) error {
	// Check llama-server
	llamaPath, err := m.findLlamaServer()
	if err != nil {
		return err
	}
	if verbose {
		fmt.Printf("✓ Found llama-server: %s\n", llamaPath)
	}

	// Check/download model
	if m.ModelExists() {
		if verbose {
			fmt.Printf("✓ Model already downloaded: %s\n", m.ModelPath())
		}
		return nil
	}

	if verbose {
		fmt.Printf("Downloading embedding model (~130MB)...\n")
	}

	lastPct := -1
	err = m.DownloadModel(func(downloaded, total int64) {
		if verbose && total > 0 {
			pct := int(downloaded * 100 / total)
			if pct != lastPct && pct%10 == 0 {
				fmt.Printf("  %d%%\n", pct)
				lastPct = pct
			}
		}
	})

	if err != nil {
		return err
	}

	if verbose {
		fmt.Printf("✓ Model downloaded: %s\n", m.ModelPath())
	}

	return nil
}

// Cleanup removes downloaded models and stops server.
func (m *Manager) Cleanup() error {
	_ = m.Stop()

	modelsDir := m.ModelsDir()
	if err := os.RemoveAll(modelsDir); err != nil {
		return fmt.Errorf("failed to remove models: %w", err)
	}

	// Remove log and pid files
	_ = os.Remove(filepath.Join(m.sgrepHome, "server.log"))
	_ = os.Remove(filepath.Join(m.sgrepHome, "server.pid"))

	return nil
}
