package util

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFlightRecorder_StartStop(t *testing.T) {
	fr := NewFlightRecorder()

	// Should not be started initially
	if fr.IsStarted() {
		t.Error("recorder should not be started initially")
	}

	// Start
	if err := fr.Start(); err != nil {
		t.Fatalf("failed to start recorder: %v", err)
	}

	if !fr.IsStarted() {
		t.Error("recorder should be started after Start()")
	}

	// Double start should be no-op
	if err := fr.Start(); err != nil {
		t.Fatalf("double start should not error: %v", err)
	}

	// Stop
	fr.Stop()

	if fr.IsStarted() {
		t.Error("recorder should not be started after Stop()")
	}
}

func TestFlightRecorder_Snapshot(t *testing.T) {
	// Use temp directory for test
	tmpDir := t.TempDir()
	fr := &FlightRecorder{
		traceDir: tmpDir,
	}

	// Start the recorder
	if err := fr.Start(); err != nil {
		t.Fatalf("failed to start recorder: %v", err)
	}
	defer fr.Stop()

	// Do some work to generate trace data
	for i := 0; i < 1000; i++ {
		_ = make([]byte, 1024)
	}
	time.Sleep(10 * time.Millisecond)

	// Take a snapshot
	tracePath, err := fr.Snapshot("test-snapshot")
	if err != nil {
		t.Fatalf("failed to take snapshot: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(tracePath); os.IsNotExist(err) {
		t.Errorf("trace file should exist at %s", tracePath)
	}

	// Verify filename format
	filename := filepath.Base(tracePath)
	if !strings.HasPrefix(filename, "trace-test-snapshot-") {
		t.Errorf("trace filename should have correct prefix, got: %s", filename)
	}
	if !strings.HasSuffix(filename, ".out") {
		t.Errorf("trace filename should have .out suffix, got: %s", filename)
	}

	// Verify file has content
	info, err := os.Stat(tracePath)
	if err != nil {
		t.Fatalf("failed to stat trace file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("trace file should not be empty")
	}

	t.Logf("Trace file: %s (%d bytes)", tracePath, info.Size())
}

func TestFlightRecorder_SnapshotNotStarted(t *testing.T) {
	fr := NewFlightRecorder()

	_, err := fr.Snapshot("test")
	if err == nil {
		t.Error("snapshot should fail when recorder not started")
	}
}
