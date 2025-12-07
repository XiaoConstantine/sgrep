package util

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/trace"
	"sync"
	"time"
)

// FlightRecorder wraps Go 1.25's runtime/trace.FlightRecorder for continuous
// low-overhead tracing with on-demand capture.
//
// Usage:
//
//	recorder := NewFlightRecorder()
//	recorder.Start()
//	defer recorder.Stop()
//
//	// Later, when something interesting happens:
//	if slowQuery {
//	    recorder.Snapshot("slow-query")
//	}
type FlightRecorder struct {
	mu       sync.Mutex
	recorder *trace.FlightRecorder
	started  bool
	traceDir string
}

// NewFlightRecorder creates a new flight recorder.
// Traces are saved to ~/.sgrep/traces/ by default.
func NewFlightRecorder() *FlightRecorder {
	traceDir := filepath.Join(os.Getenv("HOME"), ".sgrep", "traces")
	return &FlightRecorder{
		traceDir: traceDir,
	}
}

// Start begins continuous tracing in the background.
// The recorder maintains a ring buffer of recent trace data.
func (fr *FlightRecorder) Start() error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if fr.started {
		return nil
	}

	// Create trace directory
	if err := os.MkdirAll(fr.traceDir, 0755); err != nil {
		return fmt.Errorf("failed to create trace directory: %w", err)
	}

	// Create and start the flight recorder with a 10-second window
	cfg := trace.FlightRecorderConfig{
		MinAge:   10 * time.Second, // Keep at least 10 seconds of trace data
		MaxBytes: 10 * 1024 * 1024, // Cap at 10MB
	}
	recorder := trace.NewFlightRecorder(cfg)
	if err := recorder.Start(); err != nil {
		return fmt.Errorf("failed to start flight recorder: %w", err)
	}

	fr.recorder = recorder
	fr.started = true
	Debugf(DebugSummary, "FlightRecorder started, traces will be saved to %s", fr.traceDir)
	return nil
}

// Stop stops the flight recorder.
func (fr *FlightRecorder) Stop() {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.started || fr.recorder == nil {
		return
	}

	fr.recorder.Stop()
	fr.started = false
	Debugf(DebugSummary, "FlightRecorder stopped")
}

// Snapshot captures the current trace buffer to a file.
// The filename includes the reason and timestamp.
// Returns the path to the saved trace file.
func (fr *FlightRecorder) Snapshot(reason string) (string, error) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.started || fr.recorder == nil {
		return "", fmt.Errorf("flight recorder not started")
	}

	// Generate filename with timestamp
	timestamp := time.Now().Format("20060102-150405")
	filename := fmt.Sprintf("trace-%s-%s.out", reason, timestamp)
	tracePath := filepath.Join(fr.traceDir, filename)

	// Create trace file
	f, err := os.Create(tracePath)
	if err != nil {
		return "", fmt.Errorf("failed to create trace file: %w", err)
	}
	defer func() { _ = f.Close() }()

	// Write trace data
	if _, err := fr.recorder.WriteTo(f); err != nil {
		return "", fmt.Errorf("failed to write trace: %w", err)
	}

	Debugf(DebugSummary, "Trace snapshot saved to %s", tracePath)
	return tracePath, nil
}

// WriteTo writes the current trace buffer to the given writer.
func (fr *FlightRecorder) WriteTo(w io.Writer) (int64, error) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.started || fr.recorder == nil {
		return 0, fmt.Errorf("flight recorder not started")
	}

	return fr.recorder.WriteTo(w)
}

// IsStarted returns whether the recorder is currently running.
func (fr *FlightRecorder) IsStarted() bool {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	return fr.started
}

// Global flight recorder instance
var globalRecorder *FlightRecorder
var recorderOnce sync.Once

// GetFlightRecorder returns the global flight recorder instance.
func GetFlightRecorder() *FlightRecorder {
	recorderOnce.Do(func() {
		globalRecorder = NewFlightRecorder()
	})
	return globalRecorder
}

// StartGlobalRecorder starts the global flight recorder.
func StartGlobalRecorder() error {
	return GetFlightRecorder().Start()
}

// StopGlobalRecorder stops the global flight recorder.
func StopGlobalRecorder() {
	GetFlightRecorder().Stop()
}

// SnapshotGlobalRecorder captures a snapshot from the global recorder.
func SnapshotGlobalRecorder(reason string) (string, error) {
	return GetFlightRecorder().Snapshot(reason)
}
