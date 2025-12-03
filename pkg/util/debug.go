package util

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"
)

// DebugLevel controls the verbosity of debug output.
type DebugLevel int

const (
	DebugOff      DebugLevel = 0 // No debug output
	DebugSummary  DebugLevel = 1 // Stage-level timing summaries
	DebugDetailed DebugLevel = 2 // Per-operation detailed timing
)

// Global debug state
var (
	globalDebugLevel  DebugLevel = DebugOff
	globalDebugWriter io.Writer  = os.Stderr
	globalDebugMu     sync.RWMutex
)

// SetDebugLevel sets the global debug level.
func SetDebugLevel(level DebugLevel) {
	globalDebugMu.Lock()
	defer globalDebugMu.Unlock()
	globalDebugLevel = level
}

// GetDebugLevel returns the current global debug level.
func GetDebugLevel() DebugLevel {
	globalDebugMu.RLock()
	defer globalDebugMu.RUnlock()
	return globalDebugLevel
}

// SetDebugWriter sets the global debug output writer.
func SetDebugWriter(w io.Writer) {
	globalDebugMu.Lock()
	defer globalDebugMu.Unlock()
	globalDebugWriter = w
}

// GetDebugWriter returns the current debug output writer.
func GetDebugWriter() io.Writer {
	globalDebugMu.RLock()
	defer globalDebugMu.RUnlock()
	return globalDebugWriter
}

// Debugf prints a debug message if the current level >= minLevel.
func Debugf(minLevel DebugLevel, format string, args ...interface{}) {
	globalDebugMu.RLock()
	level := globalDebugLevel
	writer := globalDebugWriter
	globalDebugMu.RUnlock()

	if level >= minLevel {
		_, _ = fmt.Fprintf(writer, "[DEBUG] "+format+"\n", args...)
	}
}

// Timer measures the duration of an operation.
type Timer struct {
	name  string
	start time.Time
}

// NewTimer creates and starts a new timer.
func NewTimer(name string) *Timer {
	return &Timer{
		name:  name,
		start: time.Now(),
	}
}

// Stop stops the timer and returns the elapsed duration.
func (t *Timer) Stop() time.Duration {
	return time.Since(t.start)
}

// StopAndLog stops the timer and logs at the specified level.
func (t *Timer) StopAndLog(minLevel DebugLevel) time.Duration {
	elapsed := t.Stop()
	Debugf(minLevel, "%s: %v", t.name, elapsed)
	return elapsed
}

// TimingEntry records a single timed operation.
type TimingEntry struct {
	Name     string
	Duration time.Duration
	Count    int64 // For batch operations, how many items
}

// TimingStats collects timing statistics for multiple operations.
type TimingStats struct {
	mu     sync.Mutex
	level  DebugLevel
	writer io.Writer
	stages map[string]*stageStats
	order  []string // Preserve insertion order for stages
}

type stageStats struct {
	totalDuration time.Duration
	count         int64
	durations     []time.Duration // For percentile calculation in detailed mode
}

// NewTimingStats creates a new timing stats collector.
func NewTimingStats(level DebugLevel) *TimingStats {
	return &TimingStats{
		level:  level,
		writer: GetDebugWriter(),
		stages: make(map[string]*stageStats),
	}
}

// RecordStage records timing for a named stage.
func (s *TimingStats) RecordStage(name string, d time.Duration, count int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stats, exists := s.stages[name]
	if !exists {
		stats = &stageStats{}
		s.stages[name] = stats
		s.order = append(s.order, name)
	}

	stats.totalDuration += d
	stats.count += count
	if s.level >= DebugDetailed {
		stats.durations = append(stats.durations, d)
	}
}

// RecordOp records a single operation (for detailed logging).
func (s *TimingStats) RecordOp(name string, d time.Duration, count int64) {
	s.RecordStage(name, d, count)

	// Log immediately in detailed mode
	if s.level >= DebugDetailed {
		if count > 1 {
			Debugf(DebugDetailed, "%s: %d items in %v (%.2f/item)",
				name, count, d, float64(d.Microseconds())/float64(count)/1000)
		} else {
			Debugf(DebugDetailed, "%s: %v", name, d)
		}
	}
}

// Summary returns a formatted summary of all timing stats.
func (s *TimingStats) Summary() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.stages) == 0 {
		return ""
	}

	// Calculate total time for percentage
	var totalTime time.Duration
	for _, stats := range s.stages {
		totalTime += stats.totalDuration
	}

	// Find bottleneck
	var bottleneck string
	var maxDuration time.Duration
	for name, stats := range s.stages {
		if stats.totalDuration > maxDuration {
			maxDuration = stats.totalDuration
			bottleneck = name
		}
	}

	// Build summary string
	result := "Pipeline stages:\n"
	for _, name := range s.order {
		stats := s.stages[name]
		pct := float64(stats.totalDuration) / float64(totalTime) * 100
		avg := time.Duration(0)
		if stats.count > 0 {
			avg = stats.totalDuration / time.Duration(stats.count)
		}
		result += fmt.Sprintf("  %-15s %8v (%d ops, %v avg, %.0f%%)\n",
			name+":", stats.totalDuration.Round(time.Millisecond),
			stats.count, avg.Round(time.Microsecond), pct)
	}

	// Add bottleneck info
	if bottleneck != "" && totalTime > 0 {
		pct := float64(maxDuration) / float64(totalTime) * 100
		result += fmt.Sprintf("Bottleneck: %s (%.0f%%)\n", bottleneck, pct)
	}

	return result
}

// DetailedSummary returns percentile statistics for detailed mode.
func (s *TimingStats) DetailedSummary() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.level < DebugDetailed || len(s.stages) == 0 {
		return ""
	}

	result := "=== Percentile Statistics ===\n"
	for _, name := range s.order {
		stats := s.stages[name]
		if len(stats.durations) < 2 {
			continue
		}

		// Sort for percentile calculation
		sorted := make([]time.Duration, len(stats.durations))
		copy(sorted, stats.durations)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i] < sorted[j]
		})

		min := sorted[0]
		max := sorted[len(sorted)-1]
		p50 := sorted[len(sorted)/2]
		p95Idx := int(float64(len(sorted)) * 0.95)
		if p95Idx >= len(sorted) {
			p95Idx = len(sorted) - 1
		}
		p95 := sorted[p95Idx]

		result += fmt.Sprintf("  %s: min=%v, p50=%v, p95=%v, max=%v\n",
			name, min.Round(time.Millisecond), p50.Round(time.Millisecond),
			p95.Round(time.Millisecond), max.Round(time.Millisecond))
	}

	return result
}

// PrintSummary prints the timing summary to the configured writer.
func (s *TimingStats) PrintSummary() {
	if s.level < DebugSummary {
		return
	}

	summary := s.Summary()
	if summary != "" {
		_, _ = fmt.Fprint(s.writer, "[DEBUG] "+summary)
	}

	if s.level >= DebugDetailed {
		detailed := s.DetailedSummary()
		if detailed != "" {
			_, _ = fmt.Fprint(s.writer, "[DEBUG] "+detailed)
		}
	}
}

// Start creates and returns a timer, recording the result when stopped.
// Usage: timer := stats.Start("operation"); defer timer.Stop()
func (s *TimingStats) Start(name string) *StatsTimer {
	return &StatsTimer{
		stats: s,
		name:  name,
		start: time.Now(),
	}
}

// StatsTimer is a timer that records to TimingStats when stopped.
type StatsTimer struct {
	stats *TimingStats
	name  string
	start time.Time
	count int64
}

// WithCount sets the count for batch operations.
func (t *StatsTimer) WithCount(count int64) *StatsTimer {
	t.count = count
	return t
}

// Stop stops the timer and records the duration.
func (t *StatsTimer) Stop() time.Duration {
	elapsed := time.Since(t.start)
	count := t.count
	if count == 0 {
		count = 1
	}
	t.stats.RecordStage(t.name, elapsed, count)
	return elapsed
}

// StopWithLog stops the timer and logs immediately (for detailed mode).
func (t *StatsTimer) StopWithLog() time.Duration {
	elapsed := time.Since(t.start)
	count := t.count
	if count == 0 {
		count = 1
	}
	t.stats.RecordOp(t.name, elapsed, count)
	return elapsed
}
