package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
	"github.com/XiaoConstantine/sgrep/pkg/conv/parser"
	"github.com/fsnotify/fsnotify"
)

type convWatchOptions struct {
	verbose        bool
	cursorThrottle time.Duration
	debounce       time.Duration
}

type watchJob struct {
	parser parser.Parser
	path   string
}

type openCodeSessionCache struct {
	basePath string
	paths    map[string]string
}

func watchConversations(ctx context.Context, indexer *conv.Indexer, parsers []parser.Parser, verbose bool) error {
	opts := convWatchOptions{
		verbose:        verbose,
		cursorThrottle: 3 * time.Second,
		debounce:       750 * time.Millisecond,
	}

	if opts.verbose {
		fmt.Println("Indexing conversations (initial pass)...")
	}
	if err := indexConversationsOnce(ctx, indexer, parsers, opts.verbose); err != nil {
		return err
	}

	fmt.Println("Watching conversations... (Ctrl+C to stop)")

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer func() { _ = watcher.Close() }()

	roots := buildConvWatchRoots(parsers)
	for _, root := range roots {
		if err := addRecursiveWatch(watcher, root); err != nil && opts.verbose {
			fmt.Fprintf(os.Stderr, "Watch warning: failed to add %s: %v\n", root, err)
		}
	}

	cache := newOpenCodeSessionCache(parsers)
	cursorThrottle := make(map[string]time.Time)

	var debounce *time.Timer
	pending := make(map[string]watchJob)
	var mu sync.Mutex

	queueJob := func(job watchJob) {
		key := string(job.parser.AgentType()) + ":" + job.path
		mu.Lock()
		pending[key] = job
		if debounce != nil {
			debounce.Stop()
		}
		debounce = time.AfterFunc(opts.debounce, func() {
			processConvJobs(ctx, indexer, pending, &mu, opts.verbose)
		})
		mu.Unlock()
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}

			if event.Op&(fsnotify.Create|fsnotify.Write) == 0 {
				continue
			}

			if isDir(event.Name) && event.Op&fsnotify.Create != 0 {
				_ = addRecursiveWatch(watcher, event.Name)
				continue
			}

			agentParser := parserForEvent(parsers, event.Name)
			if agentParser == nil {
				continue
			}

			switch agentParser.AgentType() {
			case conv.AgentCursor:
				if !strings.HasSuffix(event.Name, "state.vscdb") {
					continue
				}
				if shouldThrottle(cursorThrottle, event.Name, opts.cursorThrottle) {
					continue
				}
				queueJob(watchJob{parser: agentParser, path: event.Name})

			case conv.AgentOpenCode:
				if strings.Contains(event.Name, string(filepath.Separator)+"part"+string(filepath.Separator)) {
					continue
				}
				sessionPath := resolveOpenCodeSessionPath(agentParser.DefaultPath(), event.Name, cache)
				if sessionPath == "" {
					continue
				}
				queueJob(watchJob{parser: agentParser, path: sessionPath})

			default:
				if !strings.HasSuffix(event.Name, ".jsonl") {
					continue
				}
				queueJob(watchJob{parser: agentParser, path: event.Name})
			}

		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			fmt.Fprintf(os.Stderr, "Watch error: %v\n", err)
		}
	}
}

func indexConversationsOnce(ctx context.Context, indexer *conv.Indexer, parsers []parser.Parser, verbose bool) error {
	var totalSessions, totalTurns int
	for _, p := range parsers {
		paths, err := p.Discover()
		if err != nil {
			if verbose {
				fmt.Fprintf(os.Stderr, "Warning: discovery failed for %s: %v\n", p.AgentType(), err)
			}
			continue
		}

		for _, path := range paths {
			sessions, err := p.Parse(path)
			if err != nil {
				if verbose {
					fmt.Fprintf(os.Stderr, "Warning: parse failed for %s: %v\n", path, err)
				}
				continue
			}
			if len(sessions) == 0 {
				continue
			}
			result, err := indexer.IndexSessions(ctx, sessions)
			if err != nil {
				if verbose {
					fmt.Fprintf(os.Stderr, "Warning: index failed: %v\n", err)
				}
				continue
			}
			for _, e := range result.Errors {
				if verbose {
					fmt.Fprintf(os.Stderr, "Warning: %v\n", e)
				}
			}
			totalSessions += result.SessionsIndexed
			totalTurns += result.TurnsIndexed
		}
	}
	vecCount, err := indexer.RebuildTQVectorStore(ctx)
	if err != nil {
		return fmt.Errorf("refresh compact TQ-MSE conversation vectors: %w", err)
	}
	if verbose {
		fmt.Printf("Refreshed compact TQ-MSE conversation vectors (%d turns)\n", vecCount)
		fmt.Printf("Initial index complete: %d sessions, %d turns\n", totalSessions, totalTurns)
	}
	return nil
}

func processConvJobs(ctx context.Context, indexer *conv.Indexer, pending map[string]watchJob, mu *sync.Mutex, verbose bool) {
	mu.Lock()
	jobs := make([]watchJob, 0, len(pending))
	for _, job := range pending {
		jobs = append(jobs, job)
	}
	for k := range pending {
		delete(pending, k)
	}
	mu.Unlock()

	var indexedTurns int
	for _, job := range jobs {
		sessions, err := job.parser.Parse(job.path)
		if err != nil {
			if verbose {
				fmt.Fprintf(os.Stderr, "Warning: parse failed for %s: %v\n", job.path, err)
			}
			continue
		}
		if len(sessions) == 0 {
			continue
		}
		result, err := indexer.IndexSessions(ctx, sessions)
		if err != nil {
			if verbose {
				fmt.Fprintf(os.Stderr, "Warning: index failed: %v\n", err)
			}
			continue
		}
		for _, e := range result.Errors {
			if verbose {
				fmt.Fprintf(os.Stderr, "Warning: %v\n", e)
			}
		}
		if verbose && result.SessionsIndexed > 0 {
			fmt.Printf("Indexed %d sessions (%d turns) from %s\n",
				result.SessionsIndexed, result.TurnsIndexed, job.path)
		}
		indexedTurns += result.TurnsIndexed
	}
	if indexedTurns > 0 {
		vecCount, err := indexer.RebuildTQVectorStore(ctx)
		if err != nil {
			if verbose {
				fmt.Fprintf(os.Stderr, "Warning: refresh compact TQ-MSE conversation vectors failed: %v\n", err)
			}
			return
		}
		if verbose {
			fmt.Printf("Refreshed compact TQ-MSE conversation vectors (%d turns)\n", vecCount)
		}
	}
}

func buildConvWatchRoots(parsers []parser.Parser) []string {
	roots := make([]string, 0, len(parsers))
	for _, p := range parsers {
		switch p.AgentType() {
		case conv.AgentOpenCode:
			base := p.DefaultPath()
			roots = append(roots,
				filepath.Join(base, "session"),
				filepath.Join(base, "message"),
			)
		default:
			roots = append(roots, p.DefaultPath())
		}
	}
	return roots
}

func addRecursiveWatch(watcher *fsnotify.Watcher, root string) error {
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		return err
	}
	return filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return watcher.Add(path)
		}
		return nil
	})
}

func parserForEvent(parsers []parser.Parser, path string) parser.Parser {
	clean := filepath.Clean(path)
	absPath, err := filepath.Abs(clean)
	if err != nil {
		absPath = clean
	}
	var match parser.Parser
	longest := 0
	for _, p := range parsers {
		for _, root := range watchRootsForParser(p) {
			if root == "" {
				continue
			}
			absRoot, err := filepath.Abs(root)
			if err != nil {
				absRoot = root
			}
			if strings.HasPrefix(absPath, absRoot+string(filepath.Separator)) || absPath == absRoot {
				if len(absRoot) > longest {
					longest = len(absRoot)
					match = p
				}
			}
		}
	}
	return match
}

func watchRootsForParser(p parser.Parser) []string {
	switch p.AgentType() {
	case conv.AgentOpenCode:
		base := p.DefaultPath()
		return []string{
			filepath.Join(base, "session"),
			filepath.Join(base, "message"),
		}
	default:
		return []string{p.DefaultPath()}
	}
}

func shouldThrottle(last map[string]time.Time, path string, window time.Duration) bool {
	now := time.Now()
	if prev, ok := last[path]; ok && now.Sub(prev) < window {
		return true
	}
	last[path] = now
	return false
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

func newOpenCodeSessionCache(parsers []parser.Parser) *openCodeSessionCache {
	for _, p := range parsers {
		if p.AgentType() == conv.AgentOpenCode {
			cache := &openCodeSessionCache{
				basePath: p.DefaultPath(),
				paths:    make(map[string]string),
			}
			cache.refresh()
			return cache
		}
	}
	return &openCodeSessionCache{paths: make(map[string]string)}
}

func (c *openCodeSessionCache) refresh() {
	if c.basePath == "" {
		return
	}
	sessionRoot := filepath.Join(c.basePath, "session")
	_ = filepath.WalkDir(sessionRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := filepath.Base(path)
		if strings.HasPrefix(name, "ses_") && strings.HasSuffix(name, ".json") {
			sessionID := strings.TrimSuffix(name, ".json")
			c.paths[sessionID] = path
		}
		return nil
	})
}

func resolveOpenCodeSessionPath(basePath, eventPath string, cache *openCodeSessionCache) string {
	if basePath == "" {
		return ""
	}
	clean := filepath.Clean(eventPath)
	sessionRoot := filepath.Join(basePath, "session")
	messageRoot := filepath.Join(basePath, "message")

	if strings.HasPrefix(clean, sessionRoot+string(filepath.Separator)) {
		name := filepath.Base(clean)
		if strings.HasPrefix(name, "ses_") && strings.HasSuffix(name, ".json") {
			sessionID := strings.TrimSuffix(name, ".json")
			if cache != nil {
				cache.paths[sessionID] = clean
			}
			return clean
		}
		return ""
	}

	if strings.HasPrefix(clean, messageRoot+string(filepath.Separator)) {
		sessionID := readOpenCodeSessionID(clean)
		if sessionID == "" {
			return ""
		}
		if cache != nil {
			if path, ok := cache.paths[sessionID]; ok {
				return path
			}
		}
		path := findOpenCodeSessionFile(sessionRoot, sessionID)
		if path != "" && cache != nil {
			cache.paths[sessionID] = path
		}
		return path
	}

	return ""
}

func readOpenCodeSessionID(messagePath string) string {
	data, err := os.ReadFile(messagePath)
	if err != nil {
		return ""
	}
	var msg struct {
		SessionID string `json:"sessionID"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		return ""
	}
	return msg.SessionID
}

func findOpenCodeSessionFile(sessionRoot, sessionID string) string {
	var found string
	_ = filepath.WalkDir(sessionRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if filepath.Base(path) == sessionID+".json" {
			found = path
			return filepath.SkipDir
		}
		return nil
	})
	return found
}
