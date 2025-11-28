// Package sgrep provides semantic code search capabilities.
//
// sgrep is designed for use both as a CLI tool and as an embedded library.
// For CLI usage, install with: go install github.com/XiaoConstantine/sgrep/cmd/sgrep@latest
//
// For library usage:
//
//	client, err := sgrep.New("/path/to/codebase")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
//	// Index the codebase (required before searching)
//	if err := client.Index(ctx); err != nil {
//	    log.Fatal(err)
//	}
//
//	// Search for code by intent
//	results, err := client.Search(ctx, "authentication logic", 10)
package sgrep

import (
	"context"

	"github.com/XiaoConstantine/sgrep/pkg/index"
	"github.com/XiaoConstantine/sgrep/pkg/search"
	"github.com/XiaoConstantine/sgrep/pkg/store"
)

// Result represents a search result.
type Result = search.Result

// Client provides semantic code search capabilities.
type Client struct {
	indexer  *index.Indexer
	searcher *search.Searcher
	store    store.Storer
	path     string
}

// Options configures the sgrep client.
type Options struct {
	// Threshold is the similarity threshold (cosine distance, range 0-2).
	// 0 = identical, 2 = opposite. Default: 0.65
	Threshold float64
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() Options {
	return Options{
		Threshold: 0.65,
	}
}

// New creates a new sgrep client for the given codebase path.
// Call Close() when done to release resources.
func New(path string) (*Client, error) {
	indexer, err := index.New(path)
	if err != nil {
		return nil, err
	}

	return &Client{
		indexer: indexer,
		path:    path,
	}, nil
}

// Index indexes the codebase. Required before searching.
func (c *Client) Index(ctx context.Context) error {
	return c.indexer.Index(ctx)
}

// Search finds code matching the semantic query.
// Returns up to limit results sorted by relevance.
func (c *Client) Search(ctx context.Context, query string, limit int) ([]Result, error) {
	return c.SearchWithThreshold(ctx, query, limit, 0.65)
}

// SearchWithThreshold finds code with a custom similarity threshold.
// Uses cosine distance (0 = identical, 2 = opposite). Lower threshold = stricter matching.
func (c *Client) SearchWithThreshold(ctx context.Context, query string, limit int, threshold float64) ([]Result, error) {
	if c.searcher == nil {
		s, err := c.indexer.Store()
		if err != nil {
			return nil, err
		}
		c.store = s
		c.searcher = search.New(s)
	}

	return c.searcher.Search(ctx, query, limit, threshold)
}

// Watch continuously indexes file changes.
// Blocks until context is cancelled.
func (c *Client) Watch(ctx context.Context) error {
	return c.indexer.Watch(ctx)
}

// Close releases all resources.
func (c *Client) Close() error {
	if c.store != nil {
		_ = c.store.Close()
	}
	return c.indexer.Close()
}
