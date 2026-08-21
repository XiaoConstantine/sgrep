package conv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

// DocumentEmbedder generates embeddings for conversation turns.
type DocumentEmbedder interface {
	EmbedDocuments(context.Context, []string) ([][]float32, error)
}

// Indexer handles indexing conversations.
type Indexer struct {
	store    *Store
	embedder DocumentEmbedder
	chunker  *Chunker
	force    bool
}

// IndexerConfig configures the indexer.
type IndexerConfig struct {
	Store    *Store
	Embedder DocumentEmbedder
	Chunker  *Chunker
	Force    bool // Re-index even if session exists
}

// NewIndexer creates a new conversation indexer.
func NewIndexer(cfg IndexerConfig) *Indexer {
	chunker := cfg.Chunker
	if chunker == nil {
		chunker = NewChunker()
	}

	return &Indexer{
		store:    cfg.Store,
		embedder: cfg.Embedder,
		chunker:  chunker,
		force:    cfg.Force,
	}
}

// IndexResult contains the results of an indexing operation.
type IndexResult struct {
	Agent           AgentType
	SessionsFound   int
	SessionsIndexed int
	TurnsIndexed    int
	Errors          []error
	Duration        time.Duration
}

// IndexSessions indexes a batch of parsed sessions.
func (idx *Indexer) IndexSessions(ctx context.Context, sessions []*Session) (*IndexResult, error) {
	startTime := time.Now()
	result := &IndexResult{}

	if len(sessions) > 0 {
		result.Agent = sessions[0].Agent
	}

	for _, session := range sessions {
		result.SessionsFound++
		session.TotalTokens = EstimateSessionTokens(session)
		contentHash, err := sessionContentHash(session)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("hash session %s failed: %w", session.ID, err))
			continue
		}

		if !idx.force {
			meta, exists, err := idx.store.GetSessionMeta(ctx, session.ID)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("check session meta for %s failed: %w", session.ID, err))
				continue
			}
			if exists && meta.ContentHash == contentHash {
				missing, err := idx.store.MissingEmbeddingsCountForSession(ctx, session.ID)
				if err != nil {
					result.Errors = append(result.Errors, fmt.Errorf("check missing embeddings for %s failed: %w", session.ID, err))
					continue
				}
				if missing == 0 {
					continue
				}
			}
		}

		if err := idx.indexSessionWithHash(ctx, session, contentHash); err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("index session %s failed: %w", session.ID, err))
			continue
		}

		result.SessionsIndexed++
		result.TurnsIndexed += len(session.Turns)
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// IndexSession indexes a single session unless its content-addressed snapshot is current.
func (idx *Indexer) IndexSession(ctx context.Context, session *Session) error {
	session.TotalTokens = EstimateSessionTokens(session)
	contentHash, err := sessionContentHash(session)
	if err != nil {
		return err
	}
	if !idx.force {
		meta, exists, err := idx.store.GetSessionMeta(ctx, session.ID)
		if err != nil {
			return err
		}
		if exists && meta.ContentHash == contentHash {
			missing, err := idx.store.MissingEmbeddingsCountForSession(ctx, session.ID)
			if err != nil {
				return err
			}
			if missing == 0 {
				return nil
			}
		}
	}
	return idx.indexSessionWithHash(ctx, session, contentHash)
}

// RebuildTQVectorStore refreshes the compact TQ-MSE sidecar for conversation search.
func (idx *Indexer) RebuildTQVectorStore(ctx context.Context) (int, error) {
	return idx.store.RebuildTQVectorStore(ctx)
}

func sessionContentHash(session *Session) (string, error) {
	data, err := json.Marshal(session)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// indexSessionWithHash embeds the complete parsed snapshot before publishing it.
func (idx *Indexer) indexSessionWithHash(ctx context.Context, session *Session, contentHash string) error {
	if idx.embedder == nil {
		return fmt.Errorf("conversation indexing requires an embedder")
	}

	chunks := idx.chunker.ChunkSession(session)
	type turnEmbedding struct {
		sum   []float32
		count int
	}
	pooled := make(map[string]*turnEmbedding)
	const batchSize = 10
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		contents := make([]string, len(batch))
		for j := range batch {
			contents[j] = batch[j].Content
		}

		embeddings, err := idx.embedder.EmbedDocuments(ctx, contents)
		if err != nil {
			return fmt.Errorf("failed to generate embeddings: %w", err)
		}
		if len(embeddings) != len(batch) {
			return fmt.Errorf("embedding backend returned %d vectors for %d documents", len(embeddings), len(batch))
		}
		for j, embedding := range embeddings {
			turnID := fmt.Sprintf("%s:%d", batch[j].SessionID, batch[j].TurnIndex)
			entry := pooled[turnID]
			if entry == nil {
				entry = &turnEmbedding{sum: make([]float32, len(embedding))}
				pooled[turnID] = entry
			}
			if len(entry.sum) != len(embedding) {
				return fmt.Errorf("inconsistent embedding dimensions for %s", turnID)
			}
			for dim, value := range embedding {
				entry.sum[dim] += value
			}
			entry.count++
		}
	}

	turnIDs := make([]string, 0, len(pooled))
	for turnID := range pooled {
		turnIDs = append(turnIDs, turnID)
	}
	sort.Strings(turnIDs)
	embeddings := make([][]float32, 0, len(turnIDs))
	for _, turnID := range turnIDs {
		entry := pooled[turnID]
		if entry.count == 0 {
			continue
		}
		for dim := range entry.sum {
			entry.sum[dim] /= float32(entry.count)
		}
		embeddings = append(embeddings, entry.sum)
	}
	if len(turnIDs) != len(session.Turns) {
		return fmt.Errorf("generated embeddings for %d of %d turns", len(turnIDs), len(session.Turns))
	}
	if err := idx.store.ReplaceSessionWithEmbeddings(ctx, session, turnIDs, embeddings, contentHash); err != nil {
		return fmt.Errorf("failed to publish session snapshot: %w", err)
	}
	return nil
}
