package store

import (
	"database/sql"
	"fmt"
)

const documentFTSSchemaVersion = 2

const documentMetadataUpsertSQL = `INSERT INTO documents (id, filepath, content, start_line, end_line, metadata, is_test)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
 filepath=excluded.filepath, content=excluded.content, start_line=excluded.start_line,
 end_line=excluded.end_line, metadata=excluded.metadata, is_test=excluded.is_test`

const documentEmbeddingUpsertSQL = `INSERT INTO documents (id, filepath, content, start_line, end_line, metadata, is_test, embedding)
VALUES (?, ?, ?, ?, ?, ?, ?, vector(?))
ON CONFLICT(id) DO UPDATE SET
 filepath=excluded.filepath, content=excluded.content, start_line=excluded.start_line,
 end_line=excluded.end_line, metadata=excluded.metadata, is_test=excluded.is_test,
 embedding=excluded.embedding`

const enrichedFTSContentSQL = `COALESCE(content, '') || ' ' || COALESCE(json_extract(metadata, '$.lexical'), '')`

func initDocumentFTS(db *sql.DB) error {
	var version int
	_ = db.QueryRow(`SELECT CAST(value AS INTEGER) FROM metadata WHERE key = 'fts_schema_version'`).Scan(&version)
	rebuild := version != documentFTSSchemaVersion
	if rebuild {
		for _, stmt := range []string{
			`DROP TRIGGER IF EXISTS documents_ai`,
			`DROP TRIGGER IF EXISTS documents_ad`,
			`DROP TRIGGER IF EXISTS documents_au`,
			`DROP TABLE IF EXISTS documents_fts`,
		} {
			if _, err := db.Exec(stmt); err != nil {
				return fmt.Errorf("reset FTS schema: %w", err)
			}
		}
	}

	if _, err := db.Exec(`
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
			content,
			filepath,
			content='documents',
			content_rowid='rowid',
			tokenize='unicode61'
		)
	`); err != nil {
		return fmt.Errorf("create FTS5 table: %w", err)
	}

	triggers := []string{
		`CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
			INSERT INTO documents_fts(rowid, content, filepath)
			VALUES (NEW.rowid, COALESCE(NEW.content, '') || ' ' || COALESCE(json_extract(NEW.metadata, '$.lexical'), ''), NEW.filepath);
		END`,
		`CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
			INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
			VALUES ('delete', OLD.rowid, COALESCE(OLD.content, '') || ' ' || COALESCE(json_extract(OLD.metadata, '$.lexical'), ''), OLD.filepath);
		END`,
		`CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
			INSERT INTO documents_fts(documents_fts, rowid, content, filepath)
			VALUES ('delete', OLD.rowid, COALESCE(OLD.content, '') || ' ' || COALESCE(json_extract(OLD.metadata, '$.lexical'), ''), OLD.filepath);
			INSERT INTO documents_fts(rowid, content, filepath)
			VALUES (NEW.rowid, COALESCE(NEW.content, '') || ' ' || COALESCE(json_extract(NEW.metadata, '$.lexical'), ''), NEW.filepath);
		END`,
	}
	for _, trigger := range triggers {
		if _, err := db.Exec(trigger); err != nil {
			return fmt.Errorf("create FTS trigger: %w", err)
		}
	}

	if rebuild {
		if _, err := db.Exec(`
			INSERT INTO documents_fts(rowid, content, filepath)
			SELECT rowid, ` + enrichedFTSContentSQL + `, filepath FROM documents
		`); err != nil {
			return fmt.Errorf("rebuild enriched FTS: %w", err)
		}
		if _, err := db.Exec(`INSERT OR REPLACE INTO metadata(key, value) VALUES ('fts_schema_version', ?)`, documentFTSSchemaVersion); err != nil {
			return fmt.Errorf("store FTS schema version: %w", err)
		}
	}
	return nil
}

func ensureDocumentFTS(db *sql.DB) error {
	if err := initDocumentFTS(db); err != nil {
		return err
	}
	var ftsCount, docCount int
	if err := db.QueryRow(`SELECT COUNT(*) FROM documents_fts`).Scan(&ftsCount); err != nil {
		return err
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&docCount); err != nil {
		return err
	}
	if ftsCount == docCount {
		return nil
	}
	if _, err := db.Exec(`DELETE FROM metadata WHERE key = 'fts_schema_version'`); err != nil {
		return fmt.Errorf("invalidate stale FTS schema: %w", err)
	}
	return initDocumentFTS(db)
}
