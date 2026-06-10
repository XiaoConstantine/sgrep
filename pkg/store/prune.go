package store

import (
	"context"
	"database/sql"
	"fmt"
)

func populateTempSet(ctx context.Context, tx *sql.Tx, table, column string, values []string) error {
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`CREATE TEMP TABLE IF NOT EXISTS %s (%s TEXT PRIMARY KEY)`, table, column)); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, table)); err != nil {
		return err
	}
	if len(values) == 0 {
		return nil
	}
	stmt, err := tx.PrepareContext(ctx, fmt.Sprintf(`INSERT OR IGNORE INTO %s (%s) VALUES (?)`, table, column))
	if err != nil {
		return err
	}
	defer func() { _ = stmt.Close() }()
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, err := stmt.ExecContext(ctx, value); err != nil {
			return err
		}
	}
	return nil
}

func staleDocumentIDs(ctx context.Context, tx *sql.Tx) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id FROM documents WHERE id NOT IN (SELECT id FROM _sgrep_live_doc_ids)`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}
