-- =============================================================================
-- Universal Data Ingestion Layer — PostgreSQL Schema
-- =============================================================================
-- Run this script ONCE against your PostgreSQL database to create all objects.
-- Idempotent: safe to re-run with IF NOT EXISTS guards.
-- =============================================================================

-- Enable the pgcrypto extension for gen_random_uuid() (UUID v4 generation)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- TABLE: datasets
-- Stores metadata about each ingested file or API source.
-- =============================================================================
CREATE TABLE IF NOT EXISTS datasets (
    id               UUID         NOT NULL DEFAULT gen_random_uuid(),
    name             VARCHAR(255) NOT NULL,
    source_type      VARCHAR(50)  NOT NULL DEFAULT 'file',   -- 'file' | 'api'
    source_path      TEXT,
    row_count        INTEGER,
    datetime_column  VARCHAR(255),
    numeric_columns  JSONB,          -- JSON array of column names
    text_columns     JSONB,          -- JSON array of column names
    schema_profile   JSONB,          -- Full schema profile snapshot
    status           VARCHAR(50)  NOT NULL DEFAULT 'pending', -- 'pending' | 'success' | 'failed'
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_datasets PRIMARY KEY (id),
    CONSTRAINT uq_dataset_name_path UNIQUE (name, source_path),
    CONSTRAINT chk_datasets_status CHECK (status IN ('pending', 'success', 'failed')),
    CONSTRAINT chk_datasets_source_type CHECK (source_type IN ('file', 'api'))
);

-- Indexes on datasets
CREATE INDEX IF NOT EXISTS ix_datasets_status     ON datasets (status);
CREATE INDEX IF NOT EXISTS ix_datasets_created_at ON datasets (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_datasets_name       ON datasets (name);

-- Trigger: auto-update updated_at on row modification
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_datasets_updated_at ON datasets;
CREATE TRIGGER trg_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- =============================================================================
-- TABLE: data_points
-- Stores individual data rows as JSONB payloads, linked to a dataset.
-- =============================================================================
CREATE TABLE IF NOT EXISTS data_points (
    id          BIGSERIAL    NOT NULL,
    dataset_id  UUID         NOT NULL,
    row_index   INTEGER      NOT NULL,       -- 0-based row position in source
    timestamp   TIMESTAMPTZ,                 -- Parsed datetime column value
    payload     JSONB        NOT NULL,       -- All other columns as JSON
    is_valid    BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_data_points PRIMARY KEY (id),
    CONSTRAINT fk_data_points_dataset
        FOREIGN KEY (dataset_id)
        REFERENCES datasets (id)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

-- Standard B-tree indexes
CREATE INDEX IF NOT EXISTS ix_data_points_dataset_id
    ON data_points (dataset_id);

CREATE INDEX IF NOT EXISTS ix_data_points_timestamp
    ON data_points (timestamp DESC)
    WHERE timestamp IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_data_points_dataset_timestamp
    ON data_points (dataset_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS ix_data_points_is_valid
    ON data_points (dataset_id, is_valid)
    WHERE is_valid = TRUE;

-- GIN index for fast JSONB key/value lookups (supports @>, ?, ?|, ?& operators)
CREATE INDEX IF NOT EXISTS ix_data_points_payload_gin
    ON data_points USING GIN (payload jsonb_path_ops);


-- =============================================================================
-- VIEWS (optional but useful for analytics)
-- =============================================================================

-- Summary view: one row per dataset with basic stats
CREATE OR REPLACE VIEW v_dataset_summary AS
SELECT
    d.id,
    d.name,
    d.source_type,
    d.status,
    d.row_count,
    d.datetime_column,
    d.created_at,
    COUNT(dp.id)              AS stored_rows,
    MIN(dp.timestamp)         AS earliest_timestamp,
    MAX(dp.timestamp)         AS latest_timestamp
FROM datasets d
LEFT JOIN data_points dp ON dp.dataset_id = d.id
GROUP BY d.id, d.name, d.source_type, d.status, d.row_count, d.datetime_column, d.created_at;


-- =============================================================================
-- USEFUL QUERY EXAMPLES (for reference / analytics engineers)
-- =============================================================================

-- 1. All datasets
--    SELECT * FROM v_dataset_summary ORDER BY created_at DESC;

-- 2. Fetch rows for a specific dataset in time range
--    SELECT timestamp, payload
--    FROM data_points
--    WHERE dataset_id = '<uuid>'
--      AND timestamp BETWEEN '2024-01-01' AND '2024-12-31'
--    ORDER BY timestamp;

-- 3. Filter by a JSONB field value (uses GIN index)
--    SELECT * FROM data_points
--    WHERE dataset_id = '<uuid>'
--      AND payload @> '{"region": "EMEA"}';

-- 4. Extract a numeric field and aggregate
--    SELECT
--        AVG((payload->>'revenue')::numeric) AS avg_revenue,
--        MAX((payload->>'revenue')::numeric) AS max_revenue
--    FROM data_points
--    WHERE dataset_id = '<uuid>'
--      AND payload->>'revenue' IS NOT NULL;

-- 5. Count invalid rows
--    SELECT COUNT(*) FROM data_points WHERE dataset_id = '<uuid>' AND is_valid = FALSE;
