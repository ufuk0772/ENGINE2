# Universal Data Ingestion Layer

A production-ready, domain-agnostic data ingestion system that accepts CSV/Excel files,
automatically detects schema, cleans and standardizes data, and stores everything in
PostgreSQL using a flexible JSONB-based schema.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐             │
│  │  Loader  │──▶│ Detector │──▶│ Transformer │             │
│  │(CSV/XLSX)│   │ (Schema) │   │  (Clean +   │             │
│  └──────────┘   └──────────┘   │   Serialize)│             │
│                                └──────┬──────┘             │
│                                       │                     │
│                         ┌─────────────▼──────────────┐     │
│                         │     PostgreSQL (JSONB)      │     │
│                         │  datasets + data_points     │     │
│                         └────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|---|---|
| `app/ingestion/loader.py` | Loads CSV / Excel files into DataFrames |
| `app/ingestion/detector.py` | Auto-detects datetime, numeric, boolean, text columns |
| `app/ingestion/transformer.py` | Cleans data; produces `(timestamp, payload)` records |
| `app/db/models.py` | SQLAlchemy ORM — `datasets` + `data_points` tables |
| `app/db/connection.py` | Engine + session factory with connection pooling |
| `app/db/repository.py` | Repository pattern — all DB read/write logic |
| `app/main.py` | Orchestration entry point + CLI |
| `sql/schema.sql` | Raw SQL schema with indexes, constraints, views |

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone / unzip the project
cd project_root

# 2. Start PostgreSQL (schema is auto-applied via init script)
docker compose up -d postgres

# 3. Run ingestion with the included sample file
docker compose run --rm app

# 4. Inspect the data
docker compose exec postgres psql -U ingestion_user -d ingestion_db \
  -c "SELECT * FROM v_dataset_summary;"
```

### Option B — Local Python

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure database connection
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 4. Apply schema (or use --init-db flag)
psql -U ingestion_user -d ingestion_db -f sql/schema.sql

# 5. Run ingestion
python -m app.main --file data/sample_data.csv --init-db

# With a custom name
python -m app.main --file data/sales.xlsx --name "Q1 Sales 2024"

# With a specific Excel sheet
python -m app.main --file data/report.xlsx --sheet "January"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `ingestion_user` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `ingestion_pass` | PostgreSQL password |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `ingestion_db` | Database name |

---

## Database Schema

### `datasets` table
Stores one row per ingested file/source with metadata.

| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `name` | VARCHAR | Human-readable dataset name |
| `source_type` | VARCHAR | `file` or `api` |
| `source_path` | TEXT | Absolute path to source file |
| `row_count` | INTEGER | Number of rows inserted |
| `datetime_column` | VARCHAR | Detected datetime column name |
| `numeric_columns` | JSONB | Array of numeric column names |
| `schema_profile` | JSONB | Full detected schema snapshot |
| `status` | VARCHAR | `pending` / `success` / `failed` |
| `created_at` | TIMESTAMPTZ | Ingestion timestamp |

### `data_points` table
Stores one row per data record.

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL | Primary key |
| `dataset_id` | UUID | FK → `datasets.id` |
| `row_index` | INTEGER | Original row position (0-based) |
| `timestamp` | TIMESTAMPTZ | Parsed value from datetime column |
| `payload` | JSONB | All other columns as JSON |
| `is_valid` | BOOLEAN | Row validity flag |

#### Indexes
- B-tree on `dataset_id`, `timestamp`, `(dataset_id, timestamp)`
- **GIN index** on `payload` (`jsonb_path_ops`) — enables fast key/value searches

---

## Useful Queries

```sql
-- 1. Dataset summary
SELECT * FROM v_dataset_summary ORDER BY created_at DESC;

-- 2. Time-range query
SELECT timestamp, payload
FROM data_points
WHERE dataset_id = '<uuid>'
  AND timestamp BETWEEN '2024-01-01' AND '2024-03-31';

-- 3. Filter by JSONB field (uses GIN index)
SELECT * FROM data_points
WHERE dataset_id = '<uuid>'
  AND payload @> '{"region": "EMEA"}';

-- 4. Aggregate a numeric JSONB field
SELECT
    AVG((payload->>'revenue')::numeric) AS avg_revenue,
    MAX((payload->>'revenue')::numeric) AS max_revenue
FROM data_points
WHERE dataset_id = '<uuid>';
```

---

## Extending the System

### Add API ingestion
1. Create `app/ingestion/api_loader.py` implementing a compatible interface.
2. In `main.py`, add a `--source api` flag and route accordingly.
3. The detector, transformer, and repository layers require **zero changes**.

### Add new file formats
Extend `FileLoader.SUPPORTED_EXTENSIONS` and add a `_load_<format>()` method.

### Backfill historical data
Run the pipeline multiple times — each invocation creates a new `Dataset` record
with a unique ID, so there are no conflicts.

---

## Project Structure

```
project_root/
├── app/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py        ← File loading
│   │   ├── detector.py      ← Schema detection
│   │   └── transformer.py   ← Data cleaning & serialization
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py    ← SQLAlchemy engine + session
│   │   ├── models.py        ← ORM models
│   │   └── repository.py    ← Data access layer
│   ├── __init__.py
│   └── main.py              ← Orchestration + CLI
├── sql/
│   └── schema.sql           ← Raw DDL with indexes & views
├── data/
│   └── sample_data.csv      ← 30-row sample dataset
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT — free to use, modify, and distribute.
