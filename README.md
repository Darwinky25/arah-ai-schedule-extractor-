# Arah.AI Schedule Extractor v2.1 📅

A general-purpose Python engine that extracts, normalizes, and validates **any Indonesian university schedule PDF** into clean, structured JSON. Designed as the core data pipeline for the **Arah.AI Automated Scheduler**.

> **v2.1** — Structured logging, type hints, vectorized validation, time-overlap & lecturer-conflict detection, CLI with argparse, multi-session deduplication, unit tests.

## 🚀 Key Features

### General-Purpose Extraction
- **Any Indonesian University PDF** — Score-based synonym header detection (`MATA KULIAH`, `DOSEN`, `RUANG`, etc.) automatically maps columns regardless of layout.
- **Multi-PDF Batch Processing** — Drop multiple PDFs into `jadwal/` and extract them all at once.

### Intelligent OCR Repair
- **Fuzzy Day Matching** — 4-strategy matching (exact → reversed → sorted-char → edit-distance) fixes OCR-scrambled days like `KSAMI` → `KAMIS`, `SAUBT` → `SABTU`, `JUTMA` → `JUMAT`.
- **Spaced Text Recovery** — Parses OCR-spaced times like `0 7. 0 0 - 0 7. 5 0` → `07:00 - 07:50`.
- **Targeted Text Reversal** — Only reverses `day` and `time_slot` columns (not course names).

### Time & Session Engine
- **Configurable Session Grids** — Normal (Mon-Thu/Sat), Friday (shifted afternoon), Evening (sessions XIII-XVI).
- **Roman Numeral Mapping** — Converts session IDs (I-XVI) to exact start/end times.
- **Slot Duration** — Default 50 minutes, configurable via `ScheduleExtractor(slot_duration=...)`.
- **End Time Cap** — Configurable (default 22:30) via `ScheduleExtractor(end_time_cap=...)`.

### Data Cleaning Pipeline
- **Column Shift Detection** — 4 patterns: digit course_name (leaked semester), course in class_name field, empty class+lecturer, digit lecturer + NaN room.
- **Smart Forward Fill** — Only fills course metadata for genuine team-teaching rows (same day+time), not blind `ffill`.
- **Ghost Row Removal** — Deduplicates rows where one has an empty lecturer.
- **Multi-Session Deduplication** — Collapses redundant session rows into a single entry per course/class/room.
- **Artifact Row Cleanup** — Drops rows with empty `class_name` + empty `lecturer`, or empty `lecturer` + empty `room_id`.
- **Title Stripping** — Removes academic titles (Dr., S.T., M.Kom., Prof., etc.) from lecturer names.

### Validation Suite
- Required field checks (course_name, day, start_time, end_time)
- Time logic validation (start < end)
- **Room time-overlap detection** (ignoring `DARING`/`ONLINE` rooms)
- **Lecturer time-overlap detection** (warns when same lecturer has overlapping classes)
- SKS range check (1–6)
- Duplicate detection
- Unrealistic end-time detection (> 22:30)
- Generates markdown report with error/warning separation

## 📂 Project Structure

```
├── src/
│   ├── main.py            # CLI entry point — argparse, batch processing
│   ├── extractor.py       # Core extraction engine (~920 lines)
│   ├── validator.py       # Data integrity validation suite
│   └── compare_truth.py   # Ground-truth comparison utility
├── tests/
│   └── test_core.py       # Unit tests (53 tests)
├── jadwal/                # Input: drop PDF schedules here
├── output/                # Output: JSON + validation reports
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Darwinky25/arah-ai-schedule-extractor-.git
   cd arah-ai-schedule-extractor-
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Usage

1. Place one or more schedule PDFs in the `jadwal/` folder.
2. Run the extractor:

```bash
python3 src/main.py
```

### CLI Options

```bash
python3 src/main.py --help
python3 src/main.py -i path/to/pdfs -o path/to/output
python3 src/main.py --verbose          # Enable DEBUG-level logging
```

### Output

The script generates files in the `output/` directory:

| File | Description |
|------|-------------|
| `extracted_schedule.json` | Clean structured dataset (all PDFs combined) |
| `validation_report.md` | Data quality report with errors & warnings |

### JSON Schema

Each row in the output JSON contains:

```json
{
  "day": "SENIN",
  "start_time": "07:00",
  "end_time": "09:30",
  "session_id": "I",
  "time_slot": "07:00-09:30",
  "course_name": "STRUKTUR DATA",
  "class_name": "IF-A",
  "semester": 3,
  "sks": 3,
  "lecturer": "NAMA DOSEN",
  "room_id": "GKB-301",
  "is_online": false
}
```

### Custom Time Grids

For universities with different session schedules, pass custom grids:

```python
from src.extractor import ScheduleExtractor

custom_grid = {
    1: ("07:30", "08:20"), 2: ("08:20", "09:10"),
    # ... define all sessions
}

extractor = ScheduleExtractor("jadwal/your_schedule.pdf",
                              time_grid=custom_grid,
                              slot_duration=50,
                              end_time_cap="22:00")
extractor.extract_raw()
extractor.normalize_data()
extractor.clean_data()
data = extractor.to_json()
```

## 🧠 How It Works

### 1. PDF Table Extraction
Uses `pdfplumber` to extract raw tables from each page, then applies score-based synonym matching to identify column headers automatically.

### 2. OCR Artifact Repair
Fixes common PDF text extraction issues:
- **Reversed text**: `KSAMI` → `KAMIS` (fuzzy matching with 4 strategies)
- **Spaced characters**: `0 7. 0 0` → `07:00` (whitespace normalization)
- **Column shifts**: Detects when data has shifted between columns and corrects it

### 3. Data Normalization
- Maps Roman numeral sessions to time slots
- Calculates start/end times from session + SKS
- Merges team-teaching rows (multiple lecturers, same slot)
- Cleans lecturer names (removes titles, normalizes whitespace)

### 4. Validation
Runs automated integrity checks and generates a markdown report indicating PASS/FAIL with detailed findings.

## 📊 Performance

Tested on real university schedule PDFs:

| Metric | v1.0 | v2.0 | v2.1 |
|--------|------|------|------|
| Rows extracted | 1,385 | 1,327 | 585 (deduplicated) |
| Invalid days | 58 | 0 | 0 |
| Null times | 1,385 | 0 | 0 |
| SKS = 0 | 55 | 0 | 0 |
| Empty class_name | 85 | 0 | 0 |
| Ghost duplicates | ~20 | 0 | 0 |
| JSON errors (NaN) | — | 5 | 0 |
| Validation | FAIL | PASS* | ✅ PASS |

> v2.1 collapses multi-session duplicates into unique course entries and removes
> artifact rows that have no lecturer and no room.

## 🧪 Testing

Run the full test suite (53 tests):

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover: `fuzzy_match_day`, `parse_roman`, `split_time`, `ScheduleExtractor` grids,
and all validator checks (room overlap, lecturer overlap, time logic, etc.).

## 🔧 Dependencies

- `pdfplumber` ≥ 0.10 — PDF table extraction
- `pandas` ≥ 2.0 — Data manipulation & cleaning
- `openpyxl` ≥ 3.1 — Excel support (optional)
- `pytest` — Testing (dev only)
- Python 3.8+

## 🤝 Contributing

This tool is production-ready for the Arah.AI MVP. The core extraction logic lives in `src/extractor.py`. Key extension points:

- **New university formats**: Add header synonyms in `HEADER_SYNONYMS` dict
- **Custom time grids**: Pass to `ScheduleExtractor` constructor
- **Additional validation rules**: Extend `validate_extraction()` in `src/validator.py`
- **Tunable thresholds**: Adjust `MIN_DAY_MATCH_SCORE`, `END_TIME_CAP`, etc. in `extractor.py`
