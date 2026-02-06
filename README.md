# Arah.AI Schedule Extractor v2.0 📅

A general-purpose Python engine that extracts, normalizes, and validates **any Indonesian university schedule PDF** into clean, structured JSON. Designed as the core data pipeline for the **Arah.AI Automated Scheduler**.

> **v2.0** — Complete rewrite. Now handles arbitrary PDF layouts, OCR artifacts, and multi-PDF batch processing.

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
- **End Time Cap** — Caps at 22:30 to prevent unrealistic values.

### Data Cleaning Pipeline
- **Column Shift Detection** — 4 patterns: digit course_name (leaked semester), course in class_name field, empty class+lecturer, digit lecturer + NaN room.
- **Smart Forward Fill** — Only fills course metadata for genuine team-teaching rows (same day+time), not blind `ffill`.
- **Ghost Row Removal** — Deduplicates rows where one has an empty lecturer.
- **Artifact Row Cleanup** — Drops rows with both empty `class_name` and empty `lecturer`.
- **Title Stripping** — Removes academic titles (Dr., S.T., M.Kom., Prof., etc.) from lecturer names.

### Validation Suite
- Required field checks (course_name, day, start_time, end_time)
- Time logic validation (start < end)
- Room conflict detection (ignoring `DARING`/`ONLINE` rooms)
- SKS range check (1–6)
- Duplicate detection
- Generates markdown report with error/warning separation

## 📂 Project Structure

```
├── src/
│   ├── main.py            # Entry point — multi-PDF batch processing
│   ├── extractor.py       # Core extraction engine (~860 lines)
│   ├── validator.py       # Data integrity validation suite
│   └── compare_truth.py   # Utility for ground-truth comparison
├── jadwal/                # Input: drop PDF schedules here
├── output/                # Output: JSON + validation reports
├── requirements.txt       # Python dependencies
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
  "no": 1,
  "semester": 3,
  "course_code": "TIF-2301",
  "course_name": "STRUKTUR DATA",
  "class_name": "IF-A",
  "sks": 3,
  "lecturer": "NAMA DOSEN",
  "day": "SENIN",
  "session_id": "I",
  "room_id": "GKB-301",
  "start_time": "07:00",
  "end_time": "09:30",
  "time_slot": "07:00-09:30"
}
```

### Custom Time Grids

For universities with different session schedules, pass custom grids:

```python
from src.extractor import ScheduleExtractor

custom_grid = {
    "I": "07:30", "II": "08:20", "III": "09:10",
    # ... define all sessions
}

extractor = ScheduleExtractor(time_grid=custom_grid, slot_duration=50)
df = extractor.extract("jadwal/your_schedule.pdf")
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

| Metric | v1.0 | v2.0 |
|--------|------|------|
| Rows extracted | 1,385 | 1,327 |
| Invalid days | 58 | 0 |
| Null times | 1,385 | 0 |
| SKS = 0 | 55 | 0 |
| Empty class_name | 85 | 0 |
| Ghost duplicates | ~20 | 0 |
| Validation | FAIL | PASS |

## 🔧 Dependencies

- `pdfplumber` — PDF table extraction
- `pandas` — Data manipulation & cleaning
- `openpyxl` — Excel support (optional)
- Python 3.8+

## 🤝 Contributing

This tool is production-ready for the Arah.AI MVP. The core extraction logic lives in `src/extractor.py`. Key extension points:

- **New university formats**: Add header synonyms in `HEADER_SYNONYMS` dict
- **Custom time grids**: Pass to `ScheduleExtractor` constructor
- **Additional validation rules**: Extend `validate_schedule()` in `src/validator.py`
