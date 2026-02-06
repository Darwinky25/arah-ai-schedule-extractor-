# Arah.AI Schedule Extractor 📅

A high-performance Python automated tool designed to extract, normalize, and validate complex university schedule data from PDF documents into clean, structured JSON format. This engine serves as the core data processing unit for the **Arah.AI Automated Scheduler**.

## 🚀 Key Features

- **Advanced PDF Extraction**: Uses `pdfplumber` for precise table extraction.
- **"Anti-Gravity" Normalization**: Automatically detects and fixes reversed or vertical text (OCR artifacts).
- **Intelligent Time & Session Mapping**: 
  - Converts Roman Numeral Session IDs (I-XVI) to standardized time slots.
  - Automatically handles "Evening/Online" classes with distorted OCR timestamps.
- **Structure Recovery**:
  - Fixes shifted columns (e.g., Semester/Course Name swaps).
  - Merges "Team Teaching" rows (multiple lecturers for the same slot).
- **Data Cleaning**:
  - Removes academic titles and cleans lecturer names.
  - Deduplicates "Ghost Rows" created by PDF parsing artifacts.
- **Automated Validation**: Built-in integrity checks for missing fields, time logic clashes, and room conflicts.

## 📂 Project Structure

```
├── src/
│   ├── main.py            # Entry point of the application
│   ├── extractor.py       # Core logic for extraction and normalization
│   ├── validator.py       # Data integrity validation suite
│   └── compare_truth.py   # Utility for ground-truth comparison
├── jadwal/                # Input directory for PDF schedules
├── output/                # Output directory for JSON & Reports
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Darwinky25/arah-ai-schedule-extractor-.git
   cd arah-ai-schedule-extractor-
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Usage

Place your schedule PDF in the `jadwal/` folder (default: `Jadwal Mata Kuliah Semester GANJIL TA.2024-2025.pdf`).

Run the extractor:
```bash
python3 src/main.py
```

### Output
The script generates two files in the `output/` directory:
1. **`extracted_schedule.json`**: The final clean dataset ready for LLM consumption.
2. **`validation_report.md`**: A health check report of the extracted data.

## 🧠 Logic Highlights

### Time Slot Repair
The system handles corrupt OCR timestamps like `210.9.4500-` by ignoring the raw text and recalculating the time based on the Session ID (e.g., Session XIII) and SKS duration.

### Team Teaching Merge
Rows with identical Room, Day, and Time but different lecturers are merged into a single entry:
- **Before**: 2 rows (Lecturer A, Lecturer B)
- **After**: 1 row (Lecturer: "Lecturer A, Lecturer B")

## 🤝 Contribution

This tool is production-ready for the Arah.AI MVP. For modifications, please check `src/extractor.py` for the core cleaning logic.
