"""
Arah.AI Universal Schedule Extractor
=====================================
General-purpose engine for extracting university schedule data from PDF documents.
Designed to handle any Indonesian university schedule format.

Key improvements over v1:
- Fuzzy day matching (handles OCR artifacts like KSAMI, SAUBT, JUTMA)
- Robust column shift detection using content-based heuristics
- Smarter fill-forward that respects row boundaries
- Configurable time grids (supports any university's session layout)
- Proper ghost-row elimination before team-teaching merge
- General header detection with scoring system
"""

import logging
import re
import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple, Any

import pdfplumber
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

KNOWN_DAYS = {
    'SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU',
    'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'
}

DAY_NORMALIZE = {
    'MONDAY': 'SENIN', 'TUESDAY': 'SELASA', 'WEDNESDAY': 'RABU',
    'THURSDAY': 'KAMIS', 'FRIDAY': 'JUMAT', 'SATURDAY': 'SABTU', 'SUNDAY': 'MINGGU',
    'SENIN': 'SENIN', 'SELASA': 'SELASA', 'RABU': 'RABU',
    'KAMIS': 'KAMIS', 'JUMAT': 'JUMAT', 'SABTU': 'SABTU', 'MINGGU': 'MINGGU',
}

INDONESIAN_DAYS = ['SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU']

ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
    'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
    'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16
}

HEADER_SYNONYMS = {
    'course_name': [
        'MATA KULIAH', 'MATKUL', 'COURSE', 'SUBJEK', 'NAMA_MK', 'NAMA MK',
        'NAMA MATA KULIAH', 'MK', 'SUBJECT', 'MATA_KULIAH'
    ],
    'time_slot': [
        'JAM', 'WAKTU', 'PUKUL', 'TIME', 'JAM KULIAH', 'JADWAL',
        'JAM MULAI', 'WAKTU MULAI'
    ],
    'day': [
        'HARI', 'H A R I', 'DAY', 'HARI KULIAH'
    ],
    'sks': [
        'SKS', 'KREDIT', 'CREDIT', 'JS', 'BEBAN'
    ],
    'class_name': [
        'KLS', 'KELAS', 'CLASS', 'GRUP', 'GROUP', 'KELOMPOK', 'KEL'
    ],
    'semester': [
        'SEM', 'SEMESTER', 'SM', 'SMT', 'SMTR'
    ],
    'room_id': [
        'RUANG', 'R.', 'ROOM', 'RUANGAN', 'GEDUNG', 'TEMPAT', 'LOKASI',
        'R', 'RG', 'KODE RUANG'
    ],
    'lecturer': [
        'DOSEN', 'PENGAJAR', 'NAMA_DOSEN', 'NAMA DOSEN', 'LECTURER',
        'INSTRUCTOR', 'PENGAMPU', 'DOSEN PENGAMPU', 'DOSEN PENGAJAR'
    ],
    'session_id': [
        'SESI', 'SESSION', 'PERTEMUAN', 'JAM KE', 'NO SESI'
    ],
    'prodi': [
        'PRODI', 'PROGRAM STUDI', 'JURUSAN', 'DEPARTMENT', 'FAKULTAS'
    ]
}

TITLE_PATTERNS = [
    r'\bProf\.?', r'\bDr\.?', r'\bDra\.?', r'\bDrs\.?', r'\bIr\.?',
    r'\bS\.T\.?', r'\bM\.T\.?', r'\bS\.Kom\.?', r'\bM\.Kom\.?',
    r'\bS\.H\.?', r'\bM\.H\.?', r'\bS\.E\.?', r'\bM\.E\.?',
    r'\bM\.M\.?', r'\bM\.Si\.?', r'\bM\.Pd\.?', r'\bM\.Sc\.?',
    r'\bM\.Hum\.?', r'\bM\.Ag\.?', r'\bM\.Kes\.?', r'\bM\.Sn\.?',
    r'\bPh\.?D\.?', r'\bS\.Pd\.?', r'\bS\.Ag\.?', r'\bS\.Sos\.?',
    r'\bM\.A\.?', r'\bM\.Phil\.?', r'\bS\.S\.?',
    r'\bS\.Hut\.?', r'\bS\.P\.?', r'\bS\.Pi\.?', r'\bS\.Pt\.?',
    r'\bS\.K\.M\.?', r'\bS\.Psi\.?', r'\bM\.Psi\.?',
    r'\bH\.', r'\bHj\.', r'\bHjb\.', r'\bAssoc\.?',
    r'\bSM\.?', r'\bHK\.?', r'\bUS\.?',
    r'\bSH\b', r'\bMH\b', r'\bSE\b', r'\bMM\b', r'\bST\b', r'\bMT\b',
    r'\bSKom\b', r'\bMKom\b', r'\bSPd\b', r'\bMPd\b', r'\bMSi\b',
    r'\bMAg\b', r'\bMHum\b', r'\bPhD\b', r'\bHum\b',
]


# ─────────────────────────────────────────────────────────────
# Tunable Thresholds
# ─────────────────────────────────────────────────────────────

MIN_DAY_MATCH_SCORE: float = 0.55
MIN_DAY_MATCH_LENGTH: int = 3
HEADER_DETECT_MIN_SCORE: int = 2
HEADER_SCAN_ROWS: int = 30
END_TIME_CAP: str = "22:30"
GARBAGE_ROW_NUMERIC_RATIO: float = 0.6


# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def fuzzy_match_day(text: Any) -> Optional[str]:
    """
    Match a potentially garbled/OCR-damaged string to the closest Indonesian day name.
    Uses: exact, reversed, sorted-char, and edit-distance strategies.
    """
    if pd.isna(text) or not str(text).strip():
        return None

    s = str(text).strip().upper()
    s_alpha = re.sub(r'[^A-Z]', '', s)
    if not s_alpha:
        return None

    # 1. Exact match
    if s_alpha in DAY_NORMALIZE:
        return DAY_NORMALIZE[s_alpha]

    # 2. Reversed match
    rev = s_alpha[::-1]
    if rev in DAY_NORMALIZE:
        return DAY_NORMALIZE[rev]

    # 3. Sorted-character match (scrambled text)
    s_sorted = ''.join(sorted(s_alpha))
    for day in INDONESIAN_DAYS:
        if ''.join(sorted(day)) == s_sorted:
            return day

    # 4. Edit-distance / fuzzy match
    best_match = None
    best_score = 0.0
    for day in INDONESIAN_DAYS:
        score = SequenceMatcher(None, s_alpha, day).ratio()
        if score > best_score:
            best_score = score
            best_match = day
    if best_score >= MIN_DAY_MATCH_SCORE and len(s_alpha) >= MIN_DAY_MATCH_LENGTH:
        return best_match

    return None


def parse_roman(s: Any) -> int:
    """Parse a Roman numeral string to integer. Returns 0 on failure."""
    if pd.isna(s):
        return 0
    s = str(s).strip().upper()
    for sep in ['-', ',', '/', ' ']:
        if sep in s:
            s = s.split(sep)[0].strip()
    return ROMAN_MAP.get(s, 0)


def add_minutes(time_str: str, mins: int) -> Optional[str]:
    """Add minutes to a HH:MM time string."""
    try:
        dt = datetime.strptime(time_str, "%H:%M")
        dt += timedelta(minutes=mins)
        return dt.strftime("%H:%M")
    except Exception:
        return None


def split_time(time_str: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a time range string into (start, end) tuple.
    Handles: '07.00-07.50', '07:00 - 07:50', '0 7. 0 0 - 0 7. 5 0', etc.
    """
    if pd.isna(time_str):
        return None, None

    s = str(time_str).strip()

    # First try: strip ALL spaces from the string to handle spaced-out OCR text
    # '0 7. 0 0 - 0 7. 5 0' -> '07.00-07.50'
    s_compact = re.sub(r'\s+', '', s)

    # Standard pattern: HH.MM-HH.MM or HH:MM-HH:MM
    pattern = r'(\d{1,2})[.:](\d{2})[-\u2013\u2014](\d{1,2})[.:](\d{2})'
    match = re.search(pattern, s_compact)
    if match:
        start = f"{int(match.group(1)):02d}:{match.group(2)}"
        end = f"{int(match.group(3)):02d}:{match.group(4)}"
        return start, end

    # Fallback: try original string with spaces allowed between time parts
    pattern2 = r'(\d{1,2})\s*[.:]\s*(\d{2})\s*[-\u2013\u2014]\s*(\d{1,2})\s*[.:]\s*(\d{2})'
    match2 = re.search(pattern2, s)
    if match2:
        start = f"{int(match2.group(1)):02d}:{match2.group(2)}"
        end = f"{int(match2.group(3)):02d}:{match2.group(4)}"
        return start, end

    return None, None


def is_likely_course_name(text: Any) -> bool:
    """Heuristic: Is this text likely a course name?"""
    if pd.isna(text):
        return False
    s = str(text).strip()
    if len(s) <= 2:
        return False
    if len(s) > 3 and not s.isdigit():
        return True
    return False


def is_likely_class_code(text: Any) -> bool:
    """Heuristic: Is this text likely a class code (A, B, A1, etc.)?"""
    if pd.isna(text):
        return False
    s = str(text).strip().upper()
    if len(s) == 1 and s.isalpha():
        return True
    if len(s) == 2 and s[0].isalpha() and s[1].isdigit():
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Main Extractor Class
# ─────────────────────────────────────────────────────────────

class ScheduleExtractor:
    """
    Universal PDF Schedule Extractor for Indonesian universities.

    Usage:
        extractor = ScheduleExtractor("path/to/jadwal.pdf")
        extractor.extract_raw()
        extractor.normalize_data()
        extractor.clean_data()
        data = extractor.to_json()

    Custom time grids:
        custom_grid = {1: ("08:00", "08:50"), 2: ("08:50", "09:40"), ...}
        extractor = ScheduleExtractor("jadwal.pdf", time_grid=custom_grid)
    """

    def __init__(
        self,
        pdf_path: str,
        time_grid: Optional[Dict[int, Tuple[str, str]]] = None,
        friday_grid: Optional[Dict[int, Tuple[str, str]]] = None,
        slot_duration: int = 50,
        end_time_cap: str = END_TIME_CAP,
    ):
        self.pdf_path = pdf_path
        self.raw_data: List[list] = []
        self.df: Optional[pd.DataFrame] = None
        self.slot_duration = slot_duration
        self.end_time_cap = end_time_cap

        self.schedule_normal = time_grid or self._build_default_grid()
        self.schedule_jumat = friday_grid or self._build_friday_grid()

        evening = self._build_evening_slots()
        for k, v in evening.items():
            self.schedule_normal.setdefault(k, v)
            self.schedule_jumat.setdefault(k, v)

    # ── Grid Builders ──────────────────────────────────────────

    @staticmethod
    def _build_default_grid():
        return {
            1: ("07:00", "07:50"), 2: ("07:50", "08:40"),
            3: ("08:40", "09:30"), 4: ("09:30", "10:20"),
            5: ("10:20", "11:10"), 6: ("11:10", "12:00"),
            7: ("13:00", "13:50"), 8: ("13:50", "14:40"),
            9: ("14:40", "15:30"), 10: ("15:30", "16:20"),
            11: ("16:20", "17:10"), 12: ("17:10", "18:00"),
        }

    @staticmethod
    def _build_friday_grid():
        grid = ScheduleExtractor._build_default_grid()
        grid.update({
            7: ("13:30", "14:20"), 8: ("14:20", "15:10"),
            9: ("15:10", "16:00"), 10: ("16:00", "16:50"),
            11: ("16:50", "17:40"), 12: ("17:40", "18:30"),
        })
        return grid

    @staticmethod
    def _build_evening_slots():
        return {
            13: ("19:00", "19:50"), 14: ("19:50", "20:40"),
            15: ("20:40", "21:30"), 16: ("21:30", "22:20"),
        }

    # ── Module 1: Table Extraction ─────────────────────────────

    def extract_raw(self) -> List[list]:
        """Extract raw table data from all pages of the PDF."""
        all_rows = []
        page_count = 0

        with pdfplumber.open(self.pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                tables = page.extract_tables()

                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })

                for table in tables:
                    if not table:
                        continue
                    for row in table:
                        if row and any(cell and str(cell).strip() for cell in row):
                            all_rows.append(row)

        self.raw_data = all_rows
        logger.info("Extracted %d raw rows from %d pages.", len(all_rows), page_count)
        return self.raw_data

    # ── Module 2: Normalization ────────────────────────────────

    def normalize_data(self) -> pd.DataFrame:
        """
        Normalize raw data: header detection, column mapping,
        text repair, fill-forward, and time calculation.
        """
        if not self.raw_data:
            raise ValueError("No raw data. Run extract_raw() first.")

        header_index = self._detect_header_row()
        if header_index == -1:
            raise ValueError("Could not auto-detect header row.")

        headers = self._clean_headers(self.raw_data[header_index])
        data_rows = self._align_rows(self.raw_data[header_index + 1:], len(headers))
        self.df = pd.DataFrame(data_rows, columns=headers)

        self._map_columns()

        self.df = self.df.replace(r'\n', ' ', regex=True)
        self.df.replace(['None', 'none', 'enoN', 'enon', 'nan', 'NaN'], np.nan, inplace=True)
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        self._filter_garbage_rows()
        self._repair_text_artifacts()

        if 'day' in self.df.columns:
            self.df['day'] = self.df['day'].apply(fuzzy_match_day)

        self._smart_fill_forward()
        self._calculate_times()

        return self.df

    def _detect_header_row(self):
        """Score each row to find the most likely header."""
        all_synonyms = []
        for syn_list in HEADER_SYNONYMS.values():
            all_synonyms.extend(syn_list)

        best_index = -1
        best_score = 0

        for i, row in enumerate(self.raw_data[:HEADER_SCAN_ROWS]):
            row_str = ' '.join(str(x).upper().replace('\n', ' ') if x else '' for x in row)
            score = sum(1 for syn in all_synonyms if syn in row_str)
            if score > best_score:
                best_score = score
                best_index = i

        return best_index if best_score >= HEADER_DETECT_MIN_SCORE else -1

    def _clean_headers(self, raw_headers):
        """Clean and deduplicate header names."""
        headers = [str(h).replace('\n', ' ').strip().upper() if h else 'UNKNOWN' for h in raw_headers]
        seen = {}
        unique = []
        for h in headers:
            if not h:
                h = 'UNKNOWN'
            if h in seen:
                seen[h] += 1
                unique.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique.append(h)
        return unique

    def _align_rows(self, rows, expected_len):
        """Ensure all rows have the same number of columns."""
        aligned = []
        for row in rows:
            if row is None:
                continue
            curr = len(row)
            if curr == expected_len:
                aligned.append(row)
            elif curr < expected_len:
                aligned.append(list(row) + [None] * (expected_len - curr))
            else:
                aligned.append(list(row[:expected_len]))
        return aligned

    def _map_columns(self):
        """Map raw headers to standardized column names using synonyms."""
        rename_map = {}
        used_targets = set()

        for col in self.df.columns:
            col_upper = col.upper().strip()
            # Skip very short generic columns like "NO", "NO."
            if col_upper in ('NO', 'NO.', 'NOMOR'):
                continue

            matched = False
            best_target = None
            best_score = 0

            for target, synonyms in HEADER_SYNONYMS.items():
                if target in used_targets:
                    continue
                for syn in synonyms:
                    # Exact match (highest priority)
                    if syn == col_upper:
                        score = 100
                    # Column contains the full synonym
                    elif syn in col_upper:
                        score = 80
                    # Synonym contains the full column (only if col is substantial)
                    elif col_upper in syn and len(col_upper) >= 3:
                        score = 60
                    else:
                        score = 0

                    if score > best_score:
                        best_score = score
                        best_target = target

            if best_target and best_score >= 60:
                rename_map[col] = best_target
                used_targets.add(best_target)

        self.df.rename(columns=rename_map, inplace=True)

    def _filter_garbage_rows(self):
        """Remove repeated header rows, column-number rows, and empty rows."""
        self.df.dropna(how='all', inplace=True)

        header_kw = ['MATA KULIAH', 'MATKUL', 'COURSE', 'NAMA MK', 'NAMA MATA KULIAH',
                      'HARI', 'SKS', 'DOSEN', 'KELAS']

        def is_header_echo(row):
            row_vals = [str(row[c]).upper().strip() for c in row.index if pd.notna(row[c])]
            matches = sum(1 for v in row_vals for kw in header_kw if kw == v or kw in v)
            return matches >= 2

        self.df = self.df[~self.df.apply(is_header_echo, axis=1)]

        def is_number_row(row):
            filled = 0
            numeric = 0
            for x in row:
                if pd.notna(x) and str(x).strip():
                    filled += 1
                    if str(x).strip().isdigit() and len(str(x).strip()) <= 2:
                        numeric += 1
            return filled > 0 and (numeric / filled) > GARBAGE_ROW_NUMERIC_RATIO

        self.df = self.df[~self.df.apply(is_number_row, axis=1)]
        self.df.reset_index(drop=True, inplace=True)

    def _repair_text_artifacts(self):
        """
        Detect and fix reversed/vertical text artifacts.
        Only reverses columns that actually show reversal patterns.
        Typically only 'day' and 'time_slot' are affected (merged cells in PDF).
        """
        if 'day' not in self.df.columns:
            return

        # Check if 'day' column has reversed text
        sample = self.df['day'].dropna().astype(str).head(30).tolist()
        normal_count = 0
        reversed_count = 0

        for d in sample:
            s = re.sub(r'[^A-Za-z]', '', d).upper()
            if s in DAY_NORMALIZE:
                normal_count += 1
            elif s[::-1] in DAY_NORMALIZE:
                reversed_count += 1

        if reversed_count > normal_count:
            logger.info("Detected reversed text in day/time columns. Applying targeted fix...")
            # Only reverse columns that are typically affected by vertical-text OCR
            # (day and time_slot — these are in merged cells on the left of the table)
            reverse_cols = [c for c in ['day', 'time_slot'] if c in self.df.columns]
            for col in reverse_cols:
                self.df[col] = self.df[col].apply(
                    lambda x: str(x).strip()[::-1]
                    if pd.notna(x) and str(x).strip() not in ('', 'nan')
                    else np.nan
                )

        self.df.replace(['enoN', 'enon', 'None', 'none', 'nan'], np.nan, inplace=True)

    def _smart_fill_forward(self):
        """
        Intelligent fill-forward that only propagates structural columns
        and fills course metadata only for true continuation rows
        (same day + same time slot = team teaching).
        """
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Only ffill truly structural columns (time block identity)
        structural_cols = [c for c in ['day', 'time_slot', 'session_id'] if c in self.df.columns]
        if structural_cols:
            self.df[structural_cols] = self.df[structural_cols].ffill()

        # For course metadata: only fill if the row is a GENUINE continuation
        # Must have: same day + same time_slot as previous row (team-teaching row)
        # AND must have a lecturer or room (not an empty row)
        meta_cols = [c for c in ['course_name', 'sks', 'class_name', 'semester'] if c in self.df.columns]

        if 'course_name' in self.df.columns:
            for i in range(1, len(self.df)):
                idx = self.df.index[i]
                prev_idx = self.df.index[i - 1]
                curr_course = self.df.at[idx, 'course_name']

                if pd.isna(curr_course) or str(curr_course).strip() == '':
                    # Check: same day + same time as previous row?
                    curr_day = self.df.at[idx, 'day'] if 'day' in self.df.columns else None
                    prev_day = self.df.at[prev_idx, 'day'] if 'day' in self.df.columns else None
                    same_day = (pd.notna(curr_day) and pd.notna(prev_day)
                                and str(curr_day) == str(prev_day))

                    curr_ts = self.df.at[idx, 'time_slot'] if 'time_slot' in self.df.columns else None
                    prev_ts = self.df.at[prev_idx, 'time_slot'] if 'time_slot' in self.df.columns else None
                    same_time = (pd.notna(curr_ts) and pd.notna(prev_ts)
                                 and str(curr_ts) == str(prev_ts))

                    has_lecturer = ('lecturer' in self.df.columns and
                                    pd.notna(self.df.at[idx, 'lecturer']) and
                                    str(self.df.at[idx, 'lecturer']).strip())

                    if same_day and same_time and has_lecturer:
                        # True team-teaching continuation — fill metadata
                        for col in meta_cols:
                            if pd.isna(self.df.at[idx, col]) or str(self.df.at[idx, col]).strip() == '':
                                self.df.at[idx, col] = self.df.at[prev_idx, col]

    def _calculate_times(self):
        """Calculate start_time and end_time from session grid + SKS."""
        if 'session_id' not in self.df.columns:
            if 'time_slot' in self.df.columns:
                times = self.df['time_slot'].apply(split_time)
                self.df['start_time'] = [t[0] for t in times]
                self.df['end_time'] = [t[1] for t in times]
            return

        results = []
        for _, row in self.df.iterrows():
            day = str(row.get('day', '')).upper() if pd.notna(row.get('day')) else ''
            session_str = row.get('session_id')
            sks_val = row.get('sks')
            raw_start, raw_end = split_time(row.get('time_slot'))

            try:
                sks = int(float(sks_val)) if pd.notna(sks_val) else 0
            except (ValueError, TypeError):
                sks = 0

            grid = self.schedule_jumat if 'JUMAT' in day else self.schedule_normal
            session_num = parse_roman(session_str)

            final_start = None
            final_end = None

            if session_num in grid:
                grid_start, grid_end = grid[session_num]
                final_start = grid_start
                if sks > 0:
                    end_session = session_num + sks - 1
                    if end_session in grid:
                        final_end = grid[end_session][1]
                    else:
                        final_end = add_minutes(grid_start, sks * self.slot_duration)
                else:
                    final_end = grid_end
            elif raw_start:
                final_start = raw_start
                if sks > 0:
                    final_end = add_minutes(raw_start, sks * self.slot_duration)
                else:
                    final_end = raw_end

            # Cap end time
            if final_end:
                try:
                    if datetime.strptime(final_end, "%H:%M") > datetime.strptime(self.end_time_cap, "%H:%M"):
                        final_end = self.end_time_cap
                except ValueError:
                    pass

            results.append((final_start, final_end))

        self.df['start_time'] = [r[0] for r in results]
        self.df['end_time'] = [r[1] for r in results]

    # ── Module 3: Data Cleaning ────────────────────────────────

    def _repair_session_one_artifacts(self) -> None:
        """Repair session-I rows whose course_name was shifted to class_name.

        On some PDF pages (especially JUMAT), the first session row in each
        room has ``course_name=None`` because of merged-cell rendering in the
        PDF table.  The actual course name lands in ``class_name`` instead.

        For each such artifact we copy course_name, class_name, sks, and
        lecturer from the session-II row of the same room+day, then
        recalculate ``end_time`` so the subsequent multi-session dedup will
        keep this row (earliest start_time) instead of the session-II copy.
        """
        required = {'course_name', 'session_id', 'room_id', 'day'}
        if not required.issubset(self.df.columns):
            return

        empty_course = (
            self.df['course_name'].isna()
            | (self.df['course_name'].astype(str).str.strip() == '')
        )
        is_session_one = self.df['session_id'] == 'I'
        has_room = (
            self.df['room_id'].notna()
            & (self.df['room_id'].astype(str).str.strip() != '')
        )
        mask = empty_course & is_session_one & has_room

        if not mask.any():
            return

        n_fixed = 0
        for idx in self.df[mask].index:
            room = self.df.at[idx, 'room_id']
            day = self.df.at[idx, 'day']

            # Locate the session-II row for the same room & day
            candidates = self.df[
                (self.df['room_id'] == room)
                & (self.df['day'] == day)
                & (self.df['session_id'] == 'II')
                & self.df['course_name'].notna()
                & (self.df['course_name'].astype(str).str.strip() != '')
            ]
            if candidates.empty:
                continue

            ref = candidates.iloc[0]
            self.df.at[idx, 'course_name'] = ref['course_name']
            self.df.at[idx, 'class_name'] = ref['class_name']
            self.df.at[idx, 'sks'] = ref['sks']
            self.df.at[idx, 'lecturer'] = ref['lecturer']

            # Recalculate end_time using session grid
            try:
                sks = int(float(ref['sks'])) if pd.notna(ref['sks']) else 0
            except (ValueError, TypeError):
                sks = 0

            if sks > 0:
                grid = (
                    self.schedule_jumat
                    if 'JUMAT' in str(day).upper()
                    else self.schedule_normal
                )
                end_session = 1 + sks - 1          # session I = 1
                if end_session in grid:
                    self.df.at[idx, 'end_time'] = grid[end_session][1]
                    start = self.df.at[idx, 'start_time']
                    end = self.df.at[idx, 'end_time']
                    if pd.notna(start) and pd.notna(end):
                        self.df.at[idx, 'time_slot'] = f"{start}-{end}"

            n_fixed += 1

        if n_fixed:
            logger.info(
                "Repaired %d session-I artifact rows with data from session II.",
                n_fixed,
            )

    def clean_data(self) -> pd.DataFrame:
        """Clean and finalize the extracted data."""
        if self.df is None:
            raise ValueError("No data to clean. Run normalize_data() first.")

        self._fix_column_shifts()

        # ── Fix session-I artifacts (JUMAT merged-cell issue) ──────────
        # On some PDF pages the first session row has course_name shifted
        # into class_name due to merged cells.  We repair them from the
        # session-II row of the same room+day and recalculate end_time
        # so the dedup later keeps the correct (earlier) start.
        self._repair_session_one_artifacts()

        # Identify online classes
        if 'room_id' in self.df.columns:
            self.df['is_online'] = self.df['room_id'].astype(str).str.contains(
                r'daring|online|zoom|meet|maya|virtual|gmeet|teams',
                case=False, na=False
            )
            # Fill NaN room_id with empty string for clean JSON output
            self.df['room_id'] = self.df['room_id'].fillna('')
        else:
            self.df['is_online'] = False

        # Clean lecturer names
        if 'lecturer' in self.df.columns:
            self.df['lecturer'] = self.df['lecturer'].apply(self._clean_name)

        # Normalize semester
        if 'semester' in self.df.columns:
            self.df['semester'] = self.df['semester'].apply(self._clean_semester)

        # Normalize SKS
        if 'sks' in self.df.columns:
            self.df['sks'] = pd.to_numeric(self.df['sks'], errors='coerce').fillna(0).astype(int)

        # Rebuild time_slot display
        if 'start_time' in self.df.columns and 'end_time' in self.df.columns:
            self.df['time_slot'] = self.df.apply(
                lambda r: f"{r['start_time']}-{r['end_time']}"
                if pd.notna(r.get('start_time')) and pd.notna(r.get('end_time'))
                else r.get('time_slot', ''),
                axis=1
            )

        # Strip whitespace
        for col in self.df.select_dtypes(['object']).columns:
            self.df[col] = self.df[col].apply(
                lambda x: str(x).strip() if pd.notna(x) else x
            )

        # Drop rows with no course name
        if 'course_name' in self.df.columns:
            self.df = self.df[self.df['course_name'].notna() & (self.df['course_name'].str.strip() != '')]

        # Drop rows with no valid day
        if 'day' in self.df.columns:
            self.df = self.df[self.df['day'].notna() & (self.df['day'].str.strip() != '')]

        # Drop incomplete artifact rows: no class AND no lecturer
        # A valid schedule entry must have at least one of these
        if 'class_name' in self.df.columns and 'lecturer' in self.df.columns:
            empty_cls = (self.df['class_name'].isna() | (self.df['class_name'].astype(str).str.strip() == ''))
            empty_lec = (self.df['lecturer'].isna() | (self.df['lecturer'].astype(str).str.strip() == ''))
            artifact_mask = empty_cls & empty_lec
            n_artifacts = artifact_mask.sum()
            if n_artifacts > 0:
                self.df = self.df[~artifact_mask]
                logger.info("Removed %d incomplete artifact rows (no class & no lecturer).", n_artifacts)

        # Drop incomplete artifact rows: no lecturer AND no room
        # These slip through the above check when class_name is present
        if 'lecturer' in self.df.columns and 'room_id' in self.df.columns:
            empty_lec = (self.df['lecturer'].isna() | (self.df['lecturer'].astype(str).str.strip() == ''))
            empty_room = (self.df['room_id'].isna() | (self.df['room_id'].astype(str).str.strip() == ''))
            artifact_mask2 = empty_lec & empty_room
            n_artifacts2 = artifact_mask2.sum()
            if n_artifacts2 > 0:
                self.df = self.df[~artifact_mask2]
                logger.info("Removed %d incomplete artifact rows (no lecturer & no room).", n_artifacts2)

        # Remove ghost rows BEFORE merge
        self._remove_ghost_rows()

        # Deduplicate exact rows
        subset_cols = ['day', 'start_time', 'end_time', 'room_id', 'course_name', 'class_name', 'lecturer']
        valid_subset = [c for c in subset_cols if c in self.df.columns]
        self.df.drop_duplicates(subset=valid_subset, keep='first', inplace=True)

        # Collapse multi-session duplicates:
        # Same course+class+day+room+lecturer → keep only the earliest session.
        # This removes redundant rows where fill-forward copied course info to
        # every session row, each recomputing its own (overlapping) time range.
        collapse_cols = ['course_name', 'class_name', 'day', 'room_id', 'lecturer']
        valid_collapse = [c for c in collapse_cols if c in self.df.columns]
        if valid_collapse and 'start_time' in self.df.columns:
            before = len(self.df)
            self.df = (
                self.df
                .sort_values('start_time')
                .drop_duplicates(subset=valid_collapse, keep='first')
            )
            after = len(self.df)
            if before > after:
                logger.info("Collapsed %d multi-session duplicate rows.", before - after)

        # Merge team teaching
        self._merge_team_teaching()

        # Remove utility columns (like 'NO')
        drop_cols = [c for c in self.df.columns
                     if c.upper().startswith('NO') and c != 'course_name']
        for c in drop_cols:
            vals = self.df[c].dropna().astype(str)
            if vals.str.match(r'^\d+\.?$').mean() > 0.5:
                self.df.drop(columns=[c], inplace=True)

        # Final sort by day order + start_time
        if 'day' in self.df.columns:
            day_order = {d: i for i, d in enumerate(INDONESIAN_DAYS)}
            self.df['_day_sort'] = self.df['day'].map(day_order).fillna(99)
            sort_extra = [c for c in ['start_time', 'room_id'] if c in self.df.columns]
            self.df.sort_values(['_day_sort'] + sort_extra, inplace=True)
            self.df.drop(columns=['_day_sort'], inplace=True)

        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def _fix_column_shifts(self):
        """Fix rows where columns are shifted using content-based heuristics."""
        if 'course_name' not in self.df.columns:
            return

        rows_fixed = 0
        for idx in self.df.index:
            row = self.df.loc[idx]
            c_name = str(row.get('course_name', '')).strip() if pd.notna(row.get('course_name')) else ''
            cls_name = str(row.get('class_name', '')).strip() if pd.notna(row.get('class_name')) else ''
            lecturer = str(row.get('lecturer', '')).strip() if pd.notna(row.get('lecturer')) else ''
            room = str(row.get('room_id', '')).strip() if pd.notna(row.get('room_id')) else ''

            shifted = False

            # Pattern 1: course_name is a digit (semester), class_name is a real course
            if c_name.isdigit() and is_likely_course_name(cls_name):
                self.df.at[idx, 'semester'] = int(c_name)
                self.df.at[idx, 'course_name'] = cls_name

                if is_likely_class_code(lecturer):
                    self.df.at[idx, 'class_name'] = lecturer
                    self.df.at[idx, 'lecturer'] = ''
                elif is_likely_class_code(room):
                    self.df.at[idx, 'class_name'] = room
                    self.df.at[idx, 'room_id'] = ''
                else:
                    self.df.at[idx, 'class_name'] = ''
                shifted = True

            # Pattern 2: class_name is a course name (shift artifact)
            # and lecturer is empty -> phantom row, mark for deletion
            elif is_likely_course_name(cls_name) and not is_likely_class_code(cls_name):
                if lecturer == '' and not c_name.isdigit():
                    self.df.at[idx, 'course_name'] = np.nan
                    shifted = True

            # Pattern 3: Row has no class_name AND no lecturer
            # These are "shell" rows from severely shifted PDF sections
            elif cls_name == '' and lecturer == '':
                self.df.at[idx, 'course_name'] = np.nan
                shifted = True

            # Pattern 4: lecturer is a digit (leaked sks) AND room_id is empty/NaN
            # This happens when columns shift right: sks->lecturer, room->empty
            elif (lecturer.isdigit() and len(lecturer) <= 2 and
                  (room == '' or pd.isna(row.get('room_id'))) and cls_name == ''):
                self.df.at[idx, 'course_name'] = np.nan
                shifted = True

            if shifted:
                rows_fixed += 1

        if rows_fixed > 0:
            logger.info("Fixed/removed %d column-shifted rows.", rows_fixed)

    def _remove_ghost_rows(self):
        """Remove ghost-duplicates: same slot but one row has empty lecturer."""
        if 'lecturer' not in self.df.columns:
            return

        group_cols = ['day', 'start_time', 'end_time', 'room_id', 'course_name', 'class_name']
        valid_cols = [c for c in group_cols if c in self.df.columns]
        if not valid_cols:
            return

        to_drop = []
        grouped = self.df.groupby(valid_cols, dropna=False)
        for _, group in grouped:
            if len(group) <= 1:
                continue
            has_lec = group['lecturer'].apply(
                lambda x: pd.notna(x) and str(x).strip() != ''
            )
            if has_lec.any() and not has_lec.all():
                ghost_idx = group[~has_lec].index
                to_drop.extend(ghost_idx)

        if to_drop:
            self.df.drop(index=to_drop, inplace=True)
            logger.info("Removed %d ghost rows.", len(to_drop))

    def _merge_team_teaching(self):
        """Merge rows with identical slot but different lecturers."""
        group_cols = ['day', 'start_time', 'end_time', 'room_id', 'course_name',
                      'class_name', 'session_id', 'semester', 'sks']
        valid_group = [c for c in group_cols if c in self.df.columns]

        if not valid_group or 'lecturer' not in self.df.columns:
            return

        def join_lecturers(series):
            unique = set()
            for item in series:
                if pd.notna(item) and str(item).strip():
                    for part in str(item).split(','):
                        clean = part.strip()
                        if clean:
                            unique.add(clean)
            return ', '.join(sorted(unique)) if unique else ''

        agg_dict = {'lecturer': join_lecturers}
        for col in self.df.columns:
            if col not in valid_group and col != 'lecturer':
                agg_dict[col] = 'first'

        before = len(self.df)
        self.df = self.df.groupby(valid_group, as_index=False, dropna=False).agg(agg_dict)
        after = len(self.df)

        if before > after:
            logger.info("Merged %d team-teaching rows.", before - after)

    @staticmethod
    def _clean_name(name):
        """Clean lecturer name: remove titles, fix punctuation."""
        if pd.isna(name) or str(name).strip() == '':
            return ''
        cleaned = str(name).strip()
        for pattern in TITLE_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[,.]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        parts = [p.strip() for p in cleaned.split() if len(p.strip()) > 1]
        result = ' '.join(parts).strip()
        return result.title() if result else ''

    @staticmethod
    def _clean_semester(val):
        """Normalize semester value to integer."""
        if pd.isna(val) or str(val).strip() == '':
            return 0
        s = str(val).strip().upper()
        try:
            v = int(float(s))
            if 1 <= v <= 14:
                return v
            return 0
        except (ValueError, TypeError):
            pass
        clean_s = re.sub(r'[^IVX]', '', s)
        return ROMAN_MAP.get(clean_s, 0)

    # ── Output ─────────────────────────────────────────────────

    def to_json(self) -> List[Dict[str, Any]]:
        """Convert cleaned DataFrame to list of dicts (JSON-safe, no NaN)."""
        if self.df is None:
            return []

        output_cols = [
            'day', 'start_time', 'end_time', 'session_id', 'time_slot',
            'course_name', 'class_name', 'semester', 'sks',
            'lecturer', 'room_id', 'is_online'
        ]
        for col in self.df.columns:
            if col not in output_cols and col not in ('_day_sort',):
                output_cols.append(col)

        available = [c for c in output_cols if c in self.df.columns]
        out = self.df[available].copy()
        # Replace NaN/NaT with None for valid JSON serialisation
        out = out.where(out.notna(), None)
        records = out.to_dict(orient='records')
        # Safety net: convert any remaining float NaN to None
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and pd.isna(v):
                    rec[k] = None
        return records