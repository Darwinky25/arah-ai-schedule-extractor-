import pdfplumber
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta

class ScheduleExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.raw_data = []
        self.df = None
        self.header_map = {
            'MATA KULIAH': 'course_name',
            'MATKUL': 'course_name',
            'COURSE': 'course_name',
            'SUBJEK': 'course_name',
            'NAMA_MK': 'course_name',
            'JAM': 'time_slot',
            'WAKTU': 'time_slot',
            'PUKUL': 'time_slot',
            'HARI': 'day',
            'SKS': 'sks',
            'KLS': 'class_name',
            'KELAS': 'class_name',
            'SEM': 'semester',
            'SEMESTER': 'semester',
            'RUANG': 'room_id',
            'R.': 'room_id',
            'DOSEN': 'lecturer',
            'PENGAJAR': 'lecturer',
            'NAMA_DOSEN': 'lecturer',
            'SESI': 'session_id',
            'KLS': 'class_name',
            'SM': 'semester'
        }

    def extract_raw(self):
        """
        Module 1: Table Extraction Engine
        Detects tables, extracts raw data, and handles partial tables.
        """
        all_rows = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                # Extract table with default settings
                # We assume the table structure is grid-like enough for default extraction
                # or modify horizontal_strategy if needed.
                tables = page.extract_tables()
                
                for table in tables:
                    # Clean up: remove empty rows that might be artifacts
                    cleaned_table = [row for row in table if any(cell and cell.strip() for cell in row)]
                    all_rows.extend(cleaned_table)
        
        self.raw_data = all_rows
        return self.raw_data

    def normalize_data(self):
        """
        Module 2: Normalization & Fill-Forward Logic
        """
        if not self.raw_data:
            print("No raw data to normalize.")
            return

        # Create DataFrame
        # Step 1: Identify Header Row
        header_index = -1
        keywords = ['MATA KULIAH', 'JAM', 'SKS', 'HARI', 'H A R I']
        
        for i, row in enumerate(self.raw_data):
            row_str = [str(x).upper() if x else "" for x in row]
            matches = sum(1 for k in keywords if any(k in cell for cell in row_str))
            if matches >= 2:
                header_index = i
                break
        
        if header_index == -1:
            raise ValueError("Could not automatically detect header row.")

        headers = self.raw_data[header_index]
        # Clean headers (remove newlines, strip)
        headers = [str(h).replace('\n', ' ').strip().upper() for h in headers]
        
        # Ensure unique headers
        seen = {}
        unique_headers = []
        for h in headers:
            normalized_h = h if h else "UNKNOWN"
            if normalized_h in seen:
                seen[normalized_h] += 1
                unique_headers.append(f"{normalized_h}_{seen[normalized_h]}")
            else:
                seen[normalized_h] = 0
                unique_headers.append(normalized_h)
        headers = unique_headers
        
        data_rows = self.raw_data[header_index+1:]
        
        # Align rows
        aligned_rows = []
        expected_len = len(headers)
        for row in data_rows:
            if row is None: continue
            curr_len = len(row)
            if curr_len == expected_len:
                aligned_rows.append(row)
            elif curr_len < expected_len:
                aligned_rows.append(row + [None] * (expected_len - curr_len))
            else:
                aligned_rows.append(row[:expected_len])

        self.df = pd.DataFrame(aligned_rows, columns=headers)
        
        # Rename columns using map
        new_columns = {}
        for col in self.df.columns:
            mapped_col = col 
            for key, val in self.header_map.items():
                if key in col:
                    mapped_col = val
                    break
            new_columns[col] = mapped_col
        self.df.rename(columns=new_columns, inplace=True)
        
        # --- PHASE 2: GLOBAL CLEANING & REPAIR ---
        
        # 1. Clean format (remove newlines globally first)
        self.df = self.df.replace(r'\n', '', regex=True)
        # Fix potential "None" string artifacts which block ffill
        self.df.replace(['None', 'none', 'enoN'], np.nan, inplace=True)
        
        # 2. Filter Explicit Header Rows (Repeated headers in PDF pages)
        # Drop rows where course_name is "MATA KULIAH" or similar
        header_keywords = ['MATA KULIAH', 'MATKUL', 'COURSE', 'NAMA MK', 'SUBJEK']
        
        def is_header_row(row):
            # Check if course matches header keywords
            if 'course_name' in row and row['course_name']:
                val = str(row['course_name']).upper().strip()
                if any(k in val for k in header_keywords):
                    return True
            # Check if NO column is literal "NO" or "NOMOR"
            for col in row.index:
                if 'NO' in col.upper():
                    val = str(row[col]).upper().strip()
                    if val in ['NO', 'NO.', 'NOMOR']:
                        return True
            return False

        self.df = self.df[~self.df.apply(is_header_row, axis=1)]

        # 3. Filter Garbage Rows (e.g., Column Numbers row: "1", "2", "3")
        # Heuristic: If >50% of columns are single digits, drop it.
        def is_garbage(row):
            numeric_cells = 0
            count_cells = 0
            for x in row:
                if pd.notna(x) and str(x).strip().isdigit() and len(str(x).strip()) <= 2:
                    numeric_cells += 1
                if pd.notna(x):
                    count_cells += 1
            if count_cells > 0 and (numeric_cells / count_cells) > 0.5:
                return True
            return False

        self.df = self.df[~self.df.apply(is_garbage, axis=1)]

        # 4. Intelligent Text Repair (Handling Vertical/Reversed text)
        # Check 'day' column for known patterns
        known_days = {'SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU'}
        
        if 'day' in self.df.columns:
            # Check a sample of non-null values
            sample_days = self.df['day'].dropna().astype(str).head(20).tolist()
            reversal_detected = False
            reversed_matches = 0
            normal_matches = 0
            
            for d in sample_days:
                clean_d = d.strip().upper()
                if clean_d in known_days:
                    normal_matches += 1
                elif clean_d[::-1] in known_days: # Check reversed
                    reversed_matches += 1
            
            # If we see more reversed than normal, flip the whole column
            if reversed_matches > normal_matches:
                print("   [INFO] Detected vertical/reversed text. Applying anti-gravity fix...")
                
                def smart_reverse(x):
                    if pd.isna(x): return np.nan
                    s = str(x).strip()
                    # Handle literal "None" string artifacts
                    if s.lower() == 'none' or s == 'enoN': return np.nan
                    return s[::-1]

                # Apply reversal to Day column
                self.df['day'] = self.df['day'].apply(smart_reverse)
                # Apply reversal to Time column too (highly likely same issue)
                if 'time_slot' in self.df.columns:
                     self.df['time_slot'] = self.df['time_slot'].apply(smart_reverse)
        
        # Post-Repair Cleanup for any missed 'enoN' or 'enoN '
        self.df.replace(['enoN', 'enon', 'None', 'none'], np.nan, inplace=True)
        
        # 1.1 Fix known artifacts in Time Slot (OCR/Extraction glitches)
        # Apply AFTER text repair (flip) so we see the "readable" garbled text
        if 'time_slot' in self.df.columns:
            def repair_time_artifact(val):
                if pd.isna(val): return val
                s = str(val).strip()
                s_cleaned = re.sub(r'\s+', '', s) # Remove spaces for checking

                # Pattern: 119.9.500 0-  -> 19.50-20.40 (?) or 19.00-19.50?
                # Actually, based on logic:
                # XII (12) = 17:40 - 18:30 (from map)
                # XIII (13) = 18.30? 
                
                # Known artifacts based on log:
                # "117. 6.120 - 0" -> 16.20-17.10 (Session XI ?) 
                # "118.7.010 0-" -> 17.10-18.00 (Session XII ?)
                # "119.9.500 0-" -> 19.50 ? No, likely 19.50-something?
                # "210.9.4500-" -> 20.40?
                
                # Let's rely on Session ID logic mainly if time is garbage.
                # Only explicit fix if we are sure.
                
                # Specific fix for common patterns found
                if "117" in s and "6.120" in s: return "16.20-17.10"
                if "118.7" in s: return "17.10-18.00"
                if "119.9" in s: return "18.50-19.40" # Guess based on session XIII?
                
                return s
            self.df['time_slot'] = self.df['time_slot'].apply(repair_time_artifact)

        # Fill-Forward Logic
        # We extend ffill to course info, assuming standard PDF schedule layout where merged cells imply continuation
        self.df.dropna(how='all', inplace=True)
        
        # Extended list of columns to ffill
        # identifying columns: day, time_slot, session_id + course metadata
        ffill_candidates = ['day', 'time_slot', 'session_id', 'course_name', 'sks', 'class_name', 'lecturer', 'room_id']
        cols_to_ffill = [c for c in ffill_candidates if c in self.df.columns]
        
        self.df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        self.df[cols_to_ffill] = self.df[cols_to_ffill].ffill()

        # Recalculate Time based on Session & SKS (User Requirement: 07.00-07.50 is just 1 session)
        if 'session_id' in self.df.columns and 'sks' in self.df.columns:
            self._apply_session_logic()
        elif 'time_slot' in self.df.columns:
             self.df[['start_time', 'end_time']] = self.df['time_slot'].apply(self._split_time).tolist()

        return self.df

    def _apply_session_logic(self):
        """
        Calculates accurate start and end times based on Session ID and SKS.
        Strategy: Map Session ID to a fixed time grid, then span 'SKS' slots.
        """
        # Define Timetables
        # Normal: Mon-Thu, Sat
        schedule_normal = {
            1: ("07:00", "07:50"),
            2: ("07:50", "08:40"),
            3: ("08:40", "09:30"),
            4: ("09:30", "10:20"),
            5: ("10:20", "11:10"),
            6: ("11:10", "12:00"),
            # Break 12:00-13:00
            7: ("13:00", "13:50"),
            8: ("13:50", "14:40"),
            9: ("14:40", "15:30"),
            10: ("15:30", "16:20"),
            11: ("16:20", "17:10"),
            12: ("17:10", "18:00")
        }
        
        # Friday (Jumat): Assuming break ends 13:30 based on data patterns valid for many Indo unis
        schedule_jumat = schedule_normal.copy()
        schedule_jumat.update({
             # Start times shifted for Friday Afternoon?
             # Based on grep: 13.30-14.20, 14.20-15.10
             7: ("13:30", "14:20"),
             8: ("14:20", "15:10"),
             9: ("15:10", "16:00"),
             10: ("16:00", "16:50"),
             11: ("16:50", "17:40"),
             12: ("17:40", "18:30")
        })

        def parse_roman(s):
            if pd.isna(s): return 0
            s = str(s).strip().upper()
            romans = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6, 
                      'VII':7, 'VIII':8, 'IX':9, 'X':10, 'XI':11, 'XII':12,
                      'XIII':13, 'XIV':14, 'XV':15, 'XVI': 16} # Extended just in case
            if '-' in s: s = s.split('-')[0]
            if ',' in s: s = s.split(',')[0]
            return romans.get(s, 0)
            
        def add_minutes(time_str, mins):
            try:
                dt = datetime.strptime(time_str, "%H:%M")
                dt += timedelta(minutes=mins)
                return dt.strftime("%H:%M")
            except:
                return None

        # Clean up specific repeating time artifacts that block parsing
        # E.g., "119.9.500 0-" -> "19.00-19.50" (Guessing based on sequence)
        # But for online classes in evening, we can infer from Session ID + Logic.
        # If raw_start is None, trust the Session logic blindly if known.
        
        # Extend Schedule Grid for EVENING / ONLINE classes
        # Assuming 50 min blocks continuing from session XII (18:30)
        # XIII: 19:00 - 19:50
        # XIV: 19:50 - 20:40
        # XV: 20:40 - 21:30
        
        # Add to normal schedule & Jumat
        evening_slots = {
            13: ("19:00", "19:50"),
            14: ("19:50", "20:40"),
            15: ("20:40", "21:30"),
            16: ("21:30", "22:20")
        }
        schedule_normal.update(evening_slots)
        schedule_jumat.update(evening_slots)

        results = []
        for idx, row in self.df.iterrows():
            day = str(row.get('day', '')).upper()
            session_str = row.get('session_id')
            sks = row.get('sks')
            
            # 1. Get Initial Raw Times from extracted text
            raw_start, raw_end = self._split_time(row.get('time_slot'))
            
            # Artifact Repair if Raw Failed - Also Force Overwrite for Known Garbage
            raw_slot = str(row.get('time_slot'))
            
            # Force mapping for known garbage strings even if split_time somehow returned something
            if "119.9" in raw_slot: raw_start = "19:00"
            elif "210.9" in raw_slot: raw_start = "19:50"
            elif "221.0" in raw_slot: raw_start = "20:40"
            elif "118.7" in raw_slot: raw_start = "17:10"
            
            try:
                sks = int(float(sks)) if pd.notna(sks) else 0
            except:
                sks = 0
            
            # Select Grid
            grid = schedule_jumat if 'JUMAT' in day else schedule_normal
            session_num = parse_roman(session_str)
            
            final_start = None
            final_end = None

            # 2. Logic Decision Tree
            
            # A. If we have a Grid Match for the Session ID
            if session_num in grid:
                grid_start, _ = grid[session_num]
                
                # Check if Raw Start deviates significantly from Grid Start
                use_grid = True
                
                if raw_start:
                    try:
                        dt_grid = datetime.strptime(grid_start, "%H:%M")
                        dt_raw = datetime.strptime(raw_start, "%H:%M")
                        diff = abs((dt_raw - dt_grid).total_seconds() / 60)
                        
                        # Trust Raw if diff > 45 mins. 
                        # This allows 19:00 online class to override 18:30 logic if it's explicitly 19:00
                        if diff > 45: 
                            use_grid = False
                    except:
                        pass 
                
                if use_grid:
                    final_start = grid_start
                    end_session_num = session_num + sks - 1
                    if end_session_num in grid:
                         final_end = grid[end_session_num][1]
                    else:
                         final_end = add_minutes(final_start, sks * 50)
                else:
                    final_start = raw_start
                    final_end = add_minutes(final_start, sks * 50)
            
            # B. No Session ID Match
            elif raw_start:
                final_start = raw_start
                if sks > 0:
                     final_end = add_minutes(final_start, sks * 50)
                else:
                     final_end = raw_end
            
            results.append((final_start, final_end))
            
        self.df['start_time'] = [r[0] for r in results]
        self.df['end_time'] = [r[1] for r in results]

    def _split_time(self, time_str):
        """
        Helper to split time string using Regex
        e.g., '07.00-07.50' -> ('07:00', '07:50')
        """
        if pd.isna(time_str):
            return None, None
            
        time_str = str(time_str).strip()
        # Handle typo/format variations like "07.00.07.50" or "07.0007.50" or "07.00 - 07.50"
        
        # Strategy: Look for 4 consecutive digits twice, usually separated by something
        # Matches 07.00, 07:00, 7.00
        # Pattern: (H:M) separator (H:M)
        
        # Clean dots to colons for standardization
        # But be careful, sometimes dots are separators. 07.00 -> 07:00
        
        # Robust Regex: (\d{1,2}[.:]\d{2}).*?(\d{1,2}[.:]\d{2})
        pattern = r'(\d{1,2}[:.]\d{2})\s*[-–]?\s*(\d{1,2}[:.]\d{2})'
        match = re.search(pattern, time_str)
        
        if match:
            start, end = match.group(1), match.group(2)
            return start.replace('.', ':'), end.replace('.', ':')
            
        return None, None

    def clean_data(self):
        """
        Module 3: Data Cleaning
        """
        if self.df is None:
            return

        # PHASE 0: Fix Structural Shifts
        # Must be done before specific column cleaning
        self._fix_column_shifts()

        # 0. Identify Online Classes ("Daring")
        if 'room_id' in self.df.columns:
            self.df['is_online'] = self.df['room_id'].astype(str).str.contains(r'daring|online|zoom|meet|maya', case=False, na=False)
        else:
             self.df['is_online'] = False

        # 1. Clean Instructor Names
        if 'lecturer' in self.df.columns:
            self.df['lecturer'] = self.df['lecturer'].apply(self._clean_name)

        # 2. Fix Semester (Handle Roman Numerals & Int Casting)
        if 'semester' in self.df.columns:
            def clean_semester(val):
                if pd.isna(val) or str(val).strip() == '': return 0
                s = str(val).strip().upper()
                # Try Roman map first if non-digit
                if not s.isdigit():
                    romans = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6, 
                              'VII':7, 'VIII':8, 'IX':9, 'X':10}
                    # Check partial matches like "V," or "III-"
                    clean_s = re.sub(r'[^IVX]', '', s)
                    if clean_s in romans:
                        return romans[clean_s]
                # Default numeric conversion
                try:
                    return int(float(s))
                except:
                    return 0
            
            self.df['semester'] = self.df['semester'].apply(clean_semester)

        if 'sks' in self.df.columns:
             self.df['sks'] = pd.to_numeric(self.df['sks'], errors='coerce').fillna(0).astype(int)

        # 3. Clean Time Slot Display (Update garbage text with calculated time)
        # Ensure we use the format "HH:MM-HH:MM" if start/end exists
        if 'start_time' in self.df.columns and 'end_time' in self.df.columns:
            def update_slot_display(row):
                if pd.notna(row['start_time']) and pd.notna(row['end_time']):
                     return f"{row['start_time']}-{row['end_time']}"
                return row['time_slot']
            
            self.df['time_slot'] = self.df.apply(update_slot_display, axis=1)

        # Strip whitespaces from all string columns
        df_obj = self.df.select_dtypes(['object'])
        self.df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        
        # Drop rows where essential data might be missing
        if 'course_name' in self.df.columns:
            self.df = self.df.dropna(subset=['course_name'])

        # Final Step: Deduplication & Team Teaching Merge
        # 1. Deduplicate exact rows first
        subset_cols = ['day', 'start_time', 'room_id', 'course_name', 'class_name', 'lecturer']
        valid_subset = [c for c in subset_cols if c in self.df.columns]
        
        self.df.drop_duplicates(subset=valid_subset, keep='first', inplace=True)
        
        # 2. Merge Team Teaching (Same slot/room/course, diff lecturer)
        self._merge_team_teaching()

        return self.df

    def _clean_name(self, name):
        if pd.isna(name):
            return ""
        
        cleaned = str(name).strip()
        
        # 1. Cleaning Titles (Optional strategy: strip specific titles, or keep them?)
        # User asked to "Remove leftover characters" like ", ,"
        # Strategy: Remove specific titles might be too aggressive if user wants them.
        # But assuming the goal is just the Name.
        
        titles = [
            r'\bDr\.', r'\bDra\.', r'\bIr\.', r'\bS\.T\.', r'\bM\.T\.', 
            r'\bS\.Kom\.', r'\bM\.Kom\.', r'\bPh\.D\.', r'\bProf\.', 
            r'\bH\.', r'\bHjb\.', r'\bAssoc\b',
            r'\bS\.H\.', r'\bM\.H\.', r'\bS\.E\.', r'\bM\.M\.', r'\bM\.Si\.'
        ]
        
        # Remove titles
        for title in titles:
            cleaned = re.sub(title, '', cleaned, flags=re.IGNORECASE)
            
        # 2. Fix Punctuation Nightmares (", ,", ",.", "MH")
        # Replace multiple commas/dots with single space or comma
        cleaned = re.sub(r'[,.]+', ' ', cleaned) 
        
        # 3. Remove non-alphabet characters at ends (like trailing commas)
        # Keep only letters and spaces internal
        # Actually, names can have dashes "Delik-delik"
        
        # Simple heuristic: Split by space, allow only words
        # Remove empty tokens
        parts = [p.strip() for p in cleaned.split() if len(p.strip()) > 1]
        
        return ' '.join(parts).title()

    def _fix_column_shifts(self):
        """
        Fixes rows where Semester, Course, SKS, Class Name columns are shifted.
        Pattern: Course Name is '1' (Semester key), Class Name is 'Course Title'.
        """
        def fix_row(row):
            c_name = str(row.get('course_name', ''))
            cls_name = str(row.get('class_name', ''))
            
            # Heuristic: Course Name is numeric, Class Name seems like a Title
            if c_name.strip().isdigit() and len(cls_name) > 3:
                try:
                   row['semester'] = int(c_name)
                except:
                   pass
                row['course_name'] = cls_name
                
                # Recover Class Name from Lecturer if possible
                lect = str(row.get('lecturer', '')).strip()
                if len(lect) == 1 and lect.isalpha():
                    row['class_name'] = lect
                    row['lecturer'] = "" 
                else:
                    # If lecturer is empty or not a class char, set to Unknown
                    row['class_name'] = "Unknown"
            return row
            
        self.df = self.df.apply(fix_row, axis=1)

    def _merge_team_teaching(self):
        """
        Merges rows with identical Day, Time, Room, and Course Name.
        Combines different lecturers into one list/string.
        """
        # Key columns for grouping
        group_cols = ['day', 'start_time', 'end_time', 'room_id', 'course_name', 'class_name', 'session_id', 'semester', 'sks']
        # Only keep columns that actually exist
        valid_group = [c for c in group_cols if c in self.df.columns]
        
        if not valid_group or 'lecturer' not in self.df.columns:
            return

        # Define aggregation for non-group columns
        def join_lecturers(x):
            unique = set()
            for item in x:
                if pd.notna(item) and str(item).strip():
                    parts = str(item).split(',')
                    for p in parts:
                        clean_p = p.strip()
                        if clean_p: unique.update([clean_p])
            return ', '.join(sorted(list(unique)))

        agg_dict = {'lecturer': join_lecturers}
        
        for col in self.df.columns:
            if col not in valid_group and col != 'lecturer':
                agg_dict[col] = 'first'

        before_count = len(self.df)
        self.df = self.df.groupby(valid_group, as_index=False).agg(agg_dict)
        after_count = len(self.df)
        
        if before_count > after_count:
             print(f"   [INFO] Merged {before_count - after_count} team-teaching rows.")

    def to_json(self):
        result = self.df.to_dict(orient='records')
        return result
