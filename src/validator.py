"""
Arah.AI Schedule Validator
===========================
Automated integrity, logic, and quality checks for extracted schedule data.
"""

import pandas as pd
from datetime import datetime

VALID_DAYS = {'SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU'}
IGNORED_ROOMS = {'DARING', 'ONLINE', 'ZOOM', 'MEET', 'RPS', 'TUBEL', 'MAGANG',
                 'VIRTUAL', 'GMEET', 'TEAMS', 'MAYA'}


def validate_extraction(data):
    """
    Runs comprehensive automated checks on the extracted schedule data.
    Returns (markdown_report: str, success: bool).
    """
    if not data:
        return "## Validation Failed\nNo data to validate.", False

    df = pd.DataFrame(data)
    errors = []
    warnings = []
    stats = {}

    stats['total_rows'] = len(df)

    # ── Check 1: Required Fields ────────────────────────────
    required_cols = ['course_name', 'day', 'start_time', 'end_time']
    missing_count = 0
    for row_idx, row in df.iterrows():
        missing = [col for col in required_cols
                   if col not in row or pd.isna(row[col]) or str(row[col]).strip() == ""]
        if missing:
            missing_count += 1
            if missing_count <= 10:  # Limit output
                errors.append(f"- Row {row_idx}: Missing fields: {', '.join(missing)} "
                              f"(course: {row.get('course_name', '?')})")
    if missing_count > 10:
        errors.append(f"- ... and {missing_count - 10} more rows with missing fields.")
    stats['missing_fields'] = missing_count

    # ── Check 2: Time Logic (Start < End) ───────────────────
    time_errors = 0
    if 'start_time' in df.columns and 'end_time' in df.columns:
        for row_idx, row in df.iterrows():
            start, end = row.get('start_time'), row.get('end_time')
            if pd.notna(start) and pd.notna(end) and str(start).strip() and str(end).strip():
                try:
                    s_dt = datetime.strptime(str(start), '%H:%M')
                    e_dt = datetime.strptime(str(end), '%H:%M')
                    if s_dt >= e_dt:
                        time_errors += 1
                        if time_errors <= 5:
                            errors.append(f"- Row {row_idx}: Time error — {start} >= {end} "
                                          f"({row.get('course_name', '?')})")
                except ValueError:
                    pass
    if time_errors > 5:
        errors.append(f"- ... and {time_errors - 5} more time logic errors.")
    stats['time_errors'] = time_errors

    # ── Check 3: Invalid Day Values ─────────────────────────
    if 'day' in df.columns:
        invalid_days = df[~df['day'].isin(VALID_DAYS) & df['day'].notna()]
        if len(invalid_days) > 0:
            bad_vals = invalid_days['day'].value_counts().head(5).to_dict()
            for val, count in bad_vals.items():
                errors.append(f"- Invalid day value: '{val}' ({count} rows)")
        stats['invalid_days'] = len(invalid_days)

    # ── Check 4: Room Conflicts ─────────────────────────────
    room_conflicts = 0
    if all(c in df.columns for c in ['room_id', 'day', 'start_time']):
        dupes = df[df.duplicated(subset=['room_id', 'day', 'start_time'], keep=False)]
        if not dupes.empty:
            grouped = dupes.groupby(['room_id', 'day', 'start_time'])
            for (room, day, time), group in grouped:
                room_upper = str(room).upper()
                if any(ign in room_upper for ign in IGNORED_ROOMS):
                    continue
                courses = group['course_name'].unique()
                if len(courses) > 1 and room:
                    room_conflicts += 1
                    if room_conflicts <= 10:
                        errors.append(f"- Room conflict: {room} on {day} at {time} → "
                                      f"{', '.join(str(c) for c in courses)}")
        if room_conflicts > 10:
            errors.append(f"- ... and {room_conflicts - 10} more room conflicts.")
    stats['room_conflicts'] = room_conflicts

    # ── Check 5: SKS Validation ─────────────────────────────
    sks_errors = 0
    if 'sks' in df.columns:
        for row_idx, row in df.iterrows():
            try:
                sks = int(row['sks'])
                if not (1 <= sks <= 6):
                    sks_errors += 1
                    if sks_errors <= 5:
                        warnings.append(f"- Row {row_idx}: SKS={sks} out of range [1-6] "
                                        f"({row.get('course_name', '?')})")
            except (ValueError, TypeError):
                pass
    if sks_errors > 5:
        warnings.append(f"- ... and {sks_errors - 5} more SKS issues.")
    stats['sks_errors'] = sks_errors

    # ── Check 6: Empty Lecturer ─────────────────────────────
    if 'lecturer' in df.columns:
        empty_lec = df[(df['lecturer'].isna()) | (df['lecturer'].astype(str).str.strip() == '')]
        if len(empty_lec) > 0:
            warnings.append(f"- {len(empty_lec)} rows have empty lecturer field.")
        stats['empty_lecturer'] = len(empty_lec)

    # ── Check 7: Empty Class Name ───────────────────────────
    if 'class_name' in df.columns:
        empty_cls = df[(df['class_name'].isna()) | (df['class_name'].astype(str).str.strip() == '')]
        if len(empty_cls) > 0:
            warnings.append(f"- {len(empty_cls)} rows have empty class_name field.")
        stats['empty_class'] = len(empty_cls)

    # ── Check 8: Unrealistic End Times ──────────────────────
    late_count = 0
    if 'end_time' in df.columns:
        for _, row in df.iterrows():
            et = row.get('end_time')
            if pd.notna(et):
                try:
                    if datetime.strptime(str(et), '%H:%M') > datetime.strptime('22:30', '%H:%M'):
                        late_count += 1
                except ValueError:
                    pass
    if late_count > 0:
        warnings.append(f"- {late_count} rows have end_time > 22:30.")
    stats['late_end_time'] = late_count

    # ── Check 9: Duplicate Courses ──────────────────────────
    dup_cols = ['day', 'start_time', 'end_time', 'room_id', 'course_name', 'class_name']
    valid_dup = [c for c in dup_cols if c in df.columns]
    if valid_dup:
        dupes = df[df.duplicated(subset=valid_dup, keep=False)]
        if len(dupes) > 0:
            warnings.append(f"- {len(dupes)} potential duplicate rows detected.")
        stats['duplicates'] = len(dupes)

    # ── Generate Report ─────────────────────────────────────
    report = ["# Validation Report", ""]
    report.append("## Summary Statistics")
    report.append(f"- **Total Rows**: {stats['total_rows']}")
    if 'day' in df.columns:
        report.append(f"- **Days covered**: {', '.join(sorted(df['day'].dropna().unique()))}")
    if 'course_name' in df.columns:
        report.append(f"- **Unique Courses**: {df['course_name'].nunique()}")
    if 'lecturer' in df.columns:
        report.append(f"- **Unique Lecturers**: {df['lecturer'].replace('', pd.NA).dropna().nunique()}")
    report.append("")

    total_issues = len(errors)
    total_warnings = len(warnings)
    success = total_issues == 0

    if success and total_warnings == 0:
        report.append("## Result: ✅ PASSED")
        report.append("All structural and logical checks passed with no warnings.")
    elif success:
        report.append("## Result: ⚠️ PASSED WITH WARNINGS")
        report.append(f"No critical errors. {total_warnings} warning(s) found.")
        report.append("\n### Warnings:")
        report.extend(warnings)
    else:
        report.append(f"## Result: ❌ FAILED")
        report.append(f"Found **{total_issues} error(s)** and **{total_warnings} warning(s)**.")
        if errors:
            report.append("\n### Errors:")
            report.extend(errors)
        if warnings:
            report.append("\n### Warnings:")
            report.extend(warnings)

    return "\n".join(report), success
