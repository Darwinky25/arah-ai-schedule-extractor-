"""
Arah.AI Schedule Validator
===========================
Automated integrity, logic, and quality checks for extracted schedule data.

v2.1 — Vectorized checks, time-overlap detection, lecturer conflict detection.
"""

import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

VALID_DAYS = {'SENIN', 'SELASA', 'RABU', 'KAMIS', 'JUMAT', 'SABTU', 'MINGGU'}
IGNORED_ROOMS = {'DARING', 'ONLINE', 'ZOOM', 'MEET', 'RPS', 'TUBEL', 'MAGANG',
                 'VIRTUAL', 'GMEET', 'TEAMS', 'MAYA'}

MAX_DISPLAY_ERRORS = 10
MAX_DISPLAY_WARNINGS = 5


class ValidationResult(NamedTuple):
    """Structured result from validate_extraction()."""
    report: str
    success: bool


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _is_ignored_room(room: Any) -> bool:
    """Check if a room should be skipped for conflict detection."""
    if pd.isna(room) or str(room).strip() == '':
        return True
    room_upper = str(room).upper()
    return any(ign in room_upper for ign in IGNORED_ROOMS)


def _detect_time_overlaps(
    df: pd.DataFrame,
    group_col: str,
    label: str,
    limit: int = 10,
) -> List[str]:
    """
    Detect time-range overlaps within groups.

    Works for both room conflicts (same room, same day, overlapping times)
    and lecturer conflicts (same lecturer, same day, overlapping times).
    """
    conflicts: List[str] = []
    required = [group_col, 'day', 'start_time', 'end_time']
    if not all(c in df.columns for c in required):
        return conflicts

    check_df = df.copy()
    check_df = check_df[
        check_df[group_col].notna()
        & (check_df[group_col].astype(str).str.strip() != '')
    ]

    # Skip ignored rooms
    if group_col == 'room_id':
        check_df = check_df[~check_df['room_id'].apply(_is_ignored_room)]

    # Expand team-teaching rows so each lecturer is checked individually
    if group_col == 'lecturer':
        rows = []
        for _, row in check_df.iterrows():
            for lec in str(row['lecturer']).split(','):
                lec = lec.strip()
                if lec:
                    new_row = row.copy()
                    new_row['lecturer'] = lec
                    rows.append(new_row)
        if rows:
            check_df = pd.DataFrame(rows)

    total_found = 0
    for (key, day), group in check_df.groupby([group_col, 'day']):
        if len(group) < 2:
            continue
        entries = []
        for _, row in group.iterrows():
            try:
                s = datetime.strptime(str(row['start_time']), '%H:%M')
                e = datetime.strptime(str(row['end_time']), '%H:%M')
                course = row.get('course_name', '?')
                cls = row.get('class_name', '?')
                entries.append((s, e, course, cls))
            except (ValueError, TypeError):
                continue

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                s1, e1, c1, cl1 = entries[i]
                s2, e2, c2, cl2 = entries[j]
                if s1 == s2 and e1 == e2 and c1 == c2:
                    continue  # combined lecture: same course, same time

                # Overlap condition: s1 < e2 AND s2 < e1
                if s1 < e2 and s2 < e1 and (c1 != c2 or cl1 != cl2):
                    total_found += 1
                    if total_found <= limit:
                        conflicts.append(
                            f"- {label} overlap: **{key}** on {day} — "
                            f"'{c1}' ({s1:%H:%M}-{e1:%H:%M}) overlaps "
                            f"'{c2}' ({s2:%H:%M}-{e2:%H:%M})"
                        )

    if total_found > limit:
        conflicts.append(f"- ... and {total_found - limit} more {label.lower()} overlaps.")
    return conflicts


# ─────────────────────────────────────────────────────────────
# Main validation entry point
# ─────────────────────────────────────────────────────────────

def validate_extraction(data: List[Dict[str, Any]]) -> ValidationResult:
    """
    Run comprehensive automated checks on extracted schedule data.

    Returns ``ValidationResult(report, success)`` where *success* is ``True``
    when no critical errors are found.
    """
    if not data:
        return ValidationResult("## Validation Failed\nNo data to validate.", False)

    df = pd.DataFrame(data)
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {'total_rows': len(df)}

    # ── Check 1: Required Fields (vectorized) ───────────────
    required_cols = ['course_name', 'day', 'start_time', 'end_time']
    missing_per_col: Dict[str, pd.Series] = {}
    for col in required_cols:
        if col in df.columns:
            missing_per_col[col] = df[col].isna() | (df[col].astype(str).str.strip() == '')
        else:
            missing_per_col[col] = pd.Series(True, index=df.index)
    any_missing = pd.concat(missing_per_col, axis=1).any(axis=1)
    missing_count = int(any_missing.sum())
    if missing_count > 0:
        bad_rows = df[any_missing].head(MAX_DISPLAY_ERRORS)
        for row_idx, row in bad_rows.iterrows():
            miss = [c for c in required_cols if missing_per_col[c].loc[row_idx]]
            errors.append(
                f"- Row {row_idx}: Missing fields: {', '.join(miss)} "
                f"(course: {row.get('course_name', '?')})"
            )
        if missing_count > MAX_DISPLAY_ERRORS:
            errors.append(f"- ... and {missing_count - MAX_DISPLAY_ERRORS} more rows with missing fields.")
    stats['missing_fields'] = missing_count

    # ── Check 2: Time Logic — start < end (vectorized) ──────
    time_errors = 0
    if 'start_time' in df.columns and 'end_time' in df.columns:
        valid_mask = (
            df['start_time'].notna() & df['end_time'].notna()
            & (df['start_time'].astype(str).str.strip() != '')
            & (df['end_time'].astype(str).str.strip() != '')
        )
        valid_df = df[valid_mask].copy()
        try:
            starts = pd.to_datetime(valid_df['start_time'], format='%H:%M', errors='coerce')
            ends = pd.to_datetime(valid_df['end_time'], format='%H:%M', errors='coerce')
            bad_time = (starts >= ends) & starts.notna() & ends.notna()
            time_errors = int(bad_time.sum())
            if time_errors > 0:
                for row_idx, row in valid_df[bad_time].head(MAX_DISPLAY_WARNINGS).iterrows():
                    errors.append(
                        f"- Row {row_idx}: Time error — {row['start_time']} >= {row['end_time']} "
                        f"({row.get('course_name', '?')})"
                    )
                if time_errors > MAX_DISPLAY_WARNINGS:
                    errors.append(f"- ... and {time_errors - MAX_DISPLAY_WARNINGS} more time logic errors.")
        except Exception:
            pass
    stats['time_errors'] = time_errors

    # ── Check 3: Invalid Day Values ─────────────────────────
    if 'day' in df.columns:
        invalid_days = df[~df['day'].isin(VALID_DAYS) & df['day'].notna()]
        if len(invalid_days) > 0:
            bad_vals = invalid_days['day'].value_counts().head(5).to_dict()
            for val, count in bad_vals.items():
                errors.append(f"- Invalid day value: '{val}' ({count} rows)")
        stats['invalid_days'] = len(invalid_days)

    # ── Check 4: Room Time-Overlap Conflicts ────────────────
    room_overlaps = _detect_time_overlaps(df, 'room_id', 'Room')
    stats['room_conflicts'] = len(room_overlaps)
    if room_overlaps:
        warnings.extend(room_overlaps)

    # ── Check 5: Lecturer Time-Overlap Conflicts ────────────
    # Lecturer overlaps are classified as informational notes rather than
    # warnings because they reflect source-data scheduling conflicts (e.g.
    # a lecturer assigned to two courses at the same time), not extraction
    # errors.  The extractor faithfully reproduces what the PDF contains.
    lecturer_overlaps = _detect_time_overlaps(df, 'lecturer', 'Lecturer')
    stats['lecturer_conflicts'] = len(lecturer_overlaps)
    notes: List[str] = []
    if lecturer_overlaps:
        notes.extend(lecturer_overlaps)

    # ── Check 6: SKS Validation (vectorized) ────────────────
    sks_errors = 0
    if 'sks' in df.columns:
        sks_numeric = pd.to_numeric(df['sks'], errors='coerce')
        bad_sks = sks_numeric.notna() & ((sks_numeric < 1) | (sks_numeric > 6))
        sks_errors = int(bad_sks.sum())
        if sks_errors > 0:
            for row_idx, row in df[bad_sks].head(MAX_DISPLAY_WARNINGS).iterrows():
                warnings.append(
                    f"- Row {row_idx}: SKS={row['sks']} out of range [1-6] "
                    f"({row.get('course_name', '?')})"
                )
            if sks_errors > MAX_DISPLAY_WARNINGS:
                warnings.append(f"- ... and {sks_errors - MAX_DISPLAY_WARNINGS} more SKS issues.")
    stats['sks_errors'] = sks_errors

    # ── Check 7: Empty Lecturer ─────────────────────────────
    if 'lecturer' in df.columns:
        empty_lec = df[(df['lecturer'].isna()) | (df['lecturer'].astype(str).str.strip() == '')]
        if len(empty_lec) > 0:
            warnings.append(f"- {len(empty_lec)} rows have empty lecturer field.")
        stats['empty_lecturer'] = len(empty_lec)

    # ── Check 8: Empty Class Name ───────────────────────────
    if 'class_name' in df.columns:
        empty_cls = df[(df['class_name'].isna()) | (df['class_name'].astype(str).str.strip() == '')]
        if len(empty_cls) > 0:
            warnings.append(f"- {len(empty_cls)} rows have empty class_name field.")
        stats['empty_class'] = len(empty_cls)

    # ── Check 9: Unrealistic End Times (vectorized) ─────────
    late_count = 0
    if 'end_time' in df.columns:
        try:
            end_times = pd.to_datetime(df['end_time'], format='%H:%M', errors='coerce')
            cap = pd.to_datetime('22:30', format='%H:%M')
            late_count = int((end_times > cap).sum())
        except Exception:
            pass
    if late_count > 0:
        warnings.append(f"- {late_count} rows have end_time > 22:30.")
    stats['late_end_time'] = late_count

    # ── Check 10: Duplicate Courses ─────────────────────────
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
        report.append("## Result: ❌ FAILED")
        report.append(f"Found **{total_issues} error(s)** and **{total_warnings} warning(s)**.")
        if errors:
            report.append("\n### Errors:")
            report.extend(errors)
        if warnings:
            report.append("\n### Warnings:")
            report.extend(warnings)

    # Append info notes (source-data conflicts) regardless of pass/fail
    if notes:
        report.append("\n### Notes (source-data scheduling conflicts):")
        report.extend(notes)

    result = ValidationResult("\n".join(report), success)
    logger.info("Validation complete: %d errors, %d warnings", total_issues, total_warnings)
    return result
