"""
Arah.AI Ground-Truth Comparator
=================================
Compare extracted JSON against a manually-verified ground-truth JSON and
report cell-level accuracy.

v2.1 — Key-based row alignment, type-safe comparison, per-column accuracy.
"""

import json
import sys
from typing import Dict, List, Optional

import pandas as pd


# Key columns used to align extracted rows with ground-truth rows.
ALIGN_KEYS = ['course_name', 'day', 'start_time', 'class_name']


def _normalise(val: object) -> str:
    """Normalise a value to a comparable string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val).strip()
    # Treat numeric strings uniformly: "3.0" → "3"
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def compare_results(
    extracted_path: str,
    ground_truth_path: str,
) -> str:
    """
    Compare the extracted JSON against a manually verified ground-truth JSON.

    Alignment is done by merging on key columns rather than relying on sort
    order, which avoids misalignment when the row counts differ.
    """
    try:
        with open(extracted_path, 'r', encoding='utf-8') as f:
            extracted = json.load(f)
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except FileNotFoundError as e:
        return f"Error: File not found — {e}"

    df_ext = pd.DataFrame(extracted)
    df_truth = pd.DataFrame(ground_truth)

    # Decide which key columns are available for alignment
    align_cols = [c for c in ALIGN_KEYS if c in df_ext.columns and c in df_truth.columns]
    if not align_cols:
        return "Critical Error: Cannot align datasets for comparison (no shared key columns)."

    # Merge on key columns (inner) so only matched rows are compared
    df_ext['_src'] = 'ext'
    df_truth['_src'] = 'truth'

    # Tag columns that exist in both
    common_cols = sorted(set(df_ext.columns) & set(df_truth.columns) - {'_src'})

    merged = pd.merge(
        df_truth[common_cols],
        df_ext[common_cols],
        on=align_cols,
        how='outer',
        suffixes=('_truth', '_ext'),
        indicator=True,
    )

    matched = merged[merged['_merge'] == 'both']
    only_truth = merged[merged['_merge'] == 'left_only']
    only_ext = merged[merged['_merge'] == 'right_only']

    # Compare cell values for matched rows
    compare_cols = [c for c in common_cols if c not in align_cols]
    total_cells = 0
    matching_cells = 0
    mismatches: List[str] = []
    per_col_match: Dict[str, int] = {c: 0 for c in compare_cols}
    per_col_total: Dict[str, int] = {c: 0 for c in compare_cols}

    for idx, row in matched.iterrows():
        for col in compare_cols:
            truth_col = f"{col}_truth"
            ext_col = f"{col}_ext"
            if truth_col not in row or ext_col not in row:
                continue
            val_truth = _normalise(row[truth_col])
            val_ext = _normalise(row[ext_col])
            total_cells += 1
            per_col_total[col] = per_col_total.get(col, 0) + 1
            if val_truth == val_ext:
                matching_cells += 1
                per_col_match[col] = per_col_match.get(col, 0) + 1
            else:
                key_desc = ' | '.join(str(row.get(k, '?')) for k in align_cols if k in row)
                mismatches.append(
                    f"  [{key_desc}] {col}: expected '{val_truth}', got '{val_ext}'"
                )

    accuracy = (matching_cells / total_cells * 100) if total_cells > 0 else 0

    # Build report
    lines = [
        "# Accuracy Report",
        "",
        f"Overall Accuracy: **{accuracy:.2f}%**  ({matching_cells}/{total_cells} cells)",
        "",
        f"- Rows in ground truth: {len(df_truth)}",
        f"- Rows in extracted:    {len(df_ext)}",
        f"- Matched rows:         {len(matched)}",
        f"- Only in truth:        {len(only_truth)}",
        f"- Only in extracted:    {len(only_ext)}",
        "",
        "## Per-Column Accuracy",
    ]
    for col in compare_cols:
        ct = per_col_total.get(col, 0)
        cm = per_col_match.get(col, 0)
        pct = (cm / ct * 100) if ct > 0 else 0
        lines.append(f"- **{col}**: {pct:.1f}%  ({cm}/{ct})")

    lines.append("")
    lines.append(f"## Mismatches ({len(mismatches)})")
    lines.extend(mismatches[:80])
    if len(mismatches) > 80:
        lines.append(f"  ... and {len(mismatches) - 80} more.")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(compare_results(sys.argv[1], sys.argv[2]))
    else:
        print("Usage: python compare_truth.py <extracted.json> <truth.json>")
