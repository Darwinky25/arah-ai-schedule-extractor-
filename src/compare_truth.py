import json
import pandas as pd

def compare_results(extracted_path, ground_truth_path):
    """
    Compares the extracted JSON against a manually verified Ground Truth JSON.
    """
    try:
        with open(extracted_path, 'r') as f:
            extracted = json.load(f)
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
    except FileNotFoundError as e:
        return f"Error: File not found - {e}"

    df_ext = pd.DataFrame(extracted)
    df_truth = pd.DataFrame(ground_truth)

    # Normalize for comparison (sort by key fields to align rows)
    sort_keys = ['course_name', 'day', 'start_time'] 
    # Ensure columns exist before sorting
    sort_keys = [k for k in sort_keys if k in df_ext.columns and k in df_truth.columns]
    
    if not sort_keys:
        return "Critical Error: Cannot align datasets for comparison (missing keys)."

    df_ext = df_ext.sort_values(by=sort_keys).reset_index(drop=True)
    df_truth = df_truth.sort_values(by=sort_keys).reset_index(drop=True)

    # Compare
    total_cells = df_truth.size
    matching_cells = 0
    mismatches = []

    # Iterate matching rows size (in case length differs)
    rows_to_check = min(len(df_ext), len(df_truth))
    
    # Check length mismatch
    if len(df_ext) != len(df_truth):
        mismatches.append(f"Row Count Mismatch: Extracted {len(df_ext)}, Truth {len(df_truth)}")

    col_mappings = {}
    for col in df_truth.columns:
        if col in df_ext.columns:
            col_mappings[col] = col
    
    for i in range(rows_to_check):
        for true_col, ext_col in col_mappings.items():
            val_ext = str(df_ext.at[i, ext_col]).strip()
            val_true = str(df_truth.at[i, true_col]).strip()
            
            if val_ext == val_true:
                matching_cells += 1
            else:
                mismatches.append(f"Mismatch at Row {i} Col '{true_col}': Expected '{val_true}', Got '{val_ext}'")
    
    # Adjust total based on columns actually compared
    total_compared_cells = rows_to_check * len(col_mappings)
    accuracy = (matching_cells / total_compared_cells * 100) if total_compared_cells > 0 else 0

    report = [
        f"# Accuracy Report",
        f"Accuracy Score: **{accuracy:.2f}%**",
        f"\n### Mismatches ({len(mismatches)}):"
    ]
    report.extend(mismatches[:50]) # Limit output
    if len(mismatches) > 50:
        report.append(f"... and {len(mismatches) - 50} more.")

    return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) == 3:
        print(compare_results(sys.argv[1], sys.argv[2]))
    else:
        print("Usage: python compare_truth.py <extracted.json> <truth.json>")
