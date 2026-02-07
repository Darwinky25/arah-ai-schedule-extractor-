"""
Arah.AI Universal Schedule Extractor — CLI Entry Point
========================================================
Batch-processes PDF schedules from a configurable input directory
and writes JSON + validation reports to an output directory.
"""

import argparse
import json
import logging
import os
import warnings
from typing import List, Optional, Tuple

from extractor import ScheduleExtractor
from validator import validate_extraction

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def find_project_root() -> Optional[str]:
    """Find the project root by locating the 'jadwal' directory."""
    candidates = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "jadwal")):
            return root
    return None


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    index: Optional[int] = None,
) -> Tuple[Optional[list], bool]:
    """Process a single PDF file and return (data, success)."""
    basename = os.path.basename(pdf_path)
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{'='*60}")
    print(f"{prefix}📄 Processing: {basename}")
    print(f"{'='*60}")

    try:
        extractor = ScheduleExtractor(pdf_path)

        print("   - Module 1: Extracting Table Data...")
        extractor.extract_raw()

        print("   - Module 2: Normalizing & Repairing Data...")
        extractor.normalize_data()

        print("   - Module 3: Cleaning & Deduplicating...")
        extractor.clean_data()

        data = extractor.to_json()
        print(f"   ✅ Extraction Complete. {len(data)} rows extracted.")

    except Exception as e:
        logger.exception("Extraction failed for %s", basename)
        print(f"   ❌ Error during extraction: {e}")
        return None, False

    # Generate output filenames
    name_stem = os.path.splitext(basename)[0]
    safe_name = name_stem[:60].strip().replace(" ", "_")

    if index is not None:
        json_filename = f"extracted_{index}_{safe_name}.json"
        report_filename = f"validation_{index}_{safe_name}.md"
    else:
        json_filename = "extracted_schedule.json"
        report_filename = "validation_report.md"

    # Save JSON
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"   💾 Data saved: {json_filename}")

    # Validate
    print("   🔍 Running Validator...")
    result = validate_extraction(data)

    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.report)
    print(f"   📝 Report saved: {report_filename}")

    if result.success:
        print("   ✨ PASSED: All checks passed.")
    else:
        print("   ⚠️  WARNING: Issues found. See report.")

    return data, result.success


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Arah.AI Universal Schedule Extractor v2.1 — "
                    "Extract Indonesian university schedule PDFs to JSON.",
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=None,
        help="Directory containing PDF file(s). Default: <project_root>/jadwal",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for JSON + validation output. Default: <project_root>/output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("🚀 Arah.AI Universal Schedule Extractor v2.1")
    print("   Supports any Indonesian university schedule PDF\n")

    # Find project root
    base_dir = find_project_root()
    if not base_dir:
        print("❌ Error: Could not locate 'jadwal' directory.")
        print(f"   Checked: {os.getcwd()}")
        return

    jadwal_dir = args.input_dir or os.path.join(base_dir, "jadwal")
    output_dir = args.output_dir or os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(jadwal_dir):
        print(f"❌ Input directory not found: {jadwal_dir}")
        return

    # Find all PDFs
    pdf_files = sorted(
        f for f in os.listdir(jadwal_dir) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"❌ No PDF files found in '{jadwal_dir}'.")
        print("   Place your schedule PDF(s) in the folder and re-run.")
        return

    print(f"📂 Found {len(pdf_files)} PDF file(s) in {jadwal_dir}")

    # Process each PDF
    all_data: List[dict] = []
    all_success = True

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(jadwal_dir, pdf_file)
        use_index = i if len(pdf_files) > 1 else None
        data, success = process_single_pdf(pdf_path, output_dir, index=use_index)

        if data:
            all_data.extend(data)
        if not success:
            all_success = False

    # If multiple PDFs, also save a combined output
    if len(pdf_files) > 1 and all_data:
        combined_path = os.path.join(output_dir, "extracted_combined.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"\n📦 Combined output ({len(all_data)} rows): extracted_combined.json")

    # Final summary
    print(f"\n{'='*60}")
    print(f"🏁 Done. Processed {len(pdf_files)} file(s), {len(all_data)} total rows.")
    if all_success:
        print("✨ All files passed validation.")
    else:
        print("⚠️  Some files had issues. Check validation reports.")


if __name__ == "__main__":
    main()
