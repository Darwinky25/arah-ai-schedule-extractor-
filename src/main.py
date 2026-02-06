import os
import sys
import json
import warnings
from extractor import ScheduleExtractor
from validator import validate_extraction

warnings.filterwarnings("ignore")


def find_project_root():
    """Find the project root by locating the 'jadwal' directory."""
    candidates = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd()
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, 'jadwal')):
            return root
    return None


def process_single_pdf(pdf_path, output_dir, index=None):
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
        print(f"   ❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, False

    # Generate output filenames
    name_stem = os.path.splitext(basename)[0]
    # Simplify filename for output
    safe_name = name_stem[:60].strip().replace(' ', '_')
    
    if index is not None:
        json_filename = f"extracted_{index}_{safe_name}.json"
        report_filename = f"validation_{index}_{safe_name}.md"
    else:
        json_filename = "extracted_schedule.json"
        report_filename = "validation_report.md"

    # Save JSON
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"   💾 Data saved: {json_filename}")

    # Validate
    print("   🔍 Running Validator...")
    report, success = validate_extraction(data)

    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   📝 Report saved: {report_filename}")

    if success:
        print("   ✨ PASSED: All checks passed.")
    else:
        print("   ⚠️  WARNING: Issues found. See report.")

    return data, success


def main():
    print("🚀 Arah.AI Universal Schedule Extractor v2.0")
    print("   Supports any Indonesian university schedule PDF\n")

    # Find project root
    base_dir = find_project_root()
    if not base_dir:
        print("❌ Error: Could not locate 'jadwal' directory.")
        print(f"   Checked: {os.getcwd()}")
        return

    jadwal_dir = os.path.join(base_dir, 'jadwal')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs
    pdf_files = sorted([f for f in os.listdir(jadwal_dir) if f.lower().endswith('.pdf')])

    if not pdf_files:
        print("❌ No PDF files found in 'jadwal/' directory.")
        print("   Place your schedule PDF(s) in the 'jadwal/' folder and re-run.")
        return

    print(f"📂 Found {len(pdf_files)} PDF file(s) in jadwal/")

    # Process each PDF
    all_data = []
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
        combined_path = os.path.join(output_dir, 'extracted_combined.json')
        with open(combined_path, 'w', encoding='utf-8') as f:
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
