import os
import json
import warnings
from extractor import ScheduleExtractor
from validator import validate_extraction

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("🚀 Starting Arah.AI Schedule Extractor...")

    # Paths
    # Try multiple strategies to find base_dir
    possible_roots = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # Relative to script
        os.getcwd() # Current working directory
    ]
    
    jadwal_dir = None
    base_dir = None
    
    for root in possible_roots:
        check_path = os.path.join(root, 'jadwal')
        if os.path.exists(check_path):
            base_dir = root
            jadwal_dir = check_path
            break
            
    if not jadwal_dir:
        # Debugging info
        print(f"❌ Error: Could not locate 'jadwal' directory.")
        print(f"   Checked roots: {possible_roots}")
        print(f"   Current dir: {os.getcwd()}")
        return

    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find PDF
    pdf_files = [f for f in os.listdir(jadwal_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ No PDF found in 'jadwal/' directory.")
        return

    pdf_to_process = os.path.join(jadwal_dir, pdf_files[0]) # Take the first one
    print(f"📄 Processing: {pdf_files[0]}")

    # 1. Extraction
    try:
        extractor = ScheduleExtractor(pdf_to_process)
        
        print("   - Module 1: Extracting Table Data...")
        extractor.extract_raw()
        
        print("   - Module 2: Normalizing & Fill-Forward...")
        extractor.normalize_data()
        
        print("   - Module 3: Cleaning Data...")
        extractor.clean_data()
        
        data = extractor.to_json()
        print(f"✅ Extraction Complete. Extracted {len(data)} rows.")

    except Exception as e:
        print(f"❌ Critical Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save JSON
    output_json_path = os.path.join(output_dir, 'extracted_schedule.json')
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Data saved to: {output_json_path}")
    
    # 2. Validation
    print("🔍 Running Accuracy Validator...")
    report, success = validate_extraction(data)
    
    output_report_path = os.path.join(output_dir, 'validation_report.md')
    with open(output_report_path, 'w') as f:
        f.write(report)
    
    print(f"📝 Validation Report saved to: {output_report_path}")
    
    if success:
        print("\n✨ SUCCESS: Data passed all automated integrity checks.")
    else:
        print("\n⚠️ WARNING: Data contains potential issues. Check validation report.")

if __name__ == "__main__":
    main()
