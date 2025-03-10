#!/usr/bin/env python3
"""
Test script for the Financial Data Cleaner package.
"""
import os
import sys
from financial_data_cleaner.core.processor import preprocess_file
from financial_data_cleaner.utils.file_handlers import save_cleaned_data

def main():
    # Path to the test file
    test_file = '/Users/bojantesic/git-tests/Hill-data/AI Training/Fluxs - AI Training/8199668_Isins NAV.2025.01.24.xlsx'
    
    # Verify file exists
    if not os.path.isfile(test_file):
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    print(f"🔍 Testing with file: {test_file}")
    
    # Process the file
    cleaned_data = preprocess_file(test_file, debug=True)
    
    if cleaned_data:
        # Save to output directory
        output_dir = 'cleaned_output'
        os.makedirs(output_dir, exist_ok=True)
        
        output_base = os.path.basename(test_file)
        name, _ = os.path.splitext(output_base)
        output_path = os.path.join(output_dir, f"{name}_cleaned.xlsx")
        
        save_cleaned_data(cleaned_data, output_path)
        print(f"✅ Saved cleaned data to {output_path}")
    else:
        print("❌ Failed to process file")

if __name__ == "__main__":
    main()
