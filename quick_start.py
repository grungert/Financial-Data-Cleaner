#!/usr/bin/env python3
"""
Quick Start Script for Financial Data Cleaner

This script demonstrates how to use the Financial Data Cleaner package
to process a financial data file.
"""
import os
import sys
import argparse
from financial_data_cleaner.core.processor import preprocess_file
from financial_data_cleaner.utils.file_handlers import save_cleaned_data
from financial_data_cleaner.utils.dependencies import check_dependencies

def main():
    """
    Main function to demonstrate Financial Data Cleaner functionality.
    """
    parser = argparse.ArgumentParser(description="Financial Data Cleaner - Quick Start")
    parser.add_argument("file", help="Path to the financial data file to process")
    parser.add_argument("-o", "--output", 
                        help="Output directory (default: ./cleaned_output)",
                        default="./cleaned_output")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Verify file exists
    if not os.path.isfile(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)
    
    print(f"🔍 Processing file: {args.file}")
    
    # Process the file
    cleaned_data = preprocess_file(args.file, debug=True)
    
    if cleaned_data:
        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)
        
        # Create output path
        file_base = os.path.basename(args.file)
        name, _ = os.path.splitext(file_base)
        output_path = os.path.join(args.output, f"{name}_cleaned.xlsx")
        
        # Save the cleaned data
        save_cleaned_data(cleaned_data, output_path)
        print(f"\n✅ Success! Cleaned data saved to: {output_path}")
        print("\nNext steps:")
        print("  1. Examine the output file")
        print("  2. Try processing other financial data files")
        print("  3. Explore the API for more advanced usage")
    else:
        print("\n❌ Failed to process the file")
        print("Try using a different file or check if the file format is supported")

if __name__ == "__main__":
    main()
