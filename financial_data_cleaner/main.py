"""
Main entry point for the Financial Data Cleaner application.
"""
import os
import argparse
import sys

from .core.processor import preprocess_file, process_directory
from .utils.formatters import generate_summary_report
from .utils.dependencies import check_dependencies
from .utils.file_handlers import save_cleaned_data

def main():
    """
    Main entry point for the Financial Data Cleaner application.
    """
    parser = argparse.ArgumentParser(description="Financial Data Cleaner - Standardize financial data files")
    
    # Define command line arguments
    parser.add_argument("path", help="Path to a file or directory to process")
    parser.add_argument("-o", "--output", help="Output directory for cleaned files")
    parser.add_argument("-r", "--report", help="Generate a summary report and save to the specified path")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Configure debug mode based on quiet flag
    debug = not args.quiet
    
    # Process file or directory
    if os.path.isfile(args.path):
        # Process a single file
        if debug:
            print(f"🔍 Processing single file: {args.path}")
        
        cleaned_data = preprocess_file(args.path, debug=debug)
        
        if cleaned_data:
            if args.output:
                # Create output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Save to output directory
                output_base = os.path.basename(args.path)
                name, _ = os.path.splitext(output_base)
                output_path = os.path.join(args.output, f"{name}_cleaned.xlsx")
                
                save_cleaned_data(cleaned_data, output_path)
                if debug:
                    print(f"✅ Saved cleaned data to {output_path}")
            
            # Generate report if requested
            if args.report:
                results = {os.path.basename(args.path): cleaned_data}
                generate_summary_report(results, args.report)
                if debug:
                    print(f"✅ Saved report to {args.report}")
        else:
            if debug:
                print("❌ Failed to process file")
            
    elif os.path.isdir(args.path):
        # Process directory
        if debug:
            print(f"🔍 Processing directory: {args.path}")
        
        results = process_directory(args.path, args.output, debug=debug)
        
        # Generate report if requested
        if args.report and results:
            generate_summary_report(results, args.report)
            if debug:
                print(f"✅ Saved report to {args.report}")
    else:
        print(f"❌ Path not found: {args.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
