"""
Main processing functionality for the Financial Data Cleaner.
"""
import os
import time
from glob import glob
from tqdm import tqdm

from ..utils.file_handlers import read_file, save_cleaned_data
from .header_detection import detect_header_row, detect_financial_table_format
from .data_cleaner import clean_and_normalize

def preprocess_file(file_path, debug=True):
    """
    Reads a file (Excel or CSV), detects the header row, cleans the data, 
    and returns the cleaned DataFrame.
    """
    try:
        if debug:
            print(f"🔄 Processing file: {file_path}")
        
        # Read file
        sheets_dict = read_file(file_path)
        cleaned_sheets = {}

        for sheet_name, sheet_df in sheets_dict.items():
            if debug:
                print(f"📄 Processing sheet: {sheet_name}")
            try:
                # Special case for financial tables with clear headers
                if detect_financial_table_format(sheet_df, debug=debug):
                    best_header = 0
                else:
                    # Use regular detection for other cases
                    best_header = detect_header_row(sheet_df, debug=debug)
                
                # Clean and normalize data
                df_cleaned = clean_and_normalize(sheet_df, best_header)
                
                # Only keep sheet if it has meaningful data
                if not df_cleaned.empty and len(df_cleaned.columns) > 1:
                    cleaned_sheets[sheet_name] = df_cleaned
                else:
                    if debug:
                        print(f"⚠️ Skipping empty sheet '{sheet_name}'")
            except ValueError as e:
                if debug:
                    print(f"⚠️ Skipping sheet '{sheet_name}': {e}")
            except Exception as e:
                if debug:
                    print(f"❌ Error in sheet '{sheet_name}': {str(e)}")
                    import traceback
                    traceback.print_exc()

        if not cleaned_sheets:
            if debug:
                print("❌ No valid data found in any sheet")
            return None
            
        # Print a summary of the cleaned data
        if debug and cleaned_sheets:
            print("\n📊 Cleaned Data Summary:")
            for sheet_name, df in cleaned_sheets.items():
                print(f"  - Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                print(f"  - Columns: {', '.join(df.columns)}")
                print("\n")
            
        return cleaned_sheets

    except Exception as e:
        if debug:
            print(f"❌ Error processing file: {str(e)}")
        return None

def process_directory(directory_path, output_dir=None, debug=True):
    """
    Process all files with supported extensions in a directory.
    Returns a dictionary of {filename: cleaned_data}.
    """
    # Find all supported files
    supported_extensions = ['.xlsx', '.xls', '.csv']
    all_files = []
    
    for ext in supported_extensions:
        pattern = os.path.join(directory_path, f'*{ext}')
        all_files.extend(glob(pattern))
    
    if not all_files:
        if debug:
            print(f"❌ No supported files found in {directory_path}")
        return {}
    
    if debug:
        print(f"🔍 Found {len(all_files)} files to process")
    
    # Process each file
    results = {}
    success_count = 0
    failure_count = 0
    
    start_time = time.time()
    
    for file_path in tqdm(all_files, desc="Processing files", disable=not debug):
        file_name = os.path.basename(file_path)
        if debug:
            print(f"\n📂 Processing {file_name}...")
        
        cleaned_data = preprocess_file(file_path, debug=debug)
        
        if cleaned_data:
            if output_dir:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Construct output path
                file_base, _ = os.path.splitext(file_name)
                output_path = os.path.join(output_dir, f"{file_base}_cleaned.xlsx")
                
                # Save to output directory
                save_cleaned_data(cleaned_data, output_path)
            
            results[file_name] = cleaned_data
            success_count += 1
        else:
            failure_count += 1
    
    elapsed_time = time.time() - start_time
    
    if debug:
        print("\n📊 Directory Processing Summary:")
        print(f"  - Total files: {len(all_files)}")
        print(f"  - Successfully processed: {success_count}")
        print(f"  - Failed to process: {failure_count}")
        print(f"  - Total processing time: {elapsed_time:.2f} seconds")
    
    return results
