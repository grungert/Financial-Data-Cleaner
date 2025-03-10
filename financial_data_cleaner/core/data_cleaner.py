"""
Core data cleaning functionality for financial data.
"""
import pandas as pd
import re
from ..mappers.column_mapper import fuzzy_column_mapping, display_mapped_data
from ..data.mapping_data import standard_field_types, currency_mapping

def clean_and_normalize(df, header_row):
    """
    Cleans and normalizes financial data, handling the case of data with no clear headers.
    """
    if header_row is None or header_row >= len(df):
        raise ValueError("❌ Invalid header row detected!")
    
    # For debugging purposes, let's first print exactly what the header row contains
    print("\n🔍 Header row content:")
    header_content = df.iloc[header_row].tolist()
    for i, item in enumerate(header_content):
        print(f"Column {i}: {item}")
    
    # Create headers from the detected header row
    headers = [
        str(df.iloc[header_row, i]).strip() if pd.notna(df.iloc[header_row, i]) else f"Column_{i}" 
        for i in range(df.shape[1])
    ]
    
    # Store original headers for later reference
    original_headers = headers.copy()
    
    # Create a new DataFrame with data rows after the header
    clean_df = df.iloc[header_row+1:].reset_index(drop=True).copy()
    clean_df.columns = headers
    
    # Remove fully empty rows
    clean_df = clean_df.dropna(how="all")
    
    # Check for and remove completely empty columns (all values are NaN or empty strings)
    empty_columns = []
    for col in clean_df.columns:
        # Check if column is completely empty (all NaN or empty strings)
        is_empty = True
        for val in clean_df[col]:
            if pd.notna(val) and str(val).strip() != "":
                is_empty = False
                break
        
        if is_empty:
            empty_columns.append(col)
    
    # Drop empty columns and log which ones were dropped
    if empty_columns:
        print(f"\n⚠️ Dropping {len(empty_columns)} empty columns:")
        for col in empty_columns:
            print(f"  - '{col}' (column is completely empty)")
        clean_df = clean_df.drop(columns=empty_columns)
    
    # Remove any remaining columns that are all NaN
    clean_df = clean_df.dropna(axis=1, how="all")
    
    # Map column names using fuzzy matching to standardize
    try:
        mapping_info = fuzzy_column_mapping(clean_df.columns)
        column_mapping, mapped_columns, original_to_mapped = mapping_info
        
        # Store the original headers in the DataFrame's metadata
        # Create a reverse mapping from standardized names to original names
        reverse_mapping = {}
        for source, target in column_mapping.items():
            if target not in reverse_mapping:
                reverse_mapping[target] = []
            reverse_mapping[target].append(source)
        
        # Get the best original header for each standardized name
        best_original_headers = {}
        for target, sources in reverse_mapping.items():
            if sources:
                # Get the source with the highest score
                best_source = max(sources, key=lambda s: original_to_mapped.get(s, (None, 0))[1])
                best_original_headers[target] = best_source
        
        # Store the mapping information in the DataFrame's metadata
        clean_df.attrs['original_headers'] = best_original_headers
        clean_df.attrs['original_to_mapped'] = original_to_mapped
        
        # Rename the columns to the standardized names
        clean_df = clean_df.rename(columns=column_mapping)
    except Exception as e:
        print(f"⚠️ Error during column mapping: {str(e)}")
        print("⚠️ Continuing with original column names")
        # Continue with original column names
        mapping_info = ({}, [], {})
        clean_df.attrs['original_headers'] = {col: col for col in clean_df.columns}
    
    # Convert columns based on their standard field type
    for col in clean_df.columns:
        if col in standard_field_types:
            field_type = standard_field_types[col]
            
            try:
                if field_type == "date":
                    # Convert to datetime
                    clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce', dayfirst=True)
                    print(f"✅ Converted '{col}' to datetime (based on standard field type)")
                    
                elif field_type == "numeric":
                    # Clean and convert to numeric
                    if clean_df[col].dtypes == object:
                        clean_df[col] = clean_df[col].apply(
                            lambda x: re.sub(r'[^\d.,\-]', '', str(x)).replace(',', '.') 
                            if pd.notna(x) and str(x) != 'nan' else x
                        )
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                    print(f"✅ Converted '{col}' to numeric (based on standard field type)")
                    
                elif field_type == "string" and col in ["Currency", "Compartment Currency"]:
                    # Standardize currency codes
                    if clean_df[col].dtypes == object:
                        clean_df[col] = clean_df[col].astype(str).str.strip().str.upper()
                        clean_df[col] = clean_df[col].map(lambda x: currency_mapping.get(x, x) 
                                                       if pd.notna(x) and x != 'NAN' else x)
                        print(f"✅ Standardized currency codes in '{col}' (based on standard field type)")
            except Exception as e:
                print(f"⚠️ Failed to convert '{col}' according to its standard type: {str(e)}")
    
    # Create a version of the DataFrame with only mapped columns
    mapped_df = display_mapped_data(clean_df, original_headers, mapping_info)
    
    print(f"✅ Data cleaned! {clean_df.isnull().sum().sum()} missing values remain.")
    return mapped_df  # Return only the mapped columns
