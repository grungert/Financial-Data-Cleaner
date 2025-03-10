"""
Formatting utilities for output data and reports.
"""
import pandas as pd
import os
import time
from datetime import datetime

def format_output_dataframe(df):
    """
    Format the DataFrame for display, showing only mapped columns.
    First row: Mapped headers (standardized names)
    Second row: Original headers (source column names)
    Rest: The actual data values
    """
    try:
        # Get the original headers from the DataFrame's metadata
        original_headers = df.attrs.get('original_headers', {})
        
        if not original_headers:
            return df  # No mapping info, return as is
        
        # Create a new DataFrame with the original headers as the first row
        result = df.copy()
        
        # Convert the first row to the original headers
        header_row = pd.DataFrame([original_headers.get(col, col) for col in df.columns]).T
        header_row.columns = df.columns
        
        # Concatenate the header row with the data
        result = pd.concat([header_row, df], ignore_index=True)
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error formatting output DataFrame: {str(e)}")
        return df

def format_table_for_report(df, original_headers, mapping_info, max_rows=5):
    """
    Format a DataFrame as a table for the report.
    
    Args:
        df: The cleaned DataFrame
        original_headers: The original headers from the input file
        mapping_info: Tuple containing (column_mapping, mapped_columns, original_to_mapped)
        max_rows: Maximum number of data rows to include in the table
    
    Returns:
        String representation of the table
    """
    column_mapping, mapped_columns, original_to_mapped = mapping_info
    
    # Get the unique target columns (standardized names)
    unique_targets = list(set(column_mapping.values()))
    
    if not unique_targets:
        return "No mappable columns found in this dataset."
    
    # Create a reverse mapping from standardized names to original names
    reverse_mapping = {}
    for source, target in column_mapping.items():
        if target not in reverse_mapping:
            reverse_mapping[target] = []
        reverse_mapping[target].append(source)
    
    # Format the table header
    header = "| Standard Field | Original Field | Sample Values |\n"
    header += "|---------------|----------------|---------------|\n"
    
    rows = []
    
    for target in unique_targets:
        # Find all source columns that map to this target
        sources = reverse_mapping.get(target, [])
        if not sources:
            continue
            
        # Get the source with the highest score
        best_source = max(sources, key=lambda s: original_to_mapped.get(s, (None, 0))[1])
        
        # Get sample values for this column
        if target in df.columns:
            sample_values = df[target].dropna().head(3).astype(str).tolist()
            if not sample_values:
                sample_values = ["<no values>"]
            samples = ", ".join(sample_values)
            
            # Truncate long sample values
            if len(samples) > 40:
                samples = samples[:37] + "..."
                
            rows.append(f"| {target} | {best_source} | {samples} |")
    
    if not rows:
        return "No sample values available for this dataset."
    
    return header + "\n".join(rows)

def generate_summary_report(results, output_path=None):
    """
    Generate a summary report of all processed files.
    """
    if not results:
        print("❌ No data to generate a report for.")
        return None
    
    # Start building the report
    report = []
    
    # Add the title
    report.append("# Financial Data Cleaner - Summary Report")
    report.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report.append("")
    
    # Add the summary section
    report.append("## Summary")
    report.append(f"- Total files processed: {len(results)}")
    
    # Count total successful sheets
    total_sheets = sum(len(sheets) for sheets in results.values())
    report.append(f"- Total sheets processed: {total_sheets}")
    
    # Section for detailed statistics, collapsed by default
    report.append("")
    report.append("## Detailed Statistics")
    
    # Process each file
    for file_name, cleaned_data in sorted(results.items()):
        if not cleaned_data:
            continue
            
        # Add a section for this file
        report.append(f"### {file_name}")
        
        # Process each sheet in the file
        for sheet_name, df in cleaned_data.items():
            report.append(f"#### Sheet: {sheet_name}")
            
            # Print dimensions and stats
            report.append(f"- Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Get missing values count and percentage
            missing_count = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_percentage = (missing_count / total_cells) * 100 if total_cells > 0 else 0
            
            report.append(f"- Missing values: {missing_count} ({missing_percentage:.1f}%)")
            
            # Get original headers from DataFrame metadata
            if hasattr(df, 'attrs') and 'original_headers' in df.attrs and 'original_to_mapped' in df.attrs:
                original_headers = list(df.attrs['original_headers'].values())
                original_to_mapped = df.attrs['original_to_mapped']
                
                # Recreate the mapping info
                column_mapping = {src: tgt for src, (tgt, _) in original_to_mapped.items()}
                mapped_columns = list(set(column_mapping.values()))
                mapping_info = (column_mapping, mapped_columns, original_to_mapped)
                
                # Add a sample table
                report.append("\n**Sample Data:**\n")
                report.append(format_table_for_report(df, original_headers, mapping_info))
            else:
                # If no mapping info, just show column names
                report.append("\n**Columns:**\n")
                report.append("- " + "\n- ".join(df.columns.tolist()))
            
            report.append("\n")
    
    # Combine all sections
    report_text = "\n".join(report)
    
    # Save to file if output path provided
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Summary report saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving report: {str(e)}")
    
    return report_text
