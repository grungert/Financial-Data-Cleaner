"""
Functions for detecting header rows in financial data tables.
"""
import pandas as pd
from ..utils.validators import (
    is_date, is_numeric, has_special_characters, 
    calculate_capitalization_score, count_words
)

def analyze_column_consistency(df, header_row_idx):
    """
    Analyze how consistent column data types are below a potential header row.
    Returns a score where higher values indicate more consistency (more likely to be data rows).
    """
    if header_row_idx >= len(df) - 1:  # Need at least one data row
        return 0
    
    data_rows = df.iloc[(header_row_idx + 1):]
    consistency_score = 0
    
    for col_idx in range(len(df.columns)):
        col_data = data_rows.iloc[:, col_idx].dropna()
        
        if len(col_data) < 2:  # Need at least 2 values for consistency check
            continue
        
        # Check type consistency
        date_count = sum(1 for val in col_data if is_date(val))
        numeric_count = sum(1 for val in col_data if is_numeric(val))
        
        # Calculate type consistency percentage
        if len(col_data) > 0:
            max_type_consistency = max(
                date_count / len(col_data),
                numeric_count / len(col_data),
                (len(col_data) - date_count - numeric_count) / len(col_data)  # Text consistency
            )
            consistency_score += max_type_consistency
    
    # Return average consistency across columns
    return consistency_score / len(df.columns) if df.columns.size > 0 else 0

def calculate_row_similarity(df, row_index):
    """
    Calculate similarity between a row and all other rows based on data types.
    Returns a score where higher means the row is more different from others (likely a header).
    """
    if len(df) <= 1:
        return 0  # Can't compare with other rows
    
    row = df.iloc[row_index]
    
    # Calculate capitalization and word count scores for this row
    cap_score = sum(calculate_capitalization_score(val) for val in row if pd.notna(val)) / max(1, row.notna().sum())
    word_count = sum(count_words(val) for val in row if pd.notna(val)) / max(1, row.notna().sum())
    
    # Special character counts - headers typically have fewer
    special_char_count = sum(1 for val in row if has_special_characters(val))
    special_char_score = 1.0 - (special_char_count / max(1, row.notna().sum()))
    
    # Check for dates and numbers - headers typically have fewer
    date_count = sum(1 for val in row if is_date(val))
    numeric_count = sum(1 for val in row if is_numeric(val))
    
    # Calculate percentage of non-date, non-numeric values (more text = more likely a header)
    text_percentage = 1.0 - ((date_count + numeric_count) / max(1, row.notna().sum()))
    
    # Analyze column consistency below this potential header
    consistency_score = analyze_column_consistency(df, row_index)
    
    # Calculate composite score
    header_likelihood = (
        cap_score * 0.3 +          # 30% weight for proper capitalization
        word_count * 0.15 +        # 15% weight for word count
        special_char_score * 0.15 + # 15% weight for lack of special characters
        text_percentage * 0.2 +    # 20% weight for text vs. dates/numbers
        consistency_score * 0.2     # 20% weight for consistency of columns below
    )
    
    return header_likelihood

def check_for_uniform_data(df):
    """
    Check if the data appears to have headers in the first row.
    Returns True if the first row is likely headers, False otherwise.
    """
    if len(df) <= 1:
        return True
    
    # Check first few rows (up to 5) for patterns
    rows_to_check = min(5, len(df))
    
    # Count typical data patterns in each row
    date_counts = []
    numeric_counts = []
    
    for i in range(rows_to_check):
        row = df.iloc[i]
        date_counts.append(sum(1 for val in row if is_date(val)))
        numeric_counts.append(sum(1 for val in row if is_numeric(val)))
    
    # If first row has significantly fewer dates/numbers than others, it's likely a header
    if len(date_counts) > 1 and len(numeric_counts) > 1:
        avg_dates_after_first = sum(date_counts[1:]) / (len(date_counts) - 1) if len(date_counts) > 1 else 0
        avg_numbers_after_first = sum(numeric_counts[1:]) / (len(numeric_counts) - 1) if len(numeric_counts) > 1 else 0
        
        if (date_counts[0] < avg_dates_after_first * 0.5 and 
            numeric_counts[0] < avg_numbers_after_first * 0.5 and
            avg_dates_after_first > 0 and avg_numbers_after_first > 0):
            return True
    
    return False

def detect_header_row(df, max_rows_to_check=10, debug=True):
    """
    Detect the best header row using multiple heuristics.
    Returns the index of the detected header row.
    Limited to checking the first max_rows_to_check rows for efficiency.
    """
    
    # Special case: If only 2 rows, first row is very likely the header
    if len(df) == 2:
        if debug:
            print("✅ Only two rows found, using first row as header.")
        return 0
    
    # Common financial column keywords to look for
    financial_keywords = [
        'isin', 'fund', 'nav', 'date', 'value', 'shares', 'currency', 'assets',
        'devise', 'code', 'valeur', 'parts', 'encours', 'libellé', 'coupon',
        'name', 'outstanding', 'total', 'wpk', 'profit', 'difference', 'trade',
        'previous', 'interim', 'code', 'price', 'percent', '%', 'net', 'day'
    ]
    
    best_score = 0
    best_row = 0  # Default to first row if nothing better is found
    
    # Check up to max_rows_to_check rows or all rows if less
    rows_to_check = min(max_rows_to_check, len(df))
    
    for i in range(rows_to_check):
        row = df.iloc[i]
        
        # Count non-empty cells
        non_empty_count = row.notna().sum()
        if non_empty_count <= len(df.columns) * 0.3:  # Skip mostly empty rows
            continue
        
        # Check for financial keywords in the row
        keyword_score = 0
        for cell in row:
            if pd.notna(cell):
                cell_str = str(cell).lower()
                for keyword in financial_keywords:
                    if keyword in cell_str:
                        keyword_score += 1
        
        # Check for header characteristics
        header_likelihood = calculate_row_similarity(df, i)
        
        # Calculate total score
        normalized_keywords = min(1.0, keyword_score / (len(financial_keywords) * 0.5))  # Allow exceeding keywords to count more
        normalized_non_empty = non_empty_count / len(df.columns)
        
        total_score = (
            normalized_non_empty * 0.2 +       # 20% weight for non-empty cells
            normalized_keywords * 0.3 +        # 30% weight for keyword matches
            header_likelihood * 0.3          # 30% weight for header characteristics
        )
        
        if debug:
            print(f"Row {i}: empty={normalized_non_empty:.2f}, keywords={normalized_keywords:.2f}, " 
                  f"header_like={header_likelihood:.2f}, total={total_score:.2f}")
        
        if total_score > best_score:
            best_score = total_score
            best_row = i
    
    if debug and best_score != 0.00:
        print(f"✅ Best header row detected at index {best_row} with score {best_score:.2f}")
    
    return best_row

def detect_financial_table_format(df, debug=True):
    """
    Detect if the data appears to be a financial table with specific characteristics.
    Returns True if the data looks like a financial table, False otherwise.
    """
    # Check if we have NAVDATE, TRADEDATE pattern in the first row
    if len(df) < 2:  # Need at least 2 rows
        return False
    
    first_row = df.iloc[0]
    
    # Check for common financial data patterns in the first row
    date_fields_count = 0
    financial_keyword_count = 0
    
    financial_keywords = ['nav', 'date', 'fund', 'isin', 'currency', 'shares', 'total', 'difference']
    
    for val in first_row:
        if pd.isna(val):
            continue
        
        val_str = str(val).lower()
        
        # Check for date-related fields
        if any(date_term in val_str for date_term in ['date', 'day']):
            date_fields_count += 1
        
        # Check for financial keywords
        for keyword in financial_keywords:
            if keyword in val_str:
                financial_keyword_count += 1
                break
    
    # If we have a decent number of financial keywords, it's likely a header row
    financial_table_likelihood = (
        date_fields_count >= 1 and 
        financial_keyword_count >= 2
    )
    
    if debug and financial_table_likelihood:
        print("✅ Data appears to be a financial table with clear headers in the first row.")
    
    return financial_table_likelihood
