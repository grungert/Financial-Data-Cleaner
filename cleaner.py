import pandas as pd
import os
import re
import sys
import chardet
from glob import glob
from datetime import datetime
import time
from tqdm import tqdm

# Try importing optional dependencies
try:
    from fuzzywuzzy import process, fuzz
except ImportError:
    print("⚠️ Missing fuzzywuzzy package. Install with: pip install fuzzywuzzy python-Levenshtein")
    print("⚠️ Continuing with limited fuzzy matching capability...")
    # Define a simple fallback implementation
    class fuzz:
        @staticmethod
        def token_sort_ratio(s1, s2):
            s1 = str(s1).lower()
            s2 = str(s2).lower()
            return 100 if s1 == s2 else (70 if s1 in s2 or s2 in s1 else 0)

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

def check_dependencies():
    """
    Check and report on required dependencies.
    """
    missing = []
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    # Only report if there are missing core dependencies
    if missing:
        print("❌ Missing required dependencies: " + ", ".join(missing))
        print("📦 Install them with: pip install " + " ".join(missing))
        return False
    
    # Check optional dependencies and provide helpful info
    warnings = []
    
    try:
        import xlrd
    except ImportError:
        warnings.append("xlrd (for .xls files)")
    
    try:
        import openpyxl
    except ImportError:
        warnings.append("openpyxl (for .xlsx files)")
    
    try:
        import fuzzywuzzy
    except ImportError:
        warnings.append("fuzzywuzzy python-Levenshtein (for better column matching)")
    
    try:
        import tqdm
    except ImportError:
        warnings.append("tqdm (for progress bars)")
    
    if warnings:
        print("⚠️ Optional dependencies missing: " + ", ".join(warnings))
        print("📦 For full functionality: pip install " + " ".join(warnings))
    
    return True

def is_date(val):
    """
    Check if a value looks like a date.
    """
    if pd.isna(val):
        return False
    
    val_str = str(val).strip()
    
    # Check for common date patterns
    date_patterns = [
        r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
        r'\d{4}[/.-]\d{1,2}[/.-]\d{1,2}',  # YYYY/MM/DD
        r'\d{1,2}[/.-][A-Za-z]{3}[/.-]\d{2,4}',  # DD-MMM-YYYY
        r'[A-Za-z]{3}[\s,]\d{1,2}[\s,]\d{4}',  # MMM DD, YYYY
        r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?'  # Date with time
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, val_str):
            return True
    
    # Try parsing as datetime
    try:
        datetime.strptime(val_str, '%d/%m/%Y')
        return True
    except:
        try:
            datetime.strptime(val_str, '%m/%d/%Y')
            return True
        except:
            try:
                pd.to_datetime(val_str)
                return True
            except:
                return False

def is_numeric(val):
    """
    Check if a value is numeric or could be converted to numeric.
    """
    if pd.isna(val):
        return False
    
    val_str = str(val).strip()
    
    # Remove currency symbols and thousands separators
    val_str = re.sub(r'[^\d.,\-]', '', val_str)
    val_str = val_str.replace(',', '.')
    
    try:
        float(val_str)
        return True
    except:
        return False

def has_special_characters(val):
    """
    Check if a value contains special characters often found in data but not headers.
    """
    if pd.isna(val):
        return False
    
    val_str = str(val).strip()
    special_chars = ['%', '$', '€', '£', '¥', '+', '-', '*', '/', '=', '<', '>']
    
    return any(char in val_str for char in special_chars)

def calculate_capitalization_score(val):
    """
    Return a score indicating how likely this value is to be a header based on capitalization.
    Headers often have specific capitalization patterns (ALL CAPS, Title Case, camelCase).
    """
    if pd.isna(val):
        return 0
    
    val_str = str(val).strip()
    if not val_str:
        return 0
    
    # Check if ALL CAPS
    if val_str.isupper():
        return 1.0
    
    # Check if Title Case
    if val_str.istitle():
        return 0.8
    
    # Check for camelCase or PascalCase (mixed case with no spaces)
    if re.match(r'^[A-Za-z0-9]+$', val_str) and any(c.isupper() for c in val_str) and any(c.islower() for c in val_str):
        return 0.7
    
    # Check if it has underscores (common in headers)
    if '_' in val_str:
        return 0.6
    
    return 0.2  # Default low score for other formats

def count_words(val):
    """
    Count words in a value - headers often have more words than data cells.
    """
    if pd.isna(val):
        return 0
    
    val_str = str(val).strip()
    # Split by common word separators
    words = re.split(r'[\s_\-/]', val_str)
    return len([w for w in words if w])  # Count non-empty words

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
            normalized_non_empty * 0.2 +       # Chapter 20% weight for non-empty cells
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

# Define standard field types
standard_field_types = {
    "ISIN": "string",
    "Fund Name": "string",
    "Share Name": "string",
    "NAV Date": "date",
    "NAV Value": "numeric",
    "Number of Shares": "numeric",
    "Currency": "string",
    "Compartment Assets": "numeric",
    "Compartment Currency": "string",
    "Coupon": "numeric",
    "Coupon Date": "date",
    "CIC Code": "string",
    "Previous NAV": "numeric",
    "Difference": "numeric",
    "Difference Percent": "numeric",
    "WPK Code": "string"
}

# Global cache for column mappings
_column_mapping_cache = {}

def fuzzy_column_mapping(columns, unmapped_action="keep", unmapped_prefix="Unknown_", use_cache=True):
    """
    Maps columns to standard financial field names using fuzzy matching.
    Only keeps the best match for each target field.
    
    Returns:
        tuple: (column_mapping, mapped_columns, original_to_mapped)
            - column_mapping: Dictionary mapping original column names to standardized names
            - mapped_columns: List of standardized column names that were successfully mapped
            - original_to_mapped: Dictionary mapping original column names to (standardized name, score)
    """
    # Generate a cache key based on the columns
    if use_cache:
        cache_key = tuple(sorted([str(c).lower() for c in columns if pd.notna(c) and c != ""]))
        if cache_key in _column_mapping_cache:
            print("✅ Using cached column mapping")
            return _column_mapping_cache[cache_key]
    
    standard_fields = {
        "ISIN": ["code isin", "isin", "isin code", "code", "code_isin"],
        "Fund Name": ["fcp", "fund", "fund name", "nom du fonds", "libellé", "libelle", "fund_name", "name", "nom", "fund name"],
        "Share Name": ["share", "share name", "share_name", "part", "nom de la part"],
        "NAV Date": ["nav date", "date vl", "date", "date de publication", "nav_date", "date valeur", "valuation date", "navdate", "tradedate"],
        "NAV Value": ["valeur liquidative", "vl", "nav value", "nav", "value", "prix", "price", "valeur", "nav_today"],
        "Number of Shares": ["nb parts", "nombre de parts", "shares", "parts", "number of shares", "nb_parts", "shares_outstanding"],
        "Currency": ["devise", "currency", "ccy", "monnaie"],
        "Compartment Assets": ["encours net", "actif net", "compartment assets", "assets", "encours", "encours_global", "aum", "total_nav"],
        "Compartment Currency": ["compartment currency", "devise du compartiment"],
        "Coupon": ["coupon", "coupon rate"],
        "Coupon Date": ["coupon date", "date de coupon", "payment date"],
        "CIC Code": ["cic code", "cic"],
        "Previous NAV": ["nav previous", "previous nav", "nav previous day", "previous value"],
        "Difference": ["difference", "diff", "change"],
        "Difference Percent": ["difference %", "diff %", "change %", "percent change", "difference percent"],
        "WPK Code": ["wpk", "code_wpk", "wpk code"]
    }
    
    # Check for duplicate column names
    seen_columns = set()
    duplicate_columns = []

    for col in columns:
        if pd.notna(col) and col != "":
            col_str = str(col).strip()
            if col_str in seen_columns:
                duplicate_columns.append(col_str)
            seen_columns.add(col_str)

    if duplicate_columns:
        print(f"\n⚠️ Found {len(duplicate_columns)} duplicate column names in input:")
        for col in duplicate_columns[:5]:
            print(f"  - '{col}' appears multiple times")
        if len(duplicate_columns) > 5:
            print(f"  - ... and {len(duplicate_columns) - 5} more")
    
    # Track all potential matches for each source column
    source_matches = {}  # {source_column: [(target_field, score), ...]}
    
    # Track skipped and ambiguous columns
    skipped_columns = []
    ambiguous_mappings = []
    
    original_column_count = len([c for c in columns if pd.notna(c) and c != ""])
    
    # First pass: Find all potential matches for each source column
    for col in columns:
        if pd.isna(col) or col == "":
            continue
            
        col_str = str(col).strip()
        matches = []
        
        # First try exact matching (case-insensitive)
        col_lower = col_str.lower()
        exact_match_found = False
        
        for standard_field, aliases in standard_fields.items():
            if col_lower in aliases or col_lower == standard_field.lower():
                matches.append((standard_field, 100))
                exact_match_found = True
                break
        
        # If no exact match, try fuzzy matching
        if not exact_match_found:
            # Check if we have fuzz from fuzzywuzzy or our simple fallback
            if hasattr(fuzz, 'token_sort_ratio'):
                for standard_field, aliases in standard_fields.items():
                    # Try matching against the standard field name
                    score = fuzz.token_sort_ratio(col_lower, standard_field.lower())
                    
                    # Try matching against each alias
                    for alias in aliases:
                        alias_score = fuzz.token_sort_ratio(col_lower, alias)
                        score = max(score, alias_score)
                    
                    if score > 70:  # 70% threshold
                        matches.append((standard_field, score))
            else:
                # Simple string containment check as fallback
                for standard_field, aliases in standard_fields.items():
                    if standard_field.lower() in col_lower or col_lower in standard_field.lower():
                        matches.append((standard_field, 90))
                        break
                    
                    for alias in aliases:
                        if alias in col_lower or col_lower in alias:
                            matches.append((standard_field, 85))
                            break
        
        # Sort matches by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Track ambiguous mappings
        if len(matches) >= 2 and (matches[0][1] - matches[1][1] < 10) and matches[1][1] > 70:
            ambiguous_mappings.append((col_str, matches[0][0], matches[0][1], matches[1][0], matches[1][1]))
        
        # Track skipped columns
        if not matches:
            skipped_columns.append(col_str)
        
        # Store all matches for this source column
        source_matches[col_str] = matches
    
    # Second pass: Resolve conflicts by selecting the best match for each target field
    target_to_source = {}  # {target_field: (source_column, score)}
    
    # Sort source columns by their best match score (descending)
    sorted_sources = sorted(
        [(source, matches[0][1]) for source, matches in source_matches.items() if matches],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Assign each source to its best target, resolving conflicts in favor of higher scores
    for source, _ in sorted_sources:
        matches = source_matches[source]
        if not matches:
            continue
            
        best_target, best_score = matches[0]
        
        # Check if this target already has a better source
        if best_target in target_to_source:
            existing_source, existing_score = target_to_source[best_target]
            if existing_score >= best_score:
                # Skip this source for this target as we already have a better match
                continue
        
        # Assign this source to this target
        target_to_source[best_target] = (source, best_score)
    
    # Create the final mapping
    column_mapping = {}
    mapped_columns = []
    original_to_mapped = {}
    
    for target, (source, score) in target_to_source.items():
        column_mapping[source] = target
        mapped_columns.append(target)
        original_to_mapped[source] = (target, score)
        print(f"✅ Mapped '{source}' to '{target}' (score: {score})")
    
    # After mapping, handle unmapped columns
    if unmapped_action == "drop":
        # Return only mapped columns
        pass  # column_mapping already contains only mapped columns
    elif unmapped_action == "prefix":
        # Add unmapped columns with prefix
        for col in columns:
            if pd.isna(col) or col == "":
                continue
            col_str = str(col).strip()
            if col_str and col_str not in column_mapping:
                column_mapping[col_str] = f"{unmapped_prefix}{col_str}"
                mapped_columns.append(f"{unmapped_prefix}{col_str}")
                original_to_mapped[col_str] = (f"{unmapped_prefix}{col_str}", 0)
    
    # Print diagnostics
    if skipped_columns:
        print(f"\n⚠️ Skipped {len(skipped_columns)} columns that couldn't be mapped:")
        for col in skipped_columns[:5]:  # Show first 5
            print(f"  - {col}")
        if len(skipped_columns) > 5:
            print(f"  - ... and {len(skipped_columns) - 5} more")

    if ambiguous_mappings:
        print(f"\n⚠️ Found {len(ambiguous_mappings)} potentially ambiguous mappings:")
        for col, target1, score1, target2, score2 in ambiguous_mappings[:3]:  # Show first 3
            print(f"  - '{col}' mapped to '{target1}' (score: {score1}) but also matches '{target2}' (score: {score2})")
        if len(ambiguous_mappings) > 3:
            print(f"  - ... and {len(ambiguous_mappings) - 3} more")
    
    mapped_column_count = len(set(column_mapping.values()))
    print(f"\n📊 Column mapping summary:")
    print(f"  - Original columns: {original_column_count}")
    print(f"  - Mapped columns: {len(column_mapping)}")
    print(f"  - Unique target columns: {mapped_column_count}")
    
    # Create a tuple with all the mapping information
    mapping_info = (column_mapping, mapped_columns, original_to_mapped)
    
    # Cache the result before returning
    if use_cache:
        _column_mapping_cache[cache_key] = mapping_info
    
    return mapping_info


def display_mapped_data(df, original_headers, mapping_info):
    """
    Display only the mapped columns in the format requested by the user.
    
    Args:
        df: The cleaned DataFrame
        original_headers: The original headers from the input file
        mapping_info: Tuple containing (column_mapping, mapped_columns, original_to_mapped)
    
    Returns:
        DataFrame containing only the mapped columns
    """
    column_mapping, mapped_columns, original_to_mapped = mapping_info
    
    # Get the unique target columns (standardized names)
    unique_targets = list(set(column_mapping.values()))
    
    # Create a new DataFrame with only the mapped columns
    if len(unique_targets) > 0:
        # Filter the DataFrame to include only mapped columns
        mapped_df = df[unique_targets].copy()
        
        # Create a mapping from original headers to standardized names
        original_to_standard = {}
        for i, header in enumerate(original_headers):
            if pd.notna(header) and header in column_mapping:
                original_to_standard[header] = column_mapping[header]
        
        # Print the mapping information in the requested format
        print("\n📊 Mapped Columns:")
        for target in unique_targets:
            # Find all source columns that map to this target
            sources = [src for src, tgt in column_mapping.items() if tgt == target]
            if sources:
                # Get the source with the highest score
                best_source = max(sources, key=lambda s: original_to_mapped.get(s, (None, 0))[1])
                score = original_to_mapped.get(best_source, (None, 0))[1]
                print(f"✅ Mapped '{best_source}' to '{target}' (score: {score})")
        
        # Print the data type for each mapped column
        print("\n📋 Column Data Types:")
        for col in unique_targets:
            data_type = standard_field_types.get(col, "string")
            print(f"  - '{col}': {data_type}")
        
        # Print summary
        print(f"\n📈 Column Summary:")
        print(f"  - Original columns: {len(original_headers)}")
        print(f"  - Mapped columns: {len(column_mapping)}")
        print(f"  - Unique target columns: {len(unique_targets)}")
        
        return mapped_df
    else:
        print("❌ No columns were successfully mapped.")
        return df

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
                    currency_mapping = {
                        '€': 'EUR', '$': 'USD', '£': 'GBP', 'EURO': 'EUR', 'EUROS': 'EUR',
                        'US DOLLAR': 'USD', 'DOLLAR': 'USD', 'DOLLARS': 'USD', 'POUND': 'GBP',
                        'YEN': 'JPY', 'JAPANESE YEN': 'JPY', 'SWISS FRANC': 'CHF', 'CHF': 'CHF',
                        'CANADIAN DOLLAR': 'CAD', 'CAD': 'CAD', 'AUSTRALIAN DOLLAR': 'AUD', 'AUD': 'AUD'
                    }
                    
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


def read_file(file_path):
    """
    Reads a file (CSV, XLS, XLSX) and returns a pandas DataFrame.
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.xls':
        try:
            # Try to read .xls file
            return pd.read_excel(file_path, sheet_name=None, engine='xlrd', header=None)
        except ImportError:
            print("❌ Missing xlrd package. Install it with: pip install xlrd>=2.0.1")
            print("⚠️ Note: For .xlsx files, you need the openpyxl package.")
            raise ImportError("Missing required dependency 'xlrd' for .xls files. Run: pip install xlrd>=2.0.1")
        except Exception as e:
            raise ValueError(f"❌ Error reading XLS file: {str(e)}")
            
    elif ext == '.xlsx':
        try:
            # Try to read .xlsx file
            return pd.read_excel(file_path, sheet_name=None, engine='openpyxl', header=None)
        except ImportError:
            print("❌ Missing openpyxl package. Install it with: pip install openpyxl")
            raise ImportError("Missing required dependency 'openpyxl' for .xlsx files. Run: pip install openpyxl")
        except Exception as e:
            raise ValueError(f"❌ Error reading XLSX file: {str(e)}")
    
    elif ext == '.csv':
        try:
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                encoding = result['encoding']
            
            # Try different delimiters
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=None)
                    return {'Sheet1': df}  # Wrap in dict to match Excel format
                except:
                    continue
            
            # If all attempts failed, try with default settings
            return {'Sheet1': pd.read_csv(file_path, header=None)}
        except ImportError as e:
            if 'chardet' in str(e):
                print("❌ Missing chardet package. Install it with: pip install chardet")
                raise ImportError("Missing optional dependency 'chardet' for better CSV encoding detection. Run: pip install chardet")
            raise
        except Exception as e:
            raise ValueError(f"❌ Error reading CSV file: {str(e)}")
    
    else:
        raise ValueError(f"❌ Unsupported file format: {ext}")

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
        print(f"  - Failed: {failure_count}")
        print(f"  - Time elapsed: {elapsed_time:.2f} seconds")
    
    return results

def format_output_dataframe(df):
    """
    Format the DataFrame for display, showing only mapped columns.
    First row: Mapped headers (standardized names)
    Second row: Original headers (source column names)
    Rest: The actual data values
    """
    if df.empty:
        return df
    
    # Create a new DataFrame for the formatted output
    formatted_df = pd.DataFrame()
    
    # For each column in the DataFrame, add it to the formatted output
    for col in df.columns:
        formatted_df[col] = df[col]
    
    return formatted_df

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
    if df.empty:
        return "No data available."
    
    column_mapping, mapped_columns, original_to_mapped = mapping_info
    
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
    
    # Create the table header rows
    header_row = " | ".join(df.columns)
    original_header_row = " | ".join([best_original_headers.get(col, col) for col in df.columns])
    
    # Create the table separator
    separator = "-|-".join(["-" * len(col) for col in df.columns])
    
    # Create the data rows
    data_rows = []
    for i, row in df.head(max_rows).iterrows():
        data_row = " | ".join([str(val) if pd.notna(val) else "" for val in row])
        data_rows.append(data_row)
    
    # Add ellipsis if there are more rows
    if len(df) > max_rows:
        data_rows.append("... (and " + str(len(df) - max_rows) + " more rows)")
    
    # Combine all parts into a table
    table = [
        "| " + header_row + " |",
        "| " + original_header_row + " |",
        "| " + separator + " |"
    ] + ["| " + row + " |" for row in data_rows]
    
    return "\n".join(table)

def save_cleaned_data(cleaned_data, output_path=None):
    """
    Saves the cleaned data to Excel or returns the first sheet if no path provided.
    """
    if not cleaned_data:
        return None
        
    if output_path:
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in cleaned_data.items():
                # Format the DataFrame for output
                formatted_df = format_output_dataframe(df)
                formatted_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"✅ Cleaned data saved to {output_path}")
        return output_path
    else:
        # Return the first sheet's DataFrame if no output path
        first_sheet_df = next(iter(cleaned_data.values()))
        return format_output_dataframe(first_sheet_df)

def generate_summary_report(results, output_path=None):
    """
    Generate a summary report of all processed files.
    """
    print(f"\n🔄 Generating summary report...")
    
    if not results:
        print("❌ No files processed successfully.")
        return "No files processed successfully."
    
    print(f"📊 Found {len(results)} files to include in the report.")
    
    report = ["# Financial Data Normalization Summary Report", "", f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    
    # Add overall statistics
    report.append("## Overall Statistics")
    report.append(f"- Total files processed: {len(results)}")
    
    total_rows = 0
    total_sheets = 0
    file_details = []
    
    for file_name, sheets in results.items():
        file_row_count = sum(len(df) for df in sheets.values())
        total_rows += file_row_count
        total_sheets += len(sheets)
        
        file_details.append({
            'file_name': file_name,
            'sheets': len(sheets),
            'rows': file_row_count,
            'columns': sum(len(df.columns) for df in sheets.values()) / len(sheets),
            'data': sheets  # Store the actual data for later use
        })
    
    report.append(f"- Total data rows: {total_rows}")
    report.append(f"- Total sheets: {total_sheets}")
    report.append("")
    
    # Add file details
    report.append("## File Details")
    
    for detail in sorted(file_details, key=lambda x: x['file_name']):
        print(f"  - Adding details for {detail['file_name']}...")
        
        report.append(f"### {detail['file_name']}")
        report.append(f"- Sheets: {detail['sheets']}")
        report.append(f"- Data rows: {detail['rows']}")
        report.append(f"- Average columns per sheet: {detail['columns']:.1f}")
        
        # Add table output for each sheet
        for sheet_name, df in detail['data'].items():
            print(f"    - Adding table for sheet '{sheet_name}'...")
            
            report.append(f"\n#### Sheet: {sheet_name}")
            report.append("- Cleaned Data:")
            
            # Get the original headers from the DataFrame's metadata
            if not df.empty:
                # Create a simplified table with just the data
                table_rows = []
                
                # Create a fixed-width Markdown table as requested by the user
                
                # Calculate column widths based on content
                col_widths = {}
                for col in df.columns:
                    # Start with the column name width
                    width = len(str(col))
                    
                    # Check original header width
                    if hasattr(df, 'attrs') and 'original_headers' in df.attrs:
                        orig_header = df.attrs['original_headers'].get(col, col)
                        width = max(width, len(str(orig_header)))
                    
                    # Check data width (first 5 rows)
                    for i in range(min(5, len(df))):
                        val = df[col].iloc[i]
                        if pd.notna(val):
                            width = max(width, len(str(val)))
                    
                    # Add some padding
                    col_widths[col] = width + 2
                
                # Format as requested:
                # 1. Header row (mapped column names)
                # 2. Separator row
                # 3. Original header row
                # 4. Confidence level row
                # 5. Data rows
                
                # Add header row (mapped column names)
                header_cells = []
                for col in df.columns:
                    header_cells.append(str(col).ljust(col_widths[col]))
                header_row = "|".join(header_cells)
                table_rows.append("|" + header_row + "|")
                
                # Add separator row
                separator_cells = []
                for col in df.columns:
                    separator_cells.append("-" * col_widths[col])
                separator_row = "|".join(separator_cells)
                table_rows.append("|" + separator_row + "|")
                
                # Add original header row
                orig_header_cells = []
                # Also prepare confidence scores for the next row
                confidence_cells = []
                
                if hasattr(df, 'attrs') and 'original_headers' in df.attrs:
                    # Use the original headers stored in the DataFrame's metadata
                    for col in df.columns:
                        orig_header = df.attrs['original_headers'].get(col, col)
                        orig_header_cells.append(str(orig_header).ljust(col_widths[col]))
                        
                        # Get confidence score for this mapping
                        score = 0
                        if hasattr(df, 'attrs') and 'original_to_mapped' in df.attrs:
                            for src, (tgt, scr) in df.attrs['original_to_mapped'].items():
                                if tgt == col and src == orig_header:
                                    score = scr
                                    break
                        
                        confidence_str = f"(score: {score})" if score > 0 else ""
                        confidence_cells.append(confidence_str.ljust(col_widths[col]))
                else:
                    # Fallback to using the mapped names if original headers are not available
                    for col in df.columns:
                        orig_header_cells.append(str(col).ljust(col_widths[col]))
                        confidence_cells.append("".ljust(col_widths[col]))
                
                orig_header_row = "|".join(orig_header_cells)
                table_rows.append("|" + orig_header_row + "|")
                
                # Add confidence level row
                confidence_row = "|".join(confidence_cells)
                table_rows.append("|" + confidence_row + "|")
                
                # Add data rows (up to 5)
                max_rows = min(5, len(df))
                for i in range(max_rows):
                    row = df.iloc[i]
                    data_cells = []
                    for col in df.columns:
                        val = row[col]
                        val_str = str(val) if pd.notna(val) else ""
                        data_cells.append(val_str.ljust(col_widths[col]))
                    data_row = "|".join(data_cells)
                    table_rows.append("|" + data_row + "|")
                
                # Add ellipsis if there are more rows
                if len(df) > max_rows:
                    table_rows.append(f"... (and {len(df) - max_rows} more rows)")
                
                # Add the table to the report
                report.append("\n" + "\n".join(table_rows))
            else:
                report.append("No data available.")
        
        report.append("")  # Add a blank line after each file
    
    report_text = "\n".join(report)
    
    if output_path:
        print(f"📝 Writing report to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Summary report saved to {output_path}")
        except Exception as e:
            print(f"❌ Error writing report to {output_path}: {str(e)}")
    else:
        print("ℹ️ No output path provided, returning report as string.")
    
    return report_text

# Example Usage
if __name__ == "__main__":
    import argparse
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description='Normalize financial data from Excel or CSV files.')
    parser.add_argument('path', help='Path to the input file or directory')
    parser.add_argument('--output', '-o', help='Output file path or directory')
    parser.add_argument('--mode', '-m', choices=['file', 'directory', 'auto'], default='auto',
                       help='Processing mode: file, directory, or auto (default: auto)')
    parser.add_argument('--report', '-r', help='Generate summary report and save to specified file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    debug_mode = not args.quiet
    
    # Determine if path is file or directory
    input_path = args.path
    mode = args.mode
    
    if mode == 'auto':
        if os.path.isfile(input_path):
            mode = 'file'
        elif os.path.isdir(input_path):
            mode = 'directory'
        else:
            print(f"❌ Invalid path: {input_path}")
            sys.exit(1)
    
    # Process based on mode
    if mode == 'file':
        if not os.path.isfile(input_path):
            print(f"❌ The specified path is not a file: {input_path}")
            sys.exit(1)
            
        cleaned_data = preprocess_file(input_path, debug=debug_mode)
        
        if cleaned_data:
            if args.output:
                save_cleaned_data(cleaned_data, args.output)
            else:
                for sheet, df in cleaned_data.items():
                    print(f"\n✅ Cleaned Data from '{sheet}':")
                    # Display only the mapped columns
                    print(df.head())  # Print first few rows
            
            # Generate summary report if requested
            if args.report:
                print(f"\n📝 Generating report to {args.report}...")
                # Create a dictionary with the file name as key and cleaned data as value
                results = {os.path.basename(input_path): cleaned_data}
                report = generate_summary_report(results, args.report)
        else:
            print("❌ No data to display")
            
    elif mode == 'directory':
        if not os.path.isdir(input_path):
            print(f"❌ The specified path is not a directory: {input_path}")
            sys.exit(1)
            
        results = process_directory(input_path, args.output, debug=debug_mode)
        
        if results:
            print(f"\n✅ Successfully processed {len(results)} files.")
            
            # Generate summary report if requested
            if args.report:
                report = generate_summary_report(results, args.report)
            
            if not args.output and not args.quiet:
                # Show preview of the first file if not saving
                first_file = next(iter(results.keys()))
                first_data = results[first_file]
                first_sheet = next(iter(first_data.keys()))
                print(f"\n📊 Preview of '{first_file}', sheet '{first_sheet}':")
                print(first_data[first_sheet].head())
        else:
            print("❌ No data to display")
            
    else:
        print(f"❌ Invalid mode: {mode}")
        sys.ex
