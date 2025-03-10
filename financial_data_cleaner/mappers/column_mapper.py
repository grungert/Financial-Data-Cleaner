"""
Column mapping functions for standardizing financial data fields.
"""
import pandas as pd
from ..data.mapping_data import standard_fields

# Global cache for column mappings
_column_mapping_cache = {}

# Try importing optional dependencies
try:
    from fuzzywuzzy import process, fuzz
except ImportError:
    # Define a simple fallback implementation
    class fuzz:
        @staticmethod
        def token_sort_ratio(s1, s2):
            s1 = str(s1).lower()
            s2 = str(s2).lower()
            return 100 if s1 == s2 else (70 if s1 in s2 or s2 in s1 else 0)

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
    from ..data.mapping_data import standard_field_types
    
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
