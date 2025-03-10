"""
Data validation utilities for the Financial Data Cleaner package.
"""
import re
import pandas as pd
from datetime import datetime

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
