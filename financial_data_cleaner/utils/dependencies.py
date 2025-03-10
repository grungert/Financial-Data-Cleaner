"""
Dependency management utilities for the Financial Data Cleaner package.
"""

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
