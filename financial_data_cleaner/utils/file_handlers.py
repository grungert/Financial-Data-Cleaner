"""
File handling utilities for reading and processing financial data files.
"""
import os
import pandas as pd
import chardet

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


def save_cleaned_data(cleaned_data, output_path=None):
    """
    Saves the cleaned data to Excel or returns the first sheet if no path provided.
    """
    if not cleaned_data:
        return None
    
    if output_path:
        try:
            # Create a new Excel writer
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in cleaned_data.items():
                    # Write each sheet to the Excel file
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✅ Saved cleaned data to {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            return None
    else:
        # Return the first sheet if no output path
        if cleaned_data and len(cleaned_data) > 0:
            first_sheet = list(cleaned_data.values())[0]
            return first_sheet
        return None
