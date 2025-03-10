# Financial Data Cleaner

A modular Python package designed to clean, standardize, and normalize financial data from various file formats (CSV, XLS, XLSX). It automatically detects headers, maps columns to standard financial fields using fuzzy matching, and converts data types based on predefined standards.

## Features

-   **Automatic Header Detection:** Intelligently identifies header rows using multiple heuristics
-   **Fuzzy Column Mapping:** Maps column names to standard financial fields (e.g., ISIN, Fund Name, NAV Date) using fuzzy matching, even with variations in naming
-   **Data Type Conversion:** Converts columns to appropriate data types (date, numeric, string) based on standard field definitions
-   **Data Cleaning:** Removes empty rows and columns, standardizes currency codes, and handles missing values
-   **Dependency Checking:** Checks for required and optional dependencies and provides installation instructions
-   **Progress Bars:** Uses `tqdm` to display progress bars during file processing
-   **Caching:** Caches column mapping results for faster processing of files with similar column structures
-   **Detailed Logging:** Provides informative messages during the cleaning process, including warnings and error messages
-   **Duplicate Column Handling:** Detects and reports duplicate column names
-   **Single File or Directory Processing:** Allows processing of individual files or entire directories
-   **Processing Report:** Generates a summary report after processing files

## Package Structure

The package has been refactored into a modular structure:

```
financial_data_cleaner/
├── __init__.py
├── main.py
├── core/
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── header_detection.py
│   └── processor.py
├── data/
│   ├── __init__.py
│   └── mapping_data.py
├── mappers/
│   ├── __init__.py
│   └── column_mapper.py
└── utils/
    ├── __init__.py
    ├── dependencies.py
    ├── file_handlers.py
    ├── formatters.py
    └── validators.py
```

## Installation

For detailed installation instructions, please see [INSTALL.txt](INSTALL.txt).

### Quick Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/grungert/Financial-Data-Cleaner.git
   cd Financial-Data-Cleaner
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install pandas chardet openpyxl xlrd fuzzywuzzy python-Levenshtein tqdm
   ```

### Dependencies

- **Required:**
  - `pandas`: For data manipulation
  - `chardet`: For character encoding detection

- **Optional but recommended:**
  - `xlrd`: For reading `.xls` files
  - `openpyxl`: For reading `.xlsx` files
  - `fuzzywuzzy` and `python-Levenshtein`: For improved fuzzy matching of column names
  - `tqdm`: For progress bars

## Usage

### Command Line Interface

The package provides a command-line interface for processing files:

**1. Processing a single file:**

```bash
python -m financial_data_cleaner.main path/to/your/file.xlsx -o output_directory
```

**2. Processing a directory:**

```bash
python -m financial_data_cleaner.main path/to/your-directory/ -o output_directory
```

**3. Generate a summary report:**

```bash
python -m financial_data_cleaner.main path/to/your/file.xlsx -o output_directory -r report.md
```

**4. Suppress detailed output:**

```bash
python -m financial_data_cleaner.main path/to/your/file.xlsx -o output_directory -q
```

### API Usage

You can also use the package programmatically in your Python code:

```python
from financial_data_cleaner.core.processor import preprocess_file
from financial_data_cleaner.utils.file_handlers import save_cleaned_data

# Process a single file
file_path = 'path/to/your/file.xlsx'
cleaned_data = preprocess_file(file_path, debug=True)

# Save the cleaned data
if cleaned_data:
    output_path = 'path/to/output/file_cleaned.xlsx'
    save_cleaned_data(cleaned_data, output_path)
```

For processing a directory:

```python
from financial_data_cleaner.core.processor import process_directory

# Process a directory
directory_path = 'path/to/your/directory/'
output_dir = 'path/to/output/directory/'
results = process_directory(directory_path, output_dir, debug=True)
```

## Development

### Project Structure

The Financial Data Cleaner is organized into the following modules:

- **core**: Contains the main processing logic
  - `header_detection.py`: Algorithms for detecting header rows in financial data
  - `data_cleaner.py`: Core data cleaning and normalization functions
  - `processor.py`: High-level file and directory processing functions

- **data**: Contains data definitions and mappings
  - `mapping_data.py`: Standard field definitions and mapping data

- **mappers**: Contains column mapping functionality
  - `column_mapper.py`: Fuzzy matching and column standardization

- **utils**: Contains utility functions
  - `dependencies.py`: Dependency checking utilities
  - `file_handlers.py`: File I/O utilities for various formats
  - `formatters.py`: Output formatting utilities
  - `validators.py`: Data validation utilities

### Testing

To test the package with a sample file:

```bash
python test_cleaner.py
```

The test script will process a sample file and output the results to the `cleaned_output` directory.

## Contributing

Contributions to the Financial Data Cleaner are welcome! Here are some ways you can contribute:

1. **Report bugs:** If you find a bug, please create an issue detailing the problem
2. **Suggest features:** If you have ideas for new features or improvements, please create an issue
3. **Submit pull requests:** If you've fixed a bug or implemented a new feature, please submit a pull request

### Development Guidelines

- Follow PEP 8 coding standards
- Write docstrings for all functions, classes, and modules
- Add appropriate error handling and logging
- Include tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
