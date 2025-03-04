# Financial Data Cleaner

This script is designed to clean and standardize financial data from various file formats (CSV, XLS, XLSX). It automatically detects headers, maps columns to standard financial fields using fuzzy matching, and converts data types based on predefined standards.

## Features

-   **Automatic Header Detection:** Intelligently identifies header rows using multiple heuristics.
-   **Fuzzy Column Mapping:** Maps column names to standard financial fields (e.g., ISIN, Fund Name, NAV Date) using fuzzy matching, even with variations in naming.
-   **Data Type Conversion:** Converts columns to appropriate data types (date, numeric, string) based on standard field definitions.
-   **Data Cleaning:** Removes empty rows and columns, standardizes currency codes, and handles missing values.
-   **Dependency Checking:** Checks for required and optional dependencies and provides installation instructions.
-   **Progress Bars:** Uses `tqdm` to display progress bars during file processing (if installed).
-   **Caching:** Caches column mapping results for faster processing of files with similar column structures.
-   **Detailed Logging:** Provides informative messages during the cleaning process, including warnings and error messages.
-   **Duplicate Column Handling:** Detects and reports duplicate column names.
-   **Single File or Directory Processing:** Allows processing of individual files or entire directories.
-   **Processing Report:** Generates a summary report after processing files.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/grungert/Financial-Data-Cleaner.git
    cd Financial-Data-Cleaner
    ```

2.  **Install required dependencies:**

    ```bash
    pip install pandas
    ```

3.  **Install optional dependencies for full functionality:**

    ```bash
    pip install xlrd openpyxl fuzzywuzzy python-Levenshtein tqdm
    ```

    -   `xlrd`: For reading `.xls` files.
    -   `openpyxl`: For reading `.xlsx` files.
    -   `fuzzywuzzy` and `python-Levenshtein`: For improved fuzzy matching of column names.
    -   `tqdm`: For progress bars.

## Usage

**Running the script:**

You can run the script to process either a single file or an entire directory.

**1. Processing a Single File:**

```bash
python your_script_name.py --file path/to/your/file.xlsx

**2. Processing a directory:**

```bash
python your_script_name.py --file path/to/your-directory/

**3. Processing a directory and create report:**

```bash
python your_script_name.py --file path/to/your-directory/ --report "path/report.md"
