import pandas as pd
import numpy as np
import warnings
import logging
import os
from typing import Optional, Tuple, List, Union

# Configure logging (consider making this configurable outside the class if used as a library)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExcelFlattener:
    """
    Dynamically detects, parses, and flattens complex Excel sheets.

    This class processes sheets with multi-level headers and/or multi-column
    row indices into a simple 'wide' or 'long' tabular format suitable for
    databases or further analysis.

    It uses heuristics to automatically detect the header and index boundaries,
    assuming headers/indices are primarily textual and the data block is
    primarily numeric or contains common placeholders (e.g., '-', blank).
    The accuracy of auto-detection depends on the sheet structure adhering
    to these patterns. Manual specification of header rows (h) and index
    columns (c) is also supported for precise control.

    Core Steps:
    1. Load raw data (preserving strings).
    2. Detect header/index split (h, c) using scoring heuristic (if not specified).
    3. Extract header, index, and data blocks.
    4. Identify 'extra' text columns within the data block to treat as indices.
    5. Build an intermediate multi-indexed DataFrame.
    6. Clean data (placeholders to NaN) and coerce to numeric types.
    7. Flatten the result to the desired 'wide' or 'long' format.
    """

    def __init__(
        self,
        path: str,
        sheet_name: Union[str, int, None] = 0, # Default to first sheet
        max_header_rows: int = 7,
        max_index_cols: int = 7,
        text_col_thresh: float = 0.8
    ):
        """
        Initializes the ExcelFlattener.

        Args:
            path (str): Path to the Excel file (.xlsx, .xls, etc.).
            sheet_name (Union[str, int, None]): Target sheet name or 0-based index.
                                                Defaults to 0 (the first sheet).
            max_header_rows (int): Max rows considered for header during auto-detection.
            max_index_cols (int): Max columns considered for index during auto-detection.
            text_col_thresh (float): Ratio threshold (0 < threshold <= 1) for classifying
                                     a data column as an 'extra' text index column.
                                     Calculated as (text cells / non-blank cells).
        """
        if not 0 < text_col_thresh <= 1:
            raise ValueError("text_col_thresh must be between 0 (exclusive) and 1 (inclusive)")

        self.path = path
        self.sheet_name = sheet_name
        self.max_header_rows = max_header_rows
        self.max_index_cols = max_index_cols
        self.text_col_thresh = text_col_thresh

        self._df_raw: Optional[pd.DataFrame] = None # Stores raw data, lazy loaded



    def _load_raw_data(self) -> pd.DataFrame:
        """Loads the raw Excel or CSV file, preserving strings and handling common NAs."""
        if self._df_raw is None:
            logging.info(f"Loading data from '{self.path}'")
            try:
                na_strings = [
                    '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                    '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL',
                    'NaN', 'n/a', 'nan', 'null', ''
                ]

                # Check if path is an UploadedFile (e.g., from Streamlit)
                if hasattr(self.path, "name") and hasattr(self.path, "read"):
                    filename = self.path.name
                    file_ext = os.path.splitext(filename)[1].lower()

                    if file_ext in ['.xls', '.xlsx']:
                        self._df_raw = pd.read_excel(
                            self.path,
                            header=None,
                            dtype=str,
                            sheet_name=self.sheet_name,
                            keep_default_na=False
                        )
                    elif file_ext == '.csv':
                        self._df_raw = pd.read_csv(
                            self.path,
                            header=None,
                            dtype=str,
                            keep_default_na=False
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {file_ext}")
                else:
                    # Assume path is a string path
                    file_ext = os.path.splitext(self.path)[1].lower()

                    if file_ext in ['.xls', '.xlsx']:
                        self._df_raw = pd.read_excel(
                            self.path,
                            header=None,
                            dtype=str,
                            sheet_name=self.sheet_name,
                            keep_default_na=False
                        )
                    elif file_ext == '.csv':
                        self._df_raw = pd.read_csv(
                            self.path,
                            header=None,
                            dtype=str,
                            keep_default_na=False
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {file_ext}")

                self._df_raw = self._df_raw.replace(na_strings, np.nan)

            except FileNotFoundError:
                logging.error(f"Error: File not found at {self.path}")
                raise
            except Exception as e:
                logging.error(f"Error reading file '{self.path}': {e}")
                raise

        return self._df_raw



    def _score_split(self, df_raw: pd.DataFrame, h: int, c: int) -> float:
        """
        Scores a potential header/index split (h, c) based on content heuristics.

        Args:
            df_raw (pd.DataFrame): The raw DataFrame loaded as strings.
            h (int): Number of header rows being considered.
            c (int): Number of index columns being considered.

        Returns:
            float: A score between ~0 and ~1 indicating how well the split matches
                   the expected pattern (textual index, numeric/placeholder data).
                   Returns -1.0 for invalid dimensions.
        """
        # Check if the split dimensions are physically possible
        if h >= df_raw.shape[0] or c >= df_raw.shape[1]:
             return -1.0 # Invalid split

        # Define the three blocks based on the split
        index_block  = df_raw.iloc[h:, :c]
        data_block   = df_raw.iloc[h:, c:]

        # --- Score Data Block: Prefers numeric, NaN, or common placeholders ---
        data_total = data_block.size
        if data_total == 0:
            val_score = 0.0 # No data block, neutral score
        else:
            is_na = data_block.isna()
            # Common placeholders suggesting missing numeric data
            is_placeholder = data_block.isin(['-', '--']) # Exclude '' here, handled by NA replace earlier
            # Check if values can be coerced to numeric
            numeric_attempt = data_block.apply(pd.to_numeric, errors='coerce')
            is_numeric = numeric_attempt.notna()
            # A cell is considered 'valid' data if it's NA, placeholder, or numeric
            valid_data_cells = is_na | is_placeholder | is_numeric
            val_score = valid_data_cells.sum().sum() / data_total

        # --- Score Index Block: Prefers non-empty, non-numeric text ---
        index_total = index_block.size
        if index_total == 0:
            idx_score = 0.0 # No index block (c=0), neutral score
        else:
            is_not_na = index_block.notna()
            # Check if values *cannot* be coerced to numeric
            numeric_attempt_idx = index_block.apply(pd.to_numeric, errors='coerce')
            is_not_numeric = numeric_attempt_idx.isna()
            # Score based on cells being non-NA and non-numeric (i.e., likely text)
            is_textual = is_not_na & is_not_numeric
            idx_score = is_textual.sum().sum() / index_total

        # --- Combine Scores ---
        # Multiply scores: A good split requires *both* a well-formed index
        # (high idx_score) *and* a well-formed data block (high val_score).
        # Epsilon avoids score cancellation if one block is empty but the other is good.
        return (val_score + 1e-6) * (idx_score + 1e-6)

    def _detect_header_index_split(self) -> tuple[int, int]:
        """
        Auto-detects the optimal header rows (h) and index columns (c).

        Iterates through possible (h, c) combinations within defined limits,
        scoring each using `_score_split`. Selects the split with the highest
        score, using sum(h, c) as a tie-breaker (preferring simpler structures).

        Returns:
            tuple[int, int]: The detected (h, c) tuple.
        """
        df_raw = self._load_raw_data()
        best_score = -1.0
        # Sensible default split (1 header, 1 index), clipped to sheet dimensions
        default_h = min(1, df_raw.shape[0])
        default_c = min(1, df_raw.shape[1])
        best_split = (default_h, default_c)

        # Determine scan range based on limits and actual sheet dimensions
        # Need at least 1 row/col remaining for data
        max_h = min(self.max_header_rows, df_raw.shape[0] - 1)
        max_c = min(self.max_index_cols, df_raw.shape[1] - 1)

        # Handle very small sheets where detection range might be invalid
        if max_h < 1 or max_c < 0: # Allow scanning c=0
             logging.warning(f"Sheet dimensions ({df_raw.shape}) too small for "
                             f"meaningful header/index detection within limits "
                             f"(max_h={self.max_header_rows}, max_c={self.max_index_cols}). "
                             f"Using default guess ({best_split}).")
             # Ensure the default guess is valid
             h_final = max(0, min(best_split[0], df_raw.shape[0] - 1))
             c_final = max(0, min(best_split[1], df_raw.shape[1])) # c can be up to shape[1]
             return (h_final, c_final)

        # Scan potential header rows (1 to max_h)
        for h in range(1, max_h + 1):
            # Scan potential index columns (0 to max_c), allowing for header-only structures
            for c in range(0, max_c + 1):
                score = self._score_split(df_raw, h, c)

                # Update if score is better, or if score is equal but h+c is smaller (simpler)
                if score > best_score or \
                   (score == best_score and (h + c) < sum(best_split)):
                    best_score = score
                    best_split = (h, c)

        # Final check: Ensure the selected split is valid for the sheet dimensions,
        # as the loops might not have found a valid scoring split in edge cases.
        final_h = max(0, min(best_split[0], df_raw.shape[0] - 1))
        final_c = max(0, min(best_split[1], df_raw.shape[1])) # c can span full width if h<rows-1
        best_split = (final_h, final_c)

        logging.info(f"Detected split: header_rows={best_split[0]}, index_cols={best_split[1]} (Score: {best_score:.4f})")
        return best_split

    def _extract_blocks(self, df_raw: pd.DataFrame, h: int, c: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Slices the raw DataFrame into header, index, and data blocks based on (h, c).

        Args:
            df_raw (pd.DataFrame): The raw DataFrame.
            h (int): Number of header rows.
            c (int): Number of index columns.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the
            header_block, index_block, and data_block DataFrames. Returns empty
            DataFrames for blocks that don't exist (e.g., index_block if c=0).
        """
        logging.debug(f"Extracting blocks with h={h}, c={c}")
        rows, cols = df_raw.shape

        # Header block: Starts at (0, c). Use ffill(axis=1) to handle merged header cells.
        header_block = df_raw.iloc[:h, c:].ffill(axis=1) if c < cols and h > 0 else pd.DataFrame()

        # Index block: Starts at (h, 0). Reset index for alignment.
        index_block = df_raw.iloc[h:, :c].reset_index(drop=True) if c > 0 and h < rows else pd.DataFrame()

        # Data block: Starts at (h, c). Reset index for alignment.
        data_block = df_raw.iloc[h:, c:].reset_index(drop=True) if c < cols and h < rows else pd.DataFrame()

        return header_block, index_block, data_block

    def _identify_extra_text_cols(self, data_block: pd.DataFrame, c: int, threshold: float) -> list[int]:
        """
        Identifies columns within the data block that are mostly text.

        These are treated as additional descriptive columns belonging to the row index.

        Args:
            data_block (pd.DataFrame): The extracted data block.
            c (int): The starting column index of the data block in `df_raw`.
            threshold (float): The text ratio threshold for classification.

        Returns:
            list[int]: List of *absolute* column indices (relative to `df_raw`)
                       identified as extra text columns. Returns empty list if none found.
        """
        logging.debug(f"Identifying extra text columns with threshold {threshold}")
        if data_block.empty or data_block.shape[1] == 0:
             return []

        # Calculate text ratio for each column in the data block
        notna_counts = data_block.notna().sum()
        # Consider only '-' and '--' as placeholders for this calculation
        is_placeholder = data_block.isin(['-', '--'])
        numeric_attempt = data_block.apply(pd.to_numeric, errors='coerce')
        is_numeric = numeric_attempt.notna()

        # Textual = Not NA, Not Placeholder, Not Numeric
        is_text = data_block.notna() & ~is_placeholder & ~is_numeric
        text_counts = is_text.sum()

        # Calculate ratio; fillna(0) handles columns that are all NA.
        text_ratios = text_counts.divide(notna_counts).fillna(0)

        # Get columns in data_block exceeding the threshold
        relative_extra_indices = list(text_ratios[text_ratios > threshold].index)

        # Convert relative data_block indices (which might be positions or labels)
        # back to absolute indices relative to the original df_raw.
        # Assumption: df_raw.columns are RangeIndex(0, 1, ...) since header=None used.
        absolute_extra_indices = []
        if relative_extra_indices:
            if all(isinstance(idx, int) for idx in relative_extra_indices):
                 # If indices are positional integers (0, 1, ... within data_block)
                 absolute_extra_indices = [c + rel_idx for rel_idx in relative_extra_indices]
            else:
                 # If indices are labels (likely the original absolute column numbers)
                 absolute_extra_indices = list(relative_extra_indices) # Assume they are already absolute


        if absolute_extra_indices:
             # Ensure indices are within valid original column bounds
             ncols_raw = self._df_raw.shape[1]
             absolute_extra_indices = [idx for idx in absolute_extra_indices if idx < ncols_raw]
             if absolute_extra_indices:
                 logging.info(f"Identified {len(absolute_extra_indices)} extra text columns (absolute indices): {absolute_extra_indices}")
             else:
                 logging.debug("Relative extra text columns found, but none mapped to valid absolute indices.")
        else:
             logging.debug("No extra text columns identified.")

        return absolute_extra_indices

    def _build_multiindexed_dataframe(self, df_raw: pd.DataFrame, h: int, c: int, absolute_extra_col_indices: list[int]) -> pd.DataFrame:
        """
        Constructs the core multi-indexed DataFrame before final cleaning/flattening.

        Uses the detected/provided h, c, and any identified extra text columns
        to define the row and column multi-indices.

        Args:
            df_raw (pd.DataFrame): The raw DataFrame.
            h (int): Number of header rows.
            c (int): Number of base index columns.
            absolute_extra_col_indices (list[int]): Absolute indices of extra text cols.

        Returns:
            pd.DataFrame: A DataFrame with a (potentially multi-level) row index
                          and a (potentially multi-level) column index. Row index levels
                          are named 'idx_0', 'idx_1', ... Column index levels (if any)
                          are named 'lvl0', 'lvl1', ...
        """
        logging.debug("Building multi-indexed DataFrame")

        # --- Determine All Row Index Columns ---
        original_index_cols = list(range(c))
        all_index_cols = sorted(list(set(original_index_cols + absolute_extra_col_indices)))

        # --- Create Row Index ---
        if not all_index_cols:
            first_measure_col = 0 # All columns are potentially measures
            # Use a simple RangeIndex reflecting original row position if no index cols found
            row_multi_index = pd.RangeIndex(start=h, stop=df_raw.shape[0], name="original_row")
        else:
            # First column *not* part of the index determines start of measures
            first_measure_col = max(all_index_cols) + 1
            # Extract index data, ensure string type, fill NaNs for robust multi-index creation
            row_index_df = df_raw.iloc[h:, all_index_cols].reset_index(drop=True)
            row_index_df = row_index_df.astype(str).fillna('') # Use empty string for NaN in index keys

            # Create index (single or multi-level), naming levels 'idx_0', 'idx_1', ...
            if row_index_df.shape[1] == 1:
                row_multi_index = pd.Index(row_index_df.iloc[:, 0], name="idx_0")
            else:
                row_multi_index = pd.MultiIndex.from_frame(row_index_df, names=[f"idx_{i}" for i in range(len(all_index_cols))])

        # --- Create Column Index ---
        if first_measure_col >= df_raw.shape[1]:
            logging.warning("No measure columns identified after accounting for all index columns.")
            # Return DataFrame with only the row index if no data columns remain
            return pd.DataFrame(index=row_multi_index)

        # Extract header data for the measure columns
        header_for_cols = df_raw.iloc[:h, first_measure_col:]
        # Fill merged header cells rightward before creating index
        header_for_cols = header_for_cols.ffill(axis=1)
        # Ensure strings and fill NaNs for consistent tuple creation
        header_for_cols = header_for_cols.astype(str).fillna('')

        # Create column index (RangeIndex, single Index, or MultiIndex)
        if h == 0: # No header rows
            col_index = pd.RangeIndex(start=first_measure_col, stop=df_raw.shape[1])
            # Provide a default name for stacking later if needed
            col_index = col_index.rename("header_level_0")
        elif h == 1: # Single header row
             # Use more descriptive column name
             col_index = pd.Index(header_for_cols.iloc[0], name="header")
        else: # Multi-level header
            # Create tuples from header rows with more descriptive level names
            tuples = list(zip(*(header_for_cols.iloc[r] for r in range(h))))
            # Use more meaningful names for header levels
            level_names = [f"header_L{r+1}" for r in range(h)]
            col_index = pd.MultiIndex.from_tuples(tuples, names=level_names)

        # --- Extract Data and Build DataFrame ---
        body_data = df_raw.iloc[h:, first_measure_col:].reset_index(drop=True)

        # Create the main DataFrame structure
        body_df = pd.DataFrame(body_data.values, index=row_multi_index, columns=col_index)

        return body_df

    def _clean_and_coerce(self, body_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans placeholder strings and coerces data columns to numeric."""
        if body_df.empty:
            return body_df

        logging.debug("Cleaning and coercing data block")
        df = body_df.copy() # Work on a copy

        # Define placeholders to replace with NaN
        # Regex for '-' or '--' potentially surrounded by whitespace
        placeholders_regex = r'^\s*--?\s*$'
        # Specific strings (Excel errors, etc.) - '' was handled in _load_raw_data
        specific_placeholders = ['#VALUE!', '#DIV/0!']

        # --- Suppress FutureWarning from replace potentially downcasting ---
        # This warning occurs because replacing strings with NaN might change
        # an 'object' column to 'float', which is considered downcasting.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            # Perform replacements
            df = df.replace(placeholders_regex, np.nan, regex=True)
            df = df.replace(specific_placeholders, np.nan)

            # Optional: Explicitly infer types after replace. Not strictly
            # necessary as to_numeric follows, but doesn't hurt.
            try:
                df = df.infer_objects(copy=False)
            except TypeError: # Handle older pandas versions
                df = df.infer_objects()
        # --- End warning suppression ---

        # Attempt numeric conversion for all columns in the body DataFrame.
        # Assumes these are intended measure columns. Non-numeric become NaN.
        df = df.apply(pd.to_numeric, errors='coerce', axis=0)
        
        return df
        
    def _make_sql_friendly(self, name):
        """
        Convert column name to a SQL-friendly format with enhanced semantics.
        
        Args:
            name: The original column name that might contain problematic characters
            
        Returns:
            str: A SQL-friendly column name that preserves semantic meaning
        """
        import re
        
        if name is None:
            return "unknown"
            
        # Handle common abbreviations and prefixes
        name_str = str(name).strip()
        
        # Replace common special characters with meaningful text
        replacements = {
            '%': '_percent_',
            '#': '_num_',
            '$': '_dollars_',
            '&': '_and_',
            '+': '_plus_',
            '@': '_at_',
            '/': '_per_',
            '(': '_',
            ')': '_',
            '[': '_',
            ']': '_',
            '{': '_',
            '}': '_',
            '<': '_lt_',
            '>': '_gt_',
            '=': '_eq_',
            '"': '',
            "'": ''
        }
        
        for char, replacement in replacements.items():
            name_str = name_str.replace(char, replacement)
        
        # Replace other special chars with underscores
        cleaned = re.sub(r'[^\w]', '_', name_str)
        
        # Remove leading digits
        cleaned = re.sub(r'^[0-9]+', '', cleaned)
        
        # Make it more readable - replace multiple underscores with a single one
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Truncate very long names while preserving meaning
        if len(cleaned) > 63:  # Common SQL column name length limit
            words = cleaned.split('_')
            if len(words) > 3:
                # Keep first and last words, abbreviate middle ones
                cleaned = words[0] + '_' + '_'.join(w[0:3] for w in words[1:-1]) + '_' + words[-1]
                # If still too long, truncate
                if len(cleaned) > 63:
                    cleaned = cleaned[:60] + '_etc'
        
        # Ensure not empty
        if not cleaned or cleaned.isdigit():
            cleaned = f"col_{cleaned if cleaned else 'unknown'}"
            
        # Lowercase and trim underscores for SQL standard compliance
        return cleaned.strip('_').lower()
        
    def _infer_column_types(self, df):
        """
        Infer and convert column types to more SQL-friendly formats.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with improved column types
        """
        # Create a copy to avoid modifying original
        result = df.copy()
        
        for col in result.columns:
            # Skip processing for index columns which often have mixed types
            if col.startswith('idx_'):
                continue
                
            # Check for date patterns in object columns
            if result[col].dtype == 'object':
                non_null_values = result[col].dropna()
                if len(non_null_values) > 0:
                    # Try to convert percentages
                    if isinstance(non_null_values.iloc[0], str) and '%' in non_null_values.iloc[0]:
                        try:
                            # Convert percentage strings to float values
                            result[col] = non_null_values.str.replace('%', '').astype(float) / 100
                        except:
                            pass
                    
                    # Try to convert to datetime for date-like columns
                    try:
                        # Use flexible parsing for dates
                        temp_dates = pd.to_datetime(non_null_values, errors='coerce')
                        # Only convert if most values parse successfully (>80%)
                        if temp_dates.notna().mean() > 0.8:
                            result[col] = pd.to_datetime(result[col], errors='coerce')
                    except:
                        pass
                        
        return result
        
    def _detect_semantic_column_type(self, col_name, sample_values):
        """
        Detect the semantic type of a column based on its name and values.
        
        Args:
            col_name: The column name
            sample_values: Sample values from the column
            
        Returns:
            str: Semantic type like 'id', 'date', 'amount', etc.
        """
        import re
        
        col_lower = str(col_name).lower()
        
        # Check for common column type patterns
        if any(id_pattern in col_lower for id_pattern in ['id', 'code', 'key', 'num', 'no', 'number']):
            return 'id'
            
        if any(date_pattern in col_lower for date_pattern in ['date', 'dt', 'day', 'month', 'year', 'time']):
            return 'date'
            
        if any(amount_pattern in col_lower for amount_pattern in ['amount', 'price', 'cost', 'fee', 'salary', 'budget', 'revenue', 'sale']):
            return 'amount'
            
        if any(name_pattern in col_lower for name_pattern in ['name', 'title', 'label', 'description']):
            return 'name'
            
        if any(status_pattern in col_lower for status_pattern in ['status', 'state', 'flag', 'type', 'category']):
            return 'category'
            
        # Try to infer from sample values
        non_null_samples = [v for v in sample_values if v is not None and pd.notna(v)][:20]  # Limit to 20 samples
        
        if non_null_samples:
            # Check for mostly numeric values
            numeric_count = sum(1 for v in non_null_samples if isinstance(v, (int, float)) or 
                              (isinstance(v, str) and re.match(r'^[\d.,]+$', str(v))))
            
            if numeric_count / len(non_null_samples) > 0.8:
                # Mostly numeric - check if it's likely an ID
                if all(float(str(v).replace(',', '')).is_integer() for v in non_null_samples if re.match(r'^[\d.,]+$', str(v))):
                    return 'id'
                else:
                    return 'numeric'
            
            # Check for mostly short text values (likely categories)
            text_lengths = [len(str(v)) for v in non_null_samples if isinstance(v, str)]
            if text_lengths and sum(text_lengths) / len(text_lengths) < 15:
                return 'category'
        
        # Default type
        return 'unknown'
        
    def _flatten_result(self, cleaned_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Flattens the cleaned, multi-indexed DataFrame to 'wide' or 'long' format with improved column naming.

        Args:
            cleaned_df (pd.DataFrame): The DataFrame after cleaning and coercion.
            method (str): 'wide' or 'long'.

        Returns:
            pd.DataFrame: The final flattened DataFrame with SQL-friendly column names.
        """
        if cleaned_df.empty:
            # Return empty frame but preserve index names if they exist
            index_names = []
            if isinstance(cleaned_df.index, pd.MultiIndex):
                index_names = [name for name in cleaned_df.index.names if name is not None]
            elif cleaned_df.index.name:
                index_names = [cleaned_df.index.name]
            return pd.DataFrame(columns=index_names)

        logging.debug(f"Flattening result using method: {method}")

        if method == 'wide':
            # Wide format: Create more meaningful and SQL-friendly column names
            flat_df = cleaned_df.copy()
            
            if isinstance(flat_df.columns, pd.MultiIndex):
                # Join levels with '_', skipping empty/NA levels and stripping whitespace
                new_cols = []
                for col_tuple in flat_df.columns.values:
                    # Filter valid parts and join with underscores
                    parts = [str(level).strip() for level in col_tuple 
                            if pd.notna(level) and str(level).strip()]
                    new_col = "_".join(parts) if parts else "unnamed"
                    # Make SQL friendly
                    new_col = self._make_sql_friendly(new_col)
                    new_cols.append(new_col)
                
                # Check for duplicates and make unique if needed
                if len(set(new_cols)) < len(new_cols):
                    counts = {}
                    for i, name in enumerate(new_cols):
                        if name in counts:
                            counts[name] += 1
                            new_cols[i] = f"{name}_{counts[name]}"
                        else:
                            counts[name] = 0
                            
                flat_df.columns = new_cols
            else:
                # Single level column index - make SQL friendly
                flat_df.columns = [self._make_sql_friendly(col) for col in flat_df.columns]
                
            # Move row index levels into columns
            result_df = flat_df.reset_index()
            
            # Make index column names SQL-friendly too
            result_df.columns = [self._make_sql_friendly(col) for col in result_df.columns]
            return result_df

        elif method == 'long':
            # Long format: Stack columns into rows with better naming
            if isinstance(cleaned_df.columns, pd.MultiIndex):
                # Stack all column levels
                stacked = cleaned_df.stack(level=list(range(cleaned_df.columns.nlevels)))
            else:
                # Single level column index
                level_name = cleaned_df.columns.name if cleaned_df.columns.name else 'variable'
                stacked = cleaned_df.stack()
                
            # Stack results in a Series; rename the data part to 'value'
            long_df = stacked.rename('value')
            
            # Move all index levels into columns
            result_df = long_df.reset_index()
            
            # Make column names SQL-friendly
            result_df.columns = [self._make_sql_friendly(col) for col in result_df.columns]
            
            # Re-apply numeric coercion as stacking can sometimes change dtypes
            if 'value' in result_df.columns:
                result_df['value'] = pd.to_numeric(result_df['value'], errors='coerce')
                
            return result_df
        else:
            # Should not be reachable if validation in flatten() works, but good practice
            raise ValueError(f"Invalid flatten method: '{method}'. Must be 'wide' or 'long'.")

    def detect_potential_relationships(self, all_dfs_dict):
        """
        Detect potential foreign key relationships between flattened tables.
        
        Args:
            all_dfs_dict: Dictionary of sheet_name -> flattened DataFrame
            
        Returns:
            List of dictionaries describing potential relationships
        """
        relationships = []
        
        # Get all tables and their column sets
        table_columns = {name: set(df.columns) for name, df in all_dfs_dict.items()}
        
        # Examine each pair of tables
        for table1, cols1 in table_columns.items():
            for table2, cols2 in table_columns.items():
                if table1 == table2:
                    continue
                    
                # Look for columns with similar names (not just exact matches)
                # This helps catch variations like 'customer_id' and 'customerid'
                potential_matches = []
                for col1 in cols1:
                    col1_clean = self._make_sql_friendly(col1).lower()
                    for col2 in cols2:
                        col2_clean = self._make_sql_friendly(col2).lower()
                        
                        # Check for name similarity
                        if col1_clean == col2_clean or \
                           (len(col1_clean) > 3 and col1_clean in col2_clean) or \
                           (len(col2_clean) > 3 and col2_clean in col1_clean):
                            potential_matches.append((col1, col2))
                
                # Validate potential relationships by checking values
                for col1, col2 in potential_matches:
                    df1 = all_dfs_dict[table1]
                    df2 = all_dfs_dict[table2]
                    
                    # Get non-null values
                    vals1 = set(df1[col1].dropna().unique())
                    vals2 = set(df2[col2].dropna().unique())
                    
                    # Skip empty sets
                    if not vals1 or not vals2:
                        continue
                    
                    # Calculate overlap percentage
                    intersection = len(vals1.intersection(vals2))
                    min_set_size = min(len(vals1), len(vals2))
                    
                    # If significant overlap exists
                    if intersection > 0 and min_set_size > 0:
                        overlap_percent = (intersection / min_set_size) * 100
                        
                        # Determine relationship direction - which is likely the parent table
                        if len(vals1) <= len(vals2) and overlap_percent > 70:
                            relationships.append({
                                'parent_table': table2,
                                'child_table': table1,
                                'parent_column': col2,
                                'child_column': col1,
                                'match_percentage': overlap_percent
                            })
                        elif len(vals2) <= len(vals1) and overlap_percent > 70:
                            relationships.append({
                                'parent_table': table1,
                                'child_table': table2,
                                'parent_column': col1,
                                'child_column': col2,
                                'match_percentage': overlap_percent
                            })
        
        return relationships
    
    def _post_process_for_sql(self, df, sheet_name):
        """
        Apply post-processing to optimize for SQL usage.
        
        Args:
            df: DataFrame to process
            sheet_name: Name of the sheet
            
        Returns:
            DataFrame optimized for SQL queries
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # 1. Apply type inference
        result_df = self._infer_column_types(result_df)
        
        # 2. Add metadata 
        # Add source sheet column - useful when querying across sheets
        result_df['source_sheet'] = sheet_name
        
        # 3. Add derived columns for time-based data
        for col in result_df.columns:
            # Add date components for datetime columns
            if pd.api.types.is_datetime64_any_dtype(result_df[col]):
                try:
                    col_prefix = col + '_'
                    result_df[col_prefix + 'year'] = result_df[col].dt.year
                    result_df[col_prefix + 'month'] = result_df[col].dt.month
                    result_df[col_prefix + 'quarter'] = result_df[col].dt.quarter 
                except Exception as e:
                    logging.warning(f"Could not create date components for {col}: {e}")
        
        # 4. Create semantic type information (useful for queries)
        semantic_types = {}
        for col in result_df.columns:
            sample_values = result_df[col].dropna().head(20).tolist()
            semantic_types[col] = self._detect_semantic_column_type(col, sample_values)
            
        # Store semantic types in DataFrame metadata (doesn't affect SQL but useful for insight generation)
        result_df.attrs['semantic_types'] = semantic_types
        
        return result_df
    
    def flatten(
        self,
        h: Optional[int] = None,
        c: Optional[int] = None,
        threshold: Optional[float] = None,
        method: str = 'wide',
        sql_optimize: bool = True
    ) -> pd.DataFrame:
        """
        Loads, processes, and flattens the specified Excel sheet with SQL optimization.

        This is the main public method orchestrating the entire process.

        Args:
            h (Optional[int]): Manually specify the number of header rows (>=0).
                               If None, auto-detection is performed.
            c (Optional[int]): Manually specify the number of index columns (>=0).
                               If None, auto-detection is performed.
            threshold (Optional[float]): Override the `text_col_thresh` for this run.
                                         Used for identifying extra text columns.
            method (str): Output format: 'wide' or 'long'. Defaults to 'wide'.
            sql_optimize (bool): Whether to apply SQL optimization post-processing.

        Returns:
            pd.DataFrame: The processed and flattened DataFrame optimized for SQL.

        Raises:
            FileNotFoundError: If the Excel file path is invalid.
            ValueError: If parameters (h, c, threshold, method) are invalid.
            Exception: For errors during Excel reading or processing steps.
        """
        # --- Parameter Validation ---
        if method not in ['wide', 'long']:
            raise ValueError(f"Invalid method '{method}'. Choose 'wide' or 'long'.")
        current_threshold = threshold if threshold is not None else self.text_col_thresh
        if not 0 < current_threshold <= 1:
             raise ValueError("text_col_thresh must be between 0 (exclusive) and 1 (inclusive)")

        try:
            # --- Step 0: Load Data ---
            df_raw = self._load_raw_data()

            if df_raw.empty:
                logging.warning(f"Input sheet '{self.sheet_name}' is empty.")
                return pd.DataFrame() # Return empty DataFrame

            # --- Step 1: Determine Header/Index Split ---
            if h is None or c is None:
                detected_h, detected_c = self._detect_header_index_split()
                h = h if h is not None else detected_h
                c = c if c is not None else detected_c
                logging.info(f"Using detected/default split: header_rows={h}, index_cols={c}")
            else:
                # Validate manually provided h and c
                logging.info(f"Using provided split: header_rows={h}, index_cols={c}")
                if not (isinstance(h, int) and h >= 0 and h < df_raw.shape[0]):
                     raise ValueError(f"Provided h={h} is invalid for sheet with {df_raw.shape[0]} rows.")
                # c can equal shape[1] if there are only index columns specified
                if not (isinstance(c, int) and c >= 0 and c <= df_raw.shape[1]):
                     raise ValueError(f"Provided c={c} is invalid for sheet with {df_raw.shape[1]} columns.")

            # --- Step 2: Extract Blocks ---
            header_block, index_block, data_block = self._extract_blocks(df_raw, h, c)

            # --- Step 3: Identify Extra Text Columns ---
            absolute_extra_indices = self._identify_extra_text_cols(data_block, c, current_threshold)

            # --- Step 4: Build Multi-indexed DataFrame ---
            body_df = self._build_multiindexed_dataframe(df_raw, h, c, absolute_extra_indices)

            # --- Step 5: Clean Data & Coerce Types ---
            cleaned_df = self._clean_and_coerce(body_df)

            # --- Step 6: Flatten to Final Format ---
            final_df = self._flatten_result(cleaned_df, method)
            
            # --- Step 7: Apply SQL-specific post-processing (new) ---
            if sql_optimize:
                sheet_name = self.sheet_name if isinstance(self.sheet_name, str) else f"Sheet_{self.sheet_name}"
                final_df = self._post_process_for_sql(final_df, sheet_name)

            logging.info(f"Flattening complete. Result shape: {final_df.shape}")
            return final_df

        except Exception as e:
            # Log any exception during the process and re-raise
            logging.exception(f"Error during flattening process for {self.path} sheet {self.sheet_name}:")
            raise e
        
    @staticmethod
    def read_instructions(filepath: str, encoding: str = 'utf-8') -> str:
        with open(filepath, 'r', encoding=encoding) as f: 
            return f.read()