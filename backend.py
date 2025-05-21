import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import streamlit as st
import pandas as pd
import sqlite3
import logging
import json
import numpy as np  
import time
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_azure_openai():
    """
    Setup Azure OpenAI client with credentials from Streamlit secrets.
    Returns LLM instance for use in generating insights.
    """
    try:
        logger.info("Setting up Azure OpenAI connection")
        
        # Get credentials from Streamlit secrets
        TENANT = st.secrets.TENANT
        CLIENT_ID = st.secrets.CLIENT_ID
        CLIENT_SECRET = st.secrets.CLIENT_SECRET
        
        # Set up authentication
        credential = ClientSecretCredential(TENANT, CLIENT_ID, CLIENT_SECRET)
        VAULT_URL = st.secrets.VAULT_URL
        client = SecretClient(vault_url=VAULT_URL, credential=credential)
        
        # Get API key from vault
        openai_key = client.get_secret("GenAIBIMInternalCapabilityOpenAIKey")
        
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_key.value
        
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4",  # exact deployment name in Azure
            azure_endpoint=st.secrets.azure_endpoint,
            api_version="2023-12-01-preview",
            temperature=0.0  # Lower temperature for more consistent SQL generation
        )
        
        # Initialize embeddings (keeping for future use)
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embeddings",  # must match your Azure deployment
            azure_endpoint=st.secrets.azure_endpoint,
            api_version="2023-12-01-preview"
        )
        
        logger.info("Azure OpenAI setup successful")
        return llm
        
    except Exception as e:
        logger.error(f"Error setting up Azure OpenAI: {str(e)}")
        raise Exception(f"Failed to initialize Azure OpenAI: {str(e)}")

# SQL capabilities reference - useful for LLM prompting
SQL_CAPABILITIES = """
SQLite SQL Capabilities:
- Aggregate functions: COUNT(), SUM(), AVG(), MIN(), MAX(), TOTAL()
- Filtering: WHERE, LIKE, BETWEEN, IN, IS NULL, IS NOT NULL
- Logical operators: AND, OR, NOT
- Grouping: GROUP BY, HAVING
- Sorting: ORDER BY (ASC/DESC), COLLATE
- Joins: INNER JOIN, LEFT JOIN, RIGHT JOIN (simulated via LEFT JOIN + COALESCE), CROSS JOIN
- Subqueries: Supported in SELECT, FROM, WHERE, and HAVING clauses
- Set operations: UNION, UNION ALL, INTERSECT, EXCEPT
- Date and time functions: date(), time(), datetime(), julianday(), strftime()
- String functions: LENGTH(), UPPER(), LOWER(), SUBSTR(), TRIM(), REPLACE(), INSTR()
- Math functions: ABS(), ROUND(), CEIL(), FLOOR(), RANDOM(), POW(), SQRT()
- Type casting: CAST(expr AS TYPE)
- Case expressions: CASE WHEN ... THEN ... ELSE ... END
- Column constraints: PRIMARY KEY, NOT NULL, UNIQUE, DEFAULT, CHECK
- Table constraints: FOREIGN KEY (limited enforcement depending on pragma)
- Indexes: CREATE INDEX, UNIQUE INDEX
- Views: CREATE VIEW, DROP VIEW
- Transactions: BEGIN, COMMIT, ROLLBACK
- PRAGMA statements: Used for metadata and configuration (e.g., `PRAGMA table_info(table_name);`)
- Common Table Expressions (CTEs): WITH clause
- Window functions: ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(), LEAD(), LAG(), FIRST_VALUE(), LAST_VALUE(), OVER(PARTITION BY ... ORDER BY ...)
- JSON functions (SQLite ≥ 3.38): json(), json_extract(), json_object(), json_array(), json_each(), json_type(), etc.
- Full-text search (FTS5 module): MATCH operator, full-text indexes (if enabled)
- Manual statistical functions:
    - STDDEV(): Simulate using SQRT(AVG(column * column) - AVG(column) * AVG(column))
    - VARIANCE(): AVG(column * column) - AVG(column) * AVG(column)
"""


def get_schema_description(df):
    """
    Generate a detailed schema description from the dataframe, 
    including column names, data types, and example values.
    """
    schema = []
    
    for col in df.columns:
        # Determine the data type
        dtype = str(df[col].dtype)
        
        # Get example values, handling potential errors
        try:
            sample_vals = df[col].dropna().unique()[:3]
            if len(sample_vals) == 0:
                examples = "No non-null examples available"
            else:
                # Format example values based on data type
                if dtype == 'object':
                    # For string values, truncate if too long
                    formatted_vals = [f'"{str(val)[:20]}..."' if len(str(val)) > 20 else f'"{val}"' for val in sample_vals]
                    examples = ", ".join(formatted_vals)
                elif 'datetime' in dtype:
                    # For datetime values, format as ISO
                    examples = ", ".join([f'"{val}"' for val in sample_vals])
                else:
                    # For numeric values
                    examples = ", ".join([str(val) for val in sample_vals])
            
            # Count null values
            null_count = df[col].isna().sum()
            null_percentage = (null_count / len(df)) * 100 #calculate Null percentage
            
            # Add detailed column information
            schema.append(f"- `{col}` ({dtype}) | Nulls: {null_count} ({null_percentage:.1f}%) | Examples: {examples}")
        
        except Exception as e:
            # Handle any errors in processing column info
            schema.append(f"- `{col}` ({dtype}) | Error getting examples: {str(e)}")
    
    return "\n".join(schema)

def preprocess_column_name(col):
    """
    Standardize column names for SQL compatibility while preserving semantics.
    Handles multi-level column names better and makes them SQL-friendly.
    """
    import re
    
    if col is None:
        return "unknown_column"
        
    # If it's a tuple (from MultiIndex columns), join the parts with underscores
    if isinstance(col, tuple):
        # Filter out None and empty strings, join valid parts with underscores
        parts = [str(part).strip() for part in col if part is not None and str(part).strip()]
        col_str = "_".join(parts) if parts else "unnamed_column"
    else:
        col_str = str(col)
    
    # Replace any characters that aren't alphanumeric or underscores with an underscore
    cleaned = re.sub(r'[^\w]', '_', col_str)
    
    # Remove leading numbers (SQLite doesn't like column names starting with numbers)
    cleaned = re.sub(r'^[0-9]+', '', cleaned)
    
    # Ensure the column name isn't empty after cleaning
    if not cleaned or cleaned.isdigit():
        cleaned = f"col_{cleaned if cleaned else 'unknown'}"
        
    # Strip leading/trailing underscores and lowercase
    cleaned = cleaned.strip('_').lower()
    
    # Replace multiple consecutive underscores with a single one
    cleaned = re.sub(r'_+', '_', cleaned)
    
    return cleaned

def normalize_indian_phone_number(val):
    """
    Normalize Indian phone numbers to a standard 10-digit format.

    - Handles null/NaN values by returning them unchanged
    - Removes "+91" or "+91-" prefix if present
    - Removes all non-digit characters
    - Returns last 10 digits if number is too long
    - Returns NaN if cleaned number has less than 10 digits

    Args:
        val: The phone number value to normalize
        
    Returns:
        Normalized phone number as string (10 digits) or np.nan
    """
    if pd.isna(val):
        return val
    
    # Convert to string and strip whitespaces
    str_val = str(val).strip()
    
    # Remove "+91-" or "+91" prefix
    if str_val.startswith('+91-'):
        str_val = str_val[4:]
    elif str_val.startswith('+91'):
        str_val = str_val[3:]
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', str_val)
    
    # If cleaned number is empty or less than 10 digits, return NaN
    if not digits_only or len(digits_only) < 10:
        return np.nan
    
    # If more than 10 digits, take the last 10 digits
    if len(digits_only) > 10:
        return np.nan
    
    # Exactly 10 digits
    return digits_only

def is_complex_sheet(df, ws):
    """
    Advanced detection of complex Excel sheets that need flattening.
    
    Args:
        df: DataFrame representation of the worksheet
        ws: openpyxl worksheet object
        
    Returns:
        bool: True if the sheet is complex and needs flattening
    """
    import numpy as np
    import re
    
    # Check for empty dataframe
    if df.empty or df.dropna(how='all').empty:
        return False
    
    # 1. Structural checks
    has_merged_cells = len(ws.merged_cells.ranges) > 0
    blank_top_rows = sum(df.iloc[i].isnull().all() for i in range(min(10, len(df))))
    row_lengths = df.dropna(how='all').apply(lambda x: x.count(), axis=1)
    inconsistent_rows = row_lengths.nunique() > 1
    
    # 2. Header checks
    repeated_headers = any(df.iloc[i].equals(df.iloc[0]) for i in range(1, min(10, len(df))))
    
    # 3. Check for multi-level headers
    potential_header_rows = min(5, len(df))  # Check first 5 rows for patterns
    text_ratio_in_rows = []
    for i in range(potential_header_rows):
        if i < len(df):
            row = df.iloc[i]
            total_cells = row.count()
            if total_cells == 0:  # Skip completely empty rows
                text_ratio_in_rows.append(0)
                continue
                
            # Count cells that are text (not numeric or NaN)
            text_cells = sum(1 for val in row if isinstance(val, str) and not re.match(r'^[\d\s.,-]+$', str(val)))
            text_ratio_in_rows.append(text_cells / total_cells if total_cells > 0 else 0)
    
    # Detect if early rows are text-heavy (potential headers)
    has_text_header_pattern = any(ratio > 0.7 for ratio in text_ratio_in_rows)
    
    # 4. Check for indentation in cells (common in hierarchical data)
    indented_cells = 0
    for i in range(min(20, len(df))):
        if i < len(df):
            for val in df.iloc[i]:
                if isinstance(val, str) and val.startswith((' ', '\t')) and val.strip():
                    indented_cells += 1
    has_indentation = indented_cells >= 3  # Arbitrary threshold
    
    # 5. Check for sparse data pattern (many gaps)
    dense_regions = 0
    top_half = df.iloc[:min(20, len(df)), :]
    dense_threshold = 0.7  # How dense a region needs to be to count
    
    for i in range(0, len(top_half.columns), 3):  # Check blocks of 3 columns
        end_col = min(i + 3, len(top_half.columns))
        block = top_half.iloc[:, i:end_col]
        density = block.count().sum() / (block.shape[0] * block.shape[1])
        if density > dense_threshold:
            dense_regions += 1
            
    has_sparse_pattern = dense_regions > 1 and dense_regions < len(top_half.columns) // 3  # Multiple dense regions with gaps
    
    # 6. Check for irregular layout signatures (common in reports and summaries)
    num_threshold = 0.7  # 70% numeric is a signature of data areas
    text_threshold = 0.7  # 70% text is a signature of header/label areas
    
    # Analyze first few rows and columns for regions
    top_rows = min(10, len(df))
    left_cols = min(5, len(df.columns))
    
    # Check if there's a numeric-heavy region after text-heavy top rows
    has_numeric_region = False
    for i in range(top_rows, min(top_rows + 10, len(df))):
        if i < len(df):
            row = df.iloc[i].dropna()
            num_ratio = sum(1 for val in row if isinstance(val, (int, float)) or 
                          (isinstance(val, str) and re.match(r'^[\d.,]+$', val))) / len(row) if len(row) > 0 else 0
            if num_ratio > num_threshold:
                has_numeric_region = True
                break
    
    # Build advanced complexity score
    complexity_factors = [
        has_merged_cells,  # Presence of merged cells
        blank_top_rows >= 2,  # Multiple blank rows at the top
        inconsistent_rows,  # Rows have different numbers of valid cells
        repeated_headers,  # Header row appears multiple times
        has_text_header_pattern,  # Text-heavy rows at the top (potential headers)
        has_indentation,  # Indented cells (hierarchical data)
        has_sparse_pattern,  # Sparse data patterns
        has_numeric_region and any(ratio > text_threshold for ratio in text_ratio_in_rows)  # Text headers with numeric data
    ]
    
    # Log detection results
    logging.debug(f"Sheet complexity factors detected: {complexity_factors}")
    
    # Weighted score - each factor contributes to complexity score
    complexity_score = sum(complexity_factors)
    
    # Return True if complexity score meets the threshold
    return complexity_score >= 3  # Require at least 3 complexity indicators

def preprocess_and_store_sheet(df, sheet_name, db_path="data_store.db"):
    """
    Preprocess a single dataframe (sheet) and store it in SQLite.
    Returns the table name for reference.
    """
    try:
        # Sanitize sheet name for SQL table name
        table_name = preprocess_column_name(sheet_name)
        logger.info(f"Processing sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Standardize column names
        processed_df.columns = [preprocess_column_name(col) for col in processed_df.columns]
        
        # Normalize phone number columns
        for col in processed_df.columns:
            if 'phone' in col.lower() or 'mobile' in col.lower():
                logger.info(f"Normalizing phone numbers in column '{col}'")
                # IMPORTANT: Convert column to string type before processing
                processed_df[col] = processed_df[col].astype(str)
                processed_df[col] = processed_df[col].apply(normalize_indian_phone_number)
                        
        # Basic data cleaning
        # Convert date-like strings to actual dates if possible
        for col in processed_df.columns:
            # Try to convert date-like columns
            if processed_df[col].dtype == 'object':
                try:
                    # Check if column might contain dates
                    date_sample = processed_df[col].dropna().iloc[0] if not processed_df[col].dropna().empty else None
                    if date_sample and isinstance(date_sample, str) and len(date_sample) > 6:
                        if any(x in date_sample for x in ['/', '-', ':', 'Jan', 'Feb', 'Mar']):
                            processed_df[col] = pd.to_datetime(processed_df[col], errors='ignore')
                except:
                    pass  # Skip if conversion fails
        
        # Fix any remaining issues with the dataframe
        processed_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        # Save to SQLite
        conn = sqlite3.connect(db_path)
        processed_df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Get table schema from SQLite for verification
        cursor = conn.cursor()
        schema_info = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        
        # Log schema information
        logger.info(f"Created SQLite table '{table_name}' with {len(schema_info)} columns")
        conn.close()
        
        return table_name
        
    except Exception as e:
        logger.error(f"Error in preprocessing sheet '{sheet_name}': {str(e)}")
        raise Exception(f"Failed to process and store sheet '{sheet_name}': {str(e)}")

def preprocess_and_store_in_sqlite(df_dict, db_path="data_store.db"):
    """
    Preprocess multiple dataframes from different sheets and store them in SQLite.
    Returns a dictionary mapping sheet names to table names.
    """
    try:
        logger.info(f"Processing {len(df_dict)} sheets/tables")
        
        table_dict = {}
        
        # Process each sheet in the Excel file
        for sheet_name, df in df_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                table_name = preprocess_and_store_sheet(df, sheet_name, db_path)
                table_dict[sheet_name] = table_name
        
        # Get all tables from SQLite for verification
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        logger.info(f"SQLite database now contains {len(tables)} tables: {[t[0] for t in tables]}")
        conn.close()
        
        return table_dict
        
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise Exception(f"Failed to process and store data: {str(e)}")

def get_database_schema(db_path="data_store.db"):
    """
    Get schema information for all tables in the database.
    Returns a dictionary mapping table names to their column information.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            # Get column info for each table
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            schema_info[table_name] = [
                {"name": col[1], "type": col[2]} for col in columns
            ]
        
        conn.close()
        return schema_info
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        return {}

def generate_sql_prompt(user_question, table_dict, table_samples, llm):
    """
    Generate an SQL query from a natural language question considering multiple tables.
    
    Args:
        user_question: The natural language question from the user
        table_dict: Dictionary mapping sheet names to table names
        table_samples: Dictionary mapping table names to sample dataframes
        llm: The language model instance to use
    """
    try:
        # Build schema information for all tables
        schema_descriptions = {}
        for sheet_name, table_name in table_dict.items():
            if table_name in table_samples:
                df_head = table_samples[table_name]
                # Ensure column names are preprocessed
                df_head.columns = [preprocess_column_name(col) for col in df_head.columns]
                schema_descriptions[table_name] = get_schema_description(df_head)
        
        # Get relationships information from table names and columns
        # table_relations = []
        # schema_info = get_database_schema()
        
        # Build combined schema information
        all_tables_info = []
        for table_name, sample_df in table_samples.items():
            column_names = ", ".join([f"`{col}`" for col in sample_df.columns])
            all_tables_info.append(f"TABLE: `{table_name}`\nCOLUMNS: {column_names}\n")
            all_tables_info.append(f"SCHEMA:\n{schema_descriptions.get(table_name, 'No schema information')}\n")
            all_tables_info.append(f"SAMPLE DATA:\n{sample_df.head(5).to_markdown(index=False)}\n")
            all_tables_info.append("-" * 40 + "\n")
        
        all_tables_str = "\n".join(all_tables_info)
        tables_list = ", ".join([f"`{t}`" for t in table_dict.values()])
        
        # Build the prompt
        prompt = f"""
You are a SQL expert tasked with translating natural language questions into accurate SQLite queries.

USER QUESTION:
"{user_question}"

AVAILABLE TABLES:
{tables_list}

TABLES INFORMATION:
{all_tables_str}

{SQL_CAPABILITIES}

INSTRUCTIONS:
1. Generate ONLY a valid SQLite SQL query that answers the user's question
2. Determine which table(s) are needed to answer the question
3. Use appropriate JOIN clauses if needed to combine data from multiple tables
4. Use GROUP BY with aggregate functions when appropriate
5. Use ORDER BY when the question asks about "top", "highest", or "lowest"
6. Include LIMIT only when the question asks about a specific number of results
7. Format dates appropriately if date-related operations are needed
8. Do not include any explanations, only output the SQL query

SQL QUERY:
""".strip()
        
        # Generate SQL query using LLM
        response = llm.invoke(prompt).content.strip()
        
        # Log and clean up the response
        logger.info(f"Generated SQL query: {response}")
        
        # Clean up the SQL query if needed (remove any markdown formatting artifacts)
        if response.startswith("```sql"):
            response = response.split("```sql")[1]
        if "```" in response:
            response = response.split("```")[0]
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating SQL query: {str(e)}")
        # Choose the first table as a fallback
        fallback_table = next(iter(table_dict.values())) if table_dict else "unknown_table"
        return f"SELECT * FROM {fallback_table} LIMIT 5 -- Error generating query: {str(e)}"

def fetch_sql_result(db_path, query):
    """
    Execute an SQL query against the SQLite database and return the results.
    """
    start_time = time.time()
    try:
        logger.info(f"Executing SQL query: {query}")
        conn = sqlite3.connect(db_path)
        
        # Add timeout to prevent long-running queries
        conn.execute("PRAGMA timeout = 5000")  # 5 second timeout
        
        # Execute the query and get results
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        # Log query execution time
        logger.info(f"Query executed in {time.time() - start_time:.2f} seconds, returned {len(result)} rows")
        return result
        
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        # Create an error dataframe with the error message
        error_df = pd.DataFrame({
            "Error": [f"SQL Error: {str(e)}"],
            "Query": [query]
        })
        return error_df

def generate_insight_from_result(result_df, user_question, llm):
    """
    Generate natural language insights from SQL query results.
    """
    try:
        # Prepare the data for the prompt
        if len(result_df) > 0:
            # Convert DataFrame to markdown table
            result_markdown = result_df.head(100).to_markdown(index=False)
            
            # Get result statistics - FIX FOR INT64 JSON SERIALIZATION ERROR
            result_stats = {}
            for col in result_df.select_dtypes(include=['number']).columns:
                # Convert NumPy types to Python native types
                result_stats[col] = {
                    "min": float(result_df[col].min()),
                    "max": float(result_df[col].max()),
                    "mean": float(result_df[col].mean()),
                    "median": float(result_df[col].median())
                }
            
            # Use the NumPy encoder for JSON serialization
            stats_str = json.dumps(result_stats, indent=2, cls=NumpyEncoder) if result_stats else "No numeric columns available"
            
            row_count = len(result_df)
            column_count = len(result_df.columns)
        else:
            result_markdown = "No results returned from the query."
            stats_str = "No results available"
            row_count = 0
            column_count = 0
        
        # Build the prompt
        prompt = f"""
You are a data analyst explaining query results to a business user.

USER QUESTION:
"{user_question}"

QUERY RESULT STATISTICS:
Total rows: {row_count}
Total columns: {column_count}

NUMERIC COLUMN STATISTICS (if available):
{stats_str}

RESULT DATA:
{result_markdown}

INSTRUCTIONS:
1. Provide a clear, concise interpretation of the results
2. Highlight key insights, trends, or anomalies
3. Use business-friendly language, not technical jargon
4. If the result is empty or shows an error, explain why the question might not be answerable with the available data
5. Include specific numbers from the results to support your insights
6. Keep your response under 200 words
7. Keep format of generated response consistant.

INSIGHT:
""".strip()
        
        # Generate insight using LLM
        insight = llm.invoke(prompt).content.strip()
        logger.info(f"Generated insight from results with {row_count} rows")
        return insight
        
    except Exception as e:
        logger.error(f"Error generating insight: {str(e)}")
        return f"Sorry, I couldn't generate an insight from the results. Error: {str(e)}"

def generate_general_data_insight(user_question, table_samples, llm):
    """
    Generate insights from dataframe samples when SQL query isn't appropriate.
    Uses information from multiple tables if available.
    """
    try:
        # Generate combined information from all table samples
        table_info = []
        
        for table_name, df_head in table_samples.items():
            # Generate schema information
            schema_description = get_schema_description(df_head)
            
            # Add table information
            table_info.append(f"TABLE: {table_name}")
            table_info.append(f"ROWS: {len(df_head)}")
            table_info.append(f"COLUMNS: {len(df_head.columns)}")
            table_info.append(f"SCHEMA:\n{schema_description}")
            
            # Generate sample data
            table_info.append(f"SAMPLE DATA:\n{df_head.head(5).to_markdown(index=False)}")
            table_info.append("-" * 40)
        
        # Join all table information
        all_tables_str = "\n".join(table_info)
        
        # Build the prompt
        prompt = f"""
You are a data analyst providing insights based on samples from multiple tables.

USER QUESTION:
"{user_question}"

DATASET INFORMATION:
{all_tables_str}

INSTRUCTIONS:
1. Analyze the data samples carefully across all tables
2. Consider how the tables might be related based on column names
3. Answer the user's question as accurately as possible based on the available information
4. If the question applies to specific tables, focus on those tables
5. If the question cannot be answered with the provided samples, explain why
6. Be honest about limitations due to the sample size
7. Keep your response under 200 words and focus on the most important insights

INSIGHT:
""".strip()
        
        # Generate insight
        insight = llm.invoke(prompt).content.strip()
        logger.info("Generated general data insight from samples of multiple tables")
        return insight
        
    except Exception as e:
        logger.error(f"Error generating general insight: {str(e)}")
        return f"I couldn't generate an insight from the data samples. Error: {str(e)}"

def is_query_relevant(user_question, sql_query, llm):
    """
    Evaluate if an SQL query is relevant to the user's question.
    Returns a confidence score from 0-100.
    """
    try:
        # Check if the query is syntactically valid
        if not sql_query or len(sql_query) < 10 or "select" not in sql_query.lower():
            logger.warning("Invalid or empty SQL query detected")
            return 0
        
        # Try to execute the query to get results for evaluation
        query_result_df = fetch_sql_result("data_store.db", sql_query)
        
        # Prepare result sample for the prompt
        if query_result_df.empty:
            result_sample = "No results returned from the query."
        else:
            result_sample = query_result_df.head(10).to_markdown(index=False)
        
        # Build the prompt for confidence scoring
        prompt = f"""
You are evaluating the relevance of an SQL query to a user question.

USER QUESTION:
"{user_question}"

SQL QUERY:
{sql_query}

QUERY RESULTS:
{result_sample}

TASK:
Analyze how well the SQL query addresses the user's question and assign a confidence score from 0 to 100.

SCORING CRITERIA:
- 80-100: Perfect match - query directly answers the question with appropriate columns and logic
- 60-79: Good match - query addresses the core of the question but might miss some nuance
- 40-59: Partial match - query is somewhat related but misses important aspects
- 20-39: Poor match - query returns data but doesn't properly address the question
- 0-19: Irrelevant - query doesn't relate to the question or is fundamentally flawed

OUTPUT FORMAT:
Your response should be ONLY a numeric score between 0 and 100.

CONFIDENCE SCORE:
""".strip()
        
        # Generate confidence score
        response = llm.invoke(prompt).content.strip()
        
        # Parse the response to get the numeric score
        try:
            # Extract just the number from the response
            import re
            score_match = re.search(r'\b([0-9]{1,3})\b', response)
            if score_match:
                confidence_score = int(score_match.group(1))
                # Ensure the score is within valid range
                confidence_score = max(0, min(100, confidence_score))
            else:
                logger.warning(f"Could not parse confidence score from: {response}")
                confidence_score = 50  # Default to medium confidence if parsing fails
        except Exception as e:
            logger.error(f"Error parsing confidence score: {str(e)}")
            confidence_score = 50  # Default to medium confidence
        
        logger.info(f"Query relevance confidence score: {confidence_score}")
        return confidence_score
        
    except Exception as e:
        logger.error(f"Error in query relevance check: {str(e)}")
        return 30  # Default to low-medium confidence on errors
