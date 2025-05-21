# Excel Flattener and SQL Analyzer

A powerful tool for converting complex Excel files into SQL-friendly datasets and generating insights from your data.

## 🚀 Features

### 1. Excel Flattening
- **Intelligent Structure Detection**: Automatically identifies complex Excel sheets that need flattening
- **Hierarchical Data Handling**: Properly handles multi-level headers and nested data structures
- **Merged Cell Resolution**: Reconstructs data from merged cells into a proper tabular format
- **Blank Row/Column Handling**: Intelligently processes spreadsheets with irregular layouts

### 2. Multi-File Support
- **Batch Processing**: Upload and process multiple Excel and CSV files simultaneously
- **File Management**: Easily switch between uploaded files with a user-friendly interface
- **Independent Data Dictionaries**: Each file maintains its own data structure and sheet information

### 3. SQL Optimization
- **SQL-Friendly Column Names**: Automatically converts Excel column names to SQL-compatible format
- **Intelligent Type Detection**: Automatically detects and converts columns to appropriate SQL types
- **Relationship Detection**: Identifies potential foreign key relationships between tables
- **Metadata Enhancement**: Adds source information and derived columns for better analysis

### 4. AI-Powered Insights
- **Natural Language Queries**: Ask questions about your data in plain English
- **Automatic SQL Generation**: Converts your questions into optimized SQL queries
- **Intelligent Result Analysis**: Generates human-readable insights from query results
- **Cross-Table Analytics**: Analyzes relationships and patterns across multiple tables

## 🔍 Workflow

```
Upload Excel/CSV Files → Automatic Flattening → SQL Database Creation → Query & Analysis → Insights
```

1. **Upload Phase**: 
   - Upload one or more Excel/CSV files through the user interface
   - System automatically detects file type and structure

2. **Processing Phase**:
   - Complex sheets are automatically identified
   - Hierarchical data is flattened into proper tabular format
   - Column names are standardized for SQL compatibility
   - Data types are inferred and optimized

3. **Database Creation**:
   - Processed data is stored in an SQLite database
   - Tables are named based on sheet names
   - Potential relationships between tables are detected

4. **Query Phase**:
   - Ask questions in natural language
   - System generates optimized SQL queries
   - Results are displayed in a readable format

5. **Insight Generation**:
   - System analyzes query results
   - Key trends and patterns are identified
   - Insights are presented in business-friendly language

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/AtulChourasia/super-duper-barnacle.git

# Navigate to the project directory
cd super-duper-barnacle

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run frontend.py
```

## 🔧 Configuration

1. Azure OpenAI services configuration is required in `.streamlit/secrets.toml`:
   ```toml
   TENANT = "your-tenant-id"
   CLIENT_ID = "your-client-id"
   CLIENT_SECRET = "your-client-secret"
   VAULT_URL = "your-vault-url"
   azure_endpoint = "your-azure-endpoint"
   ```

## 📋 Usage Guide

### Basic Usage

1. **Start the application**: Run `streamlit run frontend.py`
2. **Upload files**: Use the file uploader to upload one or more Excel/CSV files
3. **Select a file**: If multiple files are uploaded, select the one you want to work with
4. **Choose a sheet**: Select a sheet from the dropdown menu
5. **Ask questions**: Type your questions in natural language in the input box
6. **View insights**: Explore the generated insights and SQL query results

### Advanced Features

#### Customizing Flattening Process
- **Detection Sensitivity**: Adjust parameters for complex sheet detection
- **Column Naming**: Configure how column names are processed and standardized

#### Working with Complex Data
- **Cross-Sheet Analysis**: Ask questions that span multiple sheets
- **Relationship Exploration**: Investigate connections between different tables

## 🏗️ Architecture

The application consists of three main components:

1. **Frontend (frontend.py)**:
   - Streamlit-based user interface
   - File upload and management
   - Result visualization and display

2. **Backend (backend.py)**:
   - Query processing and SQL generation
   - Database management and storage
   - Insight generation from results

3. **Excel Flattener (lib/Class_ExcelFlattener_V1.py)**:
   - Core flattening algorithms
   - Structure detection and processing
   - Data optimization for SQL

## 🔄 Data Flow

```
User → Frontend → Excel Flattener → SQLite DB → SQL Query → Results → Insights → User
```

## 🛠️ Technical Details

### Key Classes and Methods

#### ExcelFlattener Class
- `flatten(sheet)`: Main method to flatten a complex Excel sheet
- `is_complex_sheet(sheet)`: Detects if a sheet needs flattening
- `_make_sql_friendly(name)`: Converts column names to SQL-compatible format
- `_post_process_for_sql(df)`: Optimizes dataframe for SQL storage
- `detect_potential_relationships(tables)`: Identifies relationships between tables

#### Backend Functions
- `generate_sql_prompt(question, tables)`: Converts natural language to SQL
- `fetch_sql_result(db_path, query)`: Executes SQL queries on the database
- `generate_insight_from_result(results, question)`: Creates insights from query results

## 📊 Examples

### Example 1: Basic Flattening

Original Excel structure with merged cells and multi-level headers:
```
+-------------+--------------------+
|    Region   |       Sales        |
+-------------+----------+---------+
|             | Q1       | Q2      |
+-------------+----------+---------+
| North       | 10,000   | 12,000  |
| South       | 8,500    | 9,200   |
+-------------+----------+---------+
```

Flattened Structure:
```
+-------------+-------------+-------------+
| region      | sales_q1    | sales_q2    |
+-------------+-------------+-------------+
| North       | 10000       | 12000       |
| South       | 8500        | 9200        |
+-------------+-------------+-------------+
```

### Example 2: Natural Language Query

User Question: "What was the total sales in Q1 and Q2 by region?"

Generated SQL:
```sql
SELECT region, 
       sales_q1 as q1_sales, 
       sales_q2 as q2_sales,
       (sales_q1 + sales_q2) as total_sales
FROM sheet1
ORDER BY total_sales DESC
```

Generated Insight:
"The North region led with $22,000 in total sales across Q1 and Q2, with Q2 showing a 20% increase over Q1. The South region had total sales of $17,700, with more modest growth of 8.2% from Q1 to Q2."

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Thanks to all the contributors who have helped improve this tool
- Special thanks to the Streamlit team for making data apps easy to build
- Thanks to Axtria for supporting the development of this tool
