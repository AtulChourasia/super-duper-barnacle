import streamlit as st
import pandas as pd
import os
import time
from openpyxl import load_workbook
from backend import (
    setup_azure_openai, 
    preprocess_and_store_in_sqlite, 
    generate_sql_prompt,
    fetch_sql_result,
    generate_insight_from_result,
    is_query_relevant,
    generate_general_data_insight,
    get_database_schema,
    is_complex_sheet,
)
from lib.Class_ExcelFlattener_V1 import ExcelFlattener
from IPython.display import Markdown

# Set page config and title
st.set_page_config(
    page_title="Excel → SQL Insights", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'df_dict' not in st.session_state:
    st.session_state.df_dict = {}
if 'table_dict' not in st.session_state:
    st.session_state.table_dict = {}
if 'table_samples' not in st.session_state:
    st.session_state.table_samples = {}
if 'active_sheet' not in st.session_state:
    st.session_state.active_sheet = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'schema_info' not in st.session_state:
    st.session_state.schema_info = {}

# Header
st.markdown("<h1 class='main-header'>📊Insights Generator</h1>", unsafe_allow_html=True)

# Load custom CSS file if it exists
css_file_path = "style.css"
if os.path.exists(css_file_path):
    try:
        with open(css_file_path, "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")
else:
    st.error("CSS file not found")

# Sidebar Layout
st.sidebar.markdown("<h2>🛠️ Controls</h2>", unsafe_allow_html=True)

# File uploader in sidebar - modified to accept multiple files
uploaded_files = st.sidebar.file_uploader("📤 Upload Excel or CSV Files", 
                                         type=["xlsx", "xls", "csv"],
                                         accept_multiple_files=True,
                                         help="Upload one or more Excel/CSV files to analyze")

# Set the selected example as the question
user_question = st.sidebar.text_area("Your question:", 
                                      placeholder="Type your question here...", 
                                      height=100)

submit_button = st.sidebar.button(" Generate Insight", use_container_width=True)

# Show previous queries in sidebar
if st.session_state.history:
    st.sidebar.markdown("### 📚 Previous Queries")
    for i, (q, a) in enumerate(st.session_state.history[-5:]):  # Show last 5 queries
        with st.sidebar.expander(f"Q: {q[:30]}{'...' if len(q) > 30 else ''}"):
            st.write(f"**Answer:** {a[:100]}{'...' if len(a) > 100 else ''}")

# Main instructions
with st.expander("ℹ️ How to Use", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1. Upload Your Data 📤
        Upload Excel files with multiple sheets to instantly process and prepare your data for SQL queries.
        """)
    
    with col2:
        st.markdown("""
        ### 2. Ask Questions ❓
        Ask questions across tables like "Which products have the highest sales?" and get SQL-driven insights.
        """)
    
    with col3:
        st.markdown("""
        ### 3. Explore & Analyze 📊
        Navigate between tables, view relationships, and discover insights across your entire dataset.
        """)

# Initialize multi-file tracking in session state if not exists
if 'files_dict' not in st.session_state:
    st.session_state.files_dict = {}
    
# Initialize active file tracking
if 'active_file' not in st.session_state:
    st.session_state.active_file = None

# Main Content Area
if uploaded_files:
    # Process any new files
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Only process each file if it's newly uploaded or different from before
        if uploaded_file.name not in st.session_state.files_dict:
            with st.spinner(f" Reading {file_type.upper()} file '{uploaded_file.name}'..."):
                progress_bar = st.progress(0)
                try:
                    if file_type == "csv":
                        # For CSV, create a single dataframe and put it in a dictionary
                        df = pd.read_csv(uploaded_file)
                        # Store in files_dict with filename as key and a dict containing file info
                        st.session_state.files_dict[uploaded_file.name] = {
                            "type": "csv",
                            "df_dict": {"main_data": df},
                            "active_sheet": "main_data"
                        }
                    elif file_type in ["xlsx", "xls"]:
                        # Load workbook with openpyxl to inspect merged cells
                        wb = load_workbook(uploaded_file, data_only=True)
                        excel_file = pd.ExcelFile(uploaded_file)
                        df_dict = {}

                        for sheet_name in excel_file.sheet_names:
                            ws = wb[sheet_name]
                            raw_df = pd.DataFrame(ws.values)

                            if is_complex_sheet(raw_df, ws):
                                df = ExcelFlattener(uploaded_file, sheet_name=sheet_name).flatten(method='wide')
                            else:
                                df = pd.read_excel(excel_file, sheet_name=sheet_name)

                            df_dict[sheet_name] = df

                        # Store in files_dict with filename as key and a dict containing file info
                        st.session_state.files_dict[uploaded_file.name] = {
                            "type": file_type,
                            "df_dict": df_dict,
                            "active_sheet": excel_file.sheet_names[0] if excel_file.sheet_names else None
                        }
                    else:
                        st.error(f"❌ Unsupported file type for '{uploaded_file.name}'. Please upload CSV or Excel files.")
                        continue
                        
                    # Simulate progress
                    for i in range(101):
                        progress_bar.progress(i)
                        time.sleep(0.01)
                        
                    progress_bar.empty()
                    
                    # Show success message with total sheets and rows
                    file_info = st.session_state.files_dict[uploaded_file.name]
                    total_sheets = len(file_info["df_dict"])
                    total_rows = sum(len(df) for df in file_info["df_dict"].values())
                    st.success(f"✅ {file_type.upper()} file '{uploaded_file.name}' loaded successfully with {total_sheets} {'sheet' if total_sheets == 1 else 'sheets'} and {total_rows:,} total rows!")
                    
                    # If this is the first file or no active file, set it as active
                    if st.session_state.active_file is None:
                        st.session_state.active_file = uploaded_file.name
                        
                except Exception as e:
                    st.error(f"🚫 Error reading file '{uploaded_file.name}': {e}")
                    continue
    
    # Only proceed if at least one file is loaded
    if st.session_state.files_dict:
        # Add file selector at the top
        st.markdown("<h3>📂 Available Files</h3>", unsafe_allow_html=True)
        available_files = list(st.session_state.files_dict.keys())
        
        # Create file selection tabs
        file_tabs_html = "<div style='display: flex; overflow-x: auto; padding-bottom: 10px;'>"
        for file_name in available_files:
            is_active = file_name == st.session_state.active_file
            tab_class = "sheet-tab-active" if is_active else "sheet-tab-inactive"
            file_tabs_html += f"<div class='sheet-tab {tab_class}' onclick=\"window.parent.postMessage({{type: 'streamlit:setComponentValue', value: '{file_name}', dataType: 'str', componentIndex: 'file_selector'}}, '*')\">{{file_name}}</div>"
        file_tabs_html += "</div>"
        
        st.markdown(file_tabs_html, unsafe_allow_html=True)
        
        # Add a hidden selectbox for file selection that will be updated by the custom tabs
        file_selector = st.empty()
        selected_file = file_selector.selectbox("Select file", options=available_files, 
                                              index=available_files.index(st.session_state.active_file) if st.session_state.active_file in available_files else 0, 
                                              key="file_selector", label_visibility="collapsed")
        
        # Update active file if changed
        if selected_file != st.session_state.active_file:
            st.session_state.active_file = selected_file
            st.rerun()
        
        # Work with the active file's data
        active_file = st.session_state.files_dict[st.session_state.active_file]
        file_type = active_file["type"]
        
        # Display sheet tabs for Excel files with multiple sheets
        if file_type in ["xlsx", "xls"] and len(active_file["df_dict"]) > 1:
            st.markdown("<h3>📑 Available Sheets</h3>", unsafe_allow_html=True)
            
            # Create horizontal sheet tabs
            sheet_tabs_html = "<div style='display: flex; overflow-x: auto; padding-bottom: 10px;'>"
            for sheet_name in active_file["df_dict"].keys():
                is_active = sheet_name == active_file["active_sheet"]
                tab_class = "sheet-tab-active" if is_active else "sheet-tab-inactive"
                sheet_tabs_html += f"<div class='sheet-tab {tab_class}' onclick=\"window.parent.postMessage({{type: 'streamlit:setComponentValue', value: '{sheet_name}', dataType: 'str', componentIndex: 'sheet_selector'}}, '*')\">{{sheet_name}}</div>"
            sheet_tabs_html += "</div>"
            
            st.markdown(sheet_tabs_html, unsafe_allow_html=True)
            
            # Add a hidden selectbox for sheet selection that will be updated by the custom tabs
            sheet_selector = st.empty()
            selected_sheet = sheet_selector.selectbox("Select sheet", 
                                                   options=list(active_file["df_dict"].keys()), 
                                                   index=list(active_file["df_dict"].keys()).index(active_file["active_sheet"]) if active_file["active_sheet"] in active_file["df_dict"] else 0, 
                                                   key="sheet_selector", 
                                                   label_visibility="collapsed")
            
            # Update active sheet if changed
            if selected_sheet != active_file["active_sheet"]:
                active_file["active_sheet"] = selected_sheet
                st.rerun()

        # Get the active dataframe for the active file
        if file_type == "csv":
            df = active_file["df_dict"].get("main_data")
            active_sheet = "main_data"
        else:
            active_sheet = active_file["active_sheet"] or list(active_file["df_dict"].keys())[0]
            df = active_file["df_dict"].get(active_sheet)

        # Display file and sheet metadata
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Sheet", active_sheet)
        col2.metric("Rows", f"{len(df):,}")
        col3.metric("Columns", len(df.columns))
        col4.metric("Total Tables", len(active_file["df_dict"]))

    # Only continue with data display if we have files loaded
    if st.session_state.files_dict and st.session_state.active_file in st.session_state.files_dict:
        # Tabs for different views
        tab1, tab2, tab3, tab4= st.tabs(["📊 Raw Data", "🔍 Columns", "📈 Stats", "📋 Data Types"])

        # --- Tab 1: Raw Data ---
        with tab1:
            st.markdown("<h3>🔎 Data Preview</h3>", unsafe_allow_html=True)
            st.caption("Select number of rows to view 👇")
            num_rows = st.slider("", min_value=5, max_value=min(100, len(df)), value=10, step=5, help="Slide to choose how many rows to display")
            st.dataframe(df.head(num_rows), use_container_width=True, height=350)

        # --- Tab 2: Column Selector ---
        with tab2:
            st.markdown("<h3>🔧 Column Selection</h3>", unsafe_allow_html=True)
            st.caption("Choose which columns to display")
            
            # Search box for columns
            search_col = st.text_input("🔍 Search columns", "")
            filtered_cols = [col for col in df.columns if search_col.lower() in str(col).lower()] if search_col else df.columns.tolist()
            
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=filtered_cols,
                default=filtered_cols[:5] if len(filtered_cols) > 5 else filtered_cols,
            )
            
            if selected_columns:
                st.dataframe(df[selected_columns], use_container_width=True, height=400)
            else:
                st.info("👆 Select at least one column to display data")

        # --- Tab 3: Stats ---
        with tab3:
            st.markdown("<h3>📊 Data Statistics</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔢 Numeric Summary**")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe().transpose().style.format("{:.2f}"), 
                                use_container_width=True, height=300)
                else:
                    st.info("No numeric columns found in the dataset")
                    
            with col2:
                st.markdown("**🔤 Categorical Summary**")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    st.dataframe(df[categorical_cols].describe().transpose(), 
                                use_container_width=True, height=300)
                else:
                    st.info("No categorical columns found in the dataset")

        # --- Tab 4: Data Types ---
        with tab4:
            st.markdown("<h3>📋 Column Data Types</h3>", unsafe_allow_html=True)
            
            # Create a dataframe with column info
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isna().sum().values,
                'Null %': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%',
                'Unique Values': [df[col].nunique() for col in df.columns],
            })
            
            # Display the column info with sorting
            st.dataframe(dtypes_df, use_container_width=True, height=400)
    else:
        # Display a message if no files are loaded
        st.info("📤 Please upload one or more Excel or CSV files to begin analysis.")

    # LLM setup - only do once and store in session state
    if st.session_state.llm is None:
        with st.spinner("Initializing AI models..."):
            progress_bar = st.progress(0)
            try:
                st.session_state.llm = setup_azure_openai()
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(0.01)
                progress_bar.empty()
            except Exception as e:
                st.error(f"Error initializing AI: {e}")
                st.stop()

    # Preprocess and store in SQLite - collect all tables from all files
    if st.session_state.files_dict and (not st.session_state.table_dict or submit_button):
        with st.spinner("Preprocessing and storing tables from all files..."):
            progress_bar = st.progress(0)
            try:
                # Collect all dataframes from all files into a single dictionary
                all_dfs = {}
                
                for file_name, file_info in st.session_state.files_dict.items():
                    # Add a prefix with file name to avoid table name collisions
                    file_prefix = file_name.split('.')[0].lower().replace(' ', '_')
                    
                    for sheet_name, df in file_info["df_dict"].items():
                        # Create a unique key combining file and sheet names
                        combined_name = f"{file_prefix}_{sheet_name}"
                        all_dfs[combined_name] = df
                
                # Process all tables
                st.session_state.table_dict = preprocess_and_store_in_sqlite(all_dfs)
                
                # Store samples for each table for use in prompts
                st.session_state.table_samples = {
                    table_name: all_dfs[sheet_name].head(20) 
                    for sheet_name, table_name in st.session_state.table_dict.items()
                }
                
                # Get schema information for potential relationship detection
                st.session_state.schema_info = get_database_schema()
                
                # Update progress
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(0.01)
                progress_bar.empty()
                
                # Success message with table count
                st.success(f"✅ Successfully processed and stored {len(st.session_state.table_dict)} tables in the database!")
            except Exception as e:
                st.error(f"Error processing data: {e}")
                st.stop()
    
    # Process user question
    if submit_button and user_question:
        st.markdown("---")
        st.markdown(f"<h4>🔍 Query: \"{user_question}\"</h4>", unsafe_allow_html=True)
        
        with st.spinner("Generating insights..."):
            progress_bar = st.progress(0)
            
            # Generate SQL query
            sql_query = generate_sql_prompt(
                user_question, 
                st.session_state.table_dict, 
                st.session_state.table_samples,
                st.session_state.llm
            )
            progress_bar.progress(30)
            
            # Check query relevance
            confidence_score = is_query_relevant(user_question, sql_query, st.session_state.llm)
            progress_bar.progress(50)
            
            # Display confidence meter
            st.markdown("<h4>Query Confidence</h4>", unsafe_allow_html=True)
            confidence_color = "red" if confidence_score < 50 else "orange" if confidence_score < 80 else "green"
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="background-color: #f0f0f0; border-radius: 10px; height: 10px; width: 100%;">
                    <div style="background-color: {confidence_color}; width: {confidence_score}%; height: 10px; border-radius: 10px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Low</span>
                    <span><b>{confidence_score}%</b></span>
                    <span>High</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Process based on confidence score
            if confidence_score >= 60:
                st.success("Using SQL-based approach for accurate results ✅")
                with st.expander("View SQL Query"):
                    st.code(sql_query, language="sql")

                # Execute SQL query
                try:
                    result_df = fetch_sql_result("data_store.db", sql_query)
                    progress_bar.progress(75)
                    
                    # Generate insight
                    answer = generate_insight_from_result(result_df, user_question, st.session_state.llm)
                    progress_bar.progress(100)
                    progress_bar.empty()
                    
                    # Display results
                    st.subheader("📊 Query Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error executing SQL: {e}")
                    # Fall back to general insight generation using samples from all tables
                    combined_samples = pd.DataFrame()
                    for table_name, sample_df in st.session_state.table_samples.items():
                        if len(combined_samples) < 40:  # Limit to 40 rows total
                            combined_samples = pd.concat([combined_samples, sample_df.head(40 // len(st.session_state.table_samples))])
                    
                    answer = generate_general_data_insight(user_question, combined_samples, st.session_state.llm)
            else:
                st.warning("Using general analysis (no valid SQL could be generated) 🧠")
                with st.expander("View Generated SQL (for reference)"):
                    st.code(sql_query, language="sql")
                
                # Generate general insight using samples from all tables
                answer = generate_general_data_insight(
                    user_question, 
                    st.session_state.table_samples, 
                    st.session_state.llm
                )
                progress_bar.progress(100)
                progress_bar.empty()

            # Display insight
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>💡 Insight</h3>", unsafe_allow_html=True)
            st.markdown(f"{answer}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add to history
            st.session_state.history.append((user_question, answer))
            
            # Show insight generation time
            # st.caption(f"Insight generated at {time.strftime('%H:%M:%S')}")
else:
    pass