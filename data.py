import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
from pandasai import SmartDataframe
from ydata_profiling import ProfileReport
import sqlite3
import tempfile
import pdfkit
import os

# Configure pdfkit (ensure wkhtmltopdf is installed in this path)
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

# Load the Ollama model
llm = Ollama(model="mistral")

# Streamlit settings
st.set_page_config(page_title="DataQuest AI", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; font-family: sans-serif; color: black; letter-spacing: 1px;'>
                       A Conversational AI for
                          Data Analysis
    </h1>
    """,
    unsafe_allow_html=True
)



# Sidebar navigation


# Sidebar navigation (Apply stylish logo-like font to this)
st.sidebar.markdown(
   """
    <h4 style='text-align: center; font-family: sans-serif; color: #e91e63; letter-spacing: 3px; font-size: 20px;'>
        🎯 DATAQUEST AI
    </h4>
    <p style='text-align: center; font-size: 12px; color: #333; letter-spacing: 2px;'>
        TURNING DATA INTO INSIGHTS
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar radio buttons
section = st.sidebar.radio(
    "",
    [
        "🧠 Smart Q&A",
        "🔍 Preview Cleaned Dataset",
        "📥 Export Cleaned Data",
        "📊 Generate Insights & Report"
    ]
)
# Session state to store data
if "data" not in st.session_state:
    st.session_state.data = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "db_path" not in st.session_state:
    st.session_state.db_path = None

# Function to load SQLite
def load_sqlite(file):
    conn = sqlite3.connect(file)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    return conn, tables['name'].tolist()

# Section 1: Smart Q&A + Upload
if section == "🧠 Smart Q&A":
    #st.header("📂 Upload & 💬 Ask with Data Smarts!")
    uploaded_file = st.file_uploader("🔼 Upload a CSV or SQLite file to begin analysis", type=["csv", "db", "sqlite"])

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]
        st.session_state.file_type = file_type

        if file_type == "csv":
            st.session_state.data = pd.read_csv(uploaded_file)

        elif file_type in ["db", "sqlite"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                db_path = tmp_file.name
            st.session_state.db_path = db_path
            conn, tables = load_sqlite(db_path)
            selected_table = st.selectbox("📋 Choose a table from database", tables)
            if selected_table:
                st.session_state.data = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)

        if st.session_state.data is not None:
            # Clean data
            data = st.session_state.data
            data.drop_duplicates(inplace=True)
            data.fillna(method='ffill', inplace=True)
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].str.strip()
            st.session_state.data = data

            # Smart Q&A with AI
            df = SmartDataframe(data, config={"llm": llm})
            st.subheader("**Ask your questions!**")
            prompt = st.text_area("🧾 Enter your question here:")
            if st.button("Get Answer"):
                if prompt:
                    with st.spinner("🤖 Thinking..."):
                        st.write(df.chat(prompt))
                else:
                    st.warning("⚠️ Please type a question to proceed.")

# Section 2: Data Preview
elif section == "🔍 Preview Cleaned Dataset":
    st.header("📊 Preview of Cleaned Dataset")
    if st.session_state.data is not None:
        st.dataframe(st.session_state.data, use_container_width=True)
    else:
        st.warning("⚠️ No data found. Upload and clean your data in the Q&A section first.")

# Section 3: Download Cleaned Data
elif section == "📥 Export Cleaned Data":
    st.header("💾 Export Cleaned Dataset")
    if st.session_state.data is not None:
        download_type = st.radio("📂 Choose a format to download:", ["CSV", "SQLite"], horizontal=True)

        if download_type == "CSV":
            cleaned_csv = st.session_state.data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download as CSV", data=cleaned_csv, file_name="cleaned_data.csv", mime='text/csv')

        elif download_type == "SQLite":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_sqlite:
                conn = sqlite3.connect(tmp_sqlite.name)
                st.session_state.data.to_sql("cleaned_table", conn, if_exists='replace', index=False)
                conn.close()
                with open(tmp_sqlite.name, "rb") as f:
                    st.download_button("⬇️ Download as SQLite", data=f, file_name="cleaned_data.db", mime='application/octet-stream')
    else:
        st.warning("⚠️ Please process and clean data in the Q&A section first.")

# Section 4: Data Analysis Report
elif section == "📊 Generate Insights & Report":
    st.header("📄 Explore In-Depth Data Insights")
    if st.session_state.data is not None:
        if st.button("📈 Create Analytical Report"):
            with st.spinner("🛠️ Generating your report..."):
                profile = ProfileReport(st.session_state.data, explorative=True)
                tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
                profile.to_file(tmp_html)

                # Preview HTML
                with open(tmp_html, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)

                # Convert to PDF and enable download
                pdf_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                pdfkit.from_file(tmp_html, pdf_file_path, configuration=config)
                with open(pdf_file_path, "rb") as f:
                    st.download_button("📥 Download Full Report (PDF)", data=f, file_name="data_analysis_report.pdf")

                os.remove(tmp_html)
                os.remove(pdf_file_path)
    else:
        st.warning("⚠️ No data available. Please upload and clean data first.")
