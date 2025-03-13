import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
from pandasai import SmartDataframe
from ydata_profiling import ProfileReport
import tempfile
import os

# Load the Ollama model
llm = Ollama(model="mistral")

st.title("DataQuest AI: Conversational AI for Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Data Cleaning
    data.drop_duplicates(inplace=True)  # Remove duplicate rows
    data.fillna(method='ffill', inplace=True)  # Fill missing values with forward fill
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()  # Remove unnecessary spaces

    # Show cleaned data preview
    st.subheader("Cleaned Data (First 3 Rows)")
    st.write(data.head(3))

    # Convert to PandasAI SmartDataframe
    df = SmartDataframe(data, config={"llm": llm})

    # Download cleaned data
    cleaned_csv = "cleaned_data.csv"
    data.to_csv(cleaned_csv, index=False)
    st.download_button("Download Cleaned CSV", data=open(cleaned_csv, "rb"), file_name="cleaned_data.csv")

    # User prompt for analysis
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))
        else:
            st.warning("Please enter a prompt!")

    # Generate and download report button
    if st.button("Download Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(data, explorative=True)

            # Save report as HTML
            tmp_html_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
            profile.to_file(tmp_html_path)

            # Provide download link
            with open(tmp_html_path, "rb") as f:
                st.download_button("Download HTML Report", data=f, file_name="data_report.html")

            # Cleanup temp file
            os.remove(tmp_html_path)
