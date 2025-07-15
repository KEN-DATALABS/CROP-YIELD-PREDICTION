import streamlit as st
import pandas as pd

def upload_dataset():
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if not uploaded_file:
        return None

    try:
        read_func = pd.read_csv if uploaded_file.name.endswith(".csv") else pd.read_excel
        df = read_func(uploaded_file)
        st.success("File uploaded and read successfully.")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    
def load_dataset():
    try:
        df = pd.read_csv('dataset/yield_df.csv')  # Ensure this file is available in the working folder
        st.success("Default dataset loaded successfully.")
        return df
    except FileNotFoundError:
        st.error("Default dataset not found in the directory.")
        return None