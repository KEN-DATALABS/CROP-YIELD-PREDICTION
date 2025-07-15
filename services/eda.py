import streamlit as st
from pandas import DataFrame
import pandas as pd
import numpy as np

class EDA:
    def __init__(self, df:DataFrame):
        self.df = df
    
    def preview(self):
        # Data preview (first 5 rows)
        st.subheader("Data Preview")
        st.dataframe(self.df.head())
    
    def types(self):
        # Show data types of all columns
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame(self.df.dtypes, columns=["Data Type"]))

    def overview(self):
        # Basic info: number of rows and columns
        st.subheader("Dataset Overview")
        st.write(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")

    def description(self):
        # Descriptive statistics for all columns 
        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe(include='all'))

    def missing_values(self):
        # Section: Missing Values Summary
        st.subheader("Missing Values")
        # Define common missing value representations not detected by default (e.g., 'NA', '-', '', etc.)
        missing_values = ['NA', 'na', 'n/a', 'N/a', '', 'null', '-']
        # Replace all custom missing value indicators with np.nan
        for column in self.df.columns:
            self.df[column] = self.df[column].replace(missing_values, np.nan)
        # Create a summary DataFrame of missing values
        missing_df = pd.DataFrame(self.df.isna().sum(), columns=['Missing Values'])
        missing_df["Percentage (%)"] = (missing_df['Missing Values'] / len(self.df)) * 100
        missing_df = missing_df[missing_df['Missing Values'] > 0]  # Show only columns with missing data
        # Display the missing values summary table in the Streamlit app
        st.dataframe(missing_df.style.format({"Percentage (%)": "{:.2f}"}))

    def duplicates(self):
        # Count and show number of duplicate rows
        st.subheader("Duplicate Rows")
        st.write(f"Number of duplicate rows: {self.df.duplicated().sum()}")