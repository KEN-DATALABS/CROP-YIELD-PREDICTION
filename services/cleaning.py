import streamlit as st
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import List
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class Cleaning:
    def __init__(self, df:DataFrame, issues:List = []):
        self.df = df.copy()
        self.issues = issues
    
    def missing_values(self):
        # Count missing values per column
        missing_counts = self.df.isna().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            self.issues.append(f"Missing values detected: {int(total_missing)} total")

            # Drop columns with more than 30% missing data
            col_thresh = int(len(self.df) * 0.7)  # keep if ≥ 70% non-null
            self.df.dropna(axis=1, thresh=col_thresh, inplace=True)

            # Impute numeric columns with mean
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                num_imputer = SimpleImputer(strategy='mean')
                self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])

            # Impute categorical columns with mode
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
            
    def rename(self):
        # Correct typo: 'Item' not 'Items'
        if 'Item' in self.df.columns:
            self.df.rename(columns={'Item': 'Crop'}, inplace=True)
        # Normalize column names
        self.df.columns = (self.df.columns.str.strip().str.replace(" ", "_", regex=False).str.capitalize())
        self.issues.append("Column names normalized and 'Item' renamed to 'Crop'")


    def handle_duplicates(self):
        #Check for Duplicate Rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.issues.append(f"Duplicate rows detected: {duplicates}")
        # Remove duplicates
        self.df = self.df.drop_duplicates()

    def handle_outliers(self, threshold:int):
        #Outlier Detection using IQR method
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        outlier_cols = []  # Keep track of columns that contain outliers
        initial_shape = self.df.shape

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            # Identify outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                self.issues.append(f"Potential outliers detected in '{col}'")
                outlier_cols.append(col)

        # Visualize detected outliers using boxplots
        if outlier_cols:
            st.subheader("Outlier Visualization")
            for col in outlier_cols:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(y=self.df[col], color='skyblue', ax=ax)
                ax.set_title(f'Boxplot of {col} (with outliers)', fontsize=14)
                st.pyplot(fig)

        # Remove outliers for the column
        self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        # Report
        if outlier_cols:
            st.warning(f"Outliers removed from the following columns: {', '.join(outlier_cols)}")
            st.write(f"Rows before outlier removal: {initial_shape[0]}, Rows after outlier removal {self.df.shape[0]}")


    def fix_column_types(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Strip whitespace and convert to string
                cleaned = self.df[col].astype(str).str.strip()
                # Try float conversion if it may contain decimal numbers
                try:
                    self.df[col] = pd.to_numeric(cleaned)
                except:
                    self.df[col] = cleaned  # Keep as string

    def summary(self):
        if self.issues:
            st.warning("The following data issues were found:")
            for issue in self.issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No major data issues detected!")

    def handle_cleandata(self):
        self.rename()
        self.missing_values()
        self.handle_duplicates()
        self.handle_outliers(1.5)
        self.fix_column_types()
    
