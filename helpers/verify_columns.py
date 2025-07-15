import streamlit as st
from pandas import DataFrame
from const.features import expected_columns

def verify(df: DataFrame):
    # Identify missing and extra columns
    df_columns = set(df.columns)
    expected_set = set(expected_columns)

    missing = list(expected_set - df_columns)
    extra = list(df_columns - expected_set)

    if missing:
        st.error("Dataset is missing the following required columns:")
        st.write(missing)
        return  None # Stop further processing if critical columns are missing

    # Drop extra columns if present
    if extra:
        df = df[expected_columns]
        st.warning(f"Extra columns dropped: {extra}")

    # Save cleaned DataFrame in session state
    st.session_state.df = df
    st.success("Dataset is valid and ready for analysis!")

    return df
