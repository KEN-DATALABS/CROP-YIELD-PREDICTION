#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set wide layout
st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)
df.rename(columns={'Item':'Crop'}, inplace=True)

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox("Select a Country", sorted(df['Area'].unique()))
selected_crop = st.sidebar.selectbox("Select a Crop", sorted(df['Crop'].unique()))

# Title
st.title("🌾 Crop Yield & Environmental Dashboard")

# Section 1: Pesticide Usage
st.header("1️⃣ Pesticide Usage Overview")

with st.expander("Top Countries by Pesticide Use"):
    pesticide_by_country = df.groupby('Area')['pesticides_tonnes'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=pesticide_by_country.values, y=pesticide_by_country.index, palette="Reds_r", ax=ax)
    ax.set_title("Top 10 Countries by Pesticide Use")
    ax.set_xlabel("Total Pesticides Used (tonnes)")
    ax.set_ylabel("Country")
    st.pyplot(fig)

with st.expander("Global Pesticide Use Over Time"):
    pesticide_by_year = df.groupby('Year')['pesticides_tonnes'].sum()
    fig, ax = plt.subplots()
    sns.lineplot(x=pesticide_by_year.index, y=pesticide_by_year.values, marker='o', ax=ax)
    ax.set_title("Pesticide Use Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pesticide Tonnes")
    st.pyplot(fig)

# Section 2: Yield & Environment
st.header("2️⃣ Yield vs Environment")

with st.expander("Effect of Pesticide on Crop Yield"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pesticides_tonnes', y='hg/ha_yield', hue='Crop', ax=ax)
    ax.set_title("Pesticide vs Crop Yield")
    ax.set_xlabel("Pesticides (tonnes)")
    ax.set_ylabel("Yield (hg/ha)")
    st.pyplot(fig)

with st.expander("Effect of Temperature on Yield"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='avg_temp', y='hg/ha_yield', hue='Crop', ax=ax)
    ax.set_title("Temperature vs Yield")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Yield (hg/ha)")
    st.pyplot(fig)

# Section 3: Rainfall & Temperature
st.header("3️⃣ Rainfall and Temperature")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Temperature by Year")
    avg_temp_year = df.groupby('Year')['avg_temp'].mean()
    fig, ax = plt.subplots()
    sns.lineplot(x=avg_temp_year.index, y=avg_temp_year.values, marker='o', ax=ax)
    ax.set_title("Avg. Temperature Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

with col2:
    st.subheader("Average Rainfall by Year")
    avg_rain_year = df.groupby('Year')['average_rain_fall_mm_per_year'].mean()
    fig, ax = plt.subplots()
    sns.lineplot(x=avg_rain_year.index, y=avg_rain_year.values, marker='o', ax=ax)
    ax.set_title("Avg. Rainfall Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    st.pyplot(fig)

# Section 4: Country-specific Yield Trends
st.header("4️⃣ Country-Specific Yield Trends")

filtered = df[df['Area'] == selected_country]
fig, ax = plt.subplots()
sns.lineplot(data=filtered, x='Year', y='hg/ha_yield', hue='Crop', marker='o', ax=ax)
ax.set_title(f"Crop Yield Over Time in {selected_country}")
ax.set_xlabel("Year")
ax.set_ylabel("Yield (hg/ha)")
st.pyplot(fig)

# Section 5: Correlation Heatmap
st.header("5️⃣ Correlation Heatmap")

st.write("Explore correlations between environmental factors and crop yield.")
corr_df = df[['hg/ha_yield', 'pesticides_tonnes', 'average_rain_fall_mm_per_year', 'avg_temp']].dropna()
fig, ax = plt.subplots()
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

# Section 6: Crop with Most Pesticide
st.header("6️⃣ Crops With Most Pesticide Used")
pesticide_by_crop = df.groupby('Crop')['pesticides_tonnes'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=pesticide_by_crop.values, y=pesticide_by_crop.index, palette="YlOrBr", ax=ax)
ax.set_title("Top 10 Crops by Pesticide Use")
st.pyplot(fig)

# Section 7: Highlight Hottest Country
hottest_country = df.groupby('Area')['avg_temp'].mean().idxmax()
max_temp = df.groupby('Area')['avg_temp'].mean().max()
st.success(f"🔥 The hottest country on average is **{hottest_country}** with **{max_temp:.2f}°C**.")


# In[ ]:




