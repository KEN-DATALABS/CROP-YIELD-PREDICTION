import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


from helpers.read_data import upload_dataset, load_dataset
from const.features import expected_columns, sidenav
from const.markdowns import welcome_note
from helpers.verify_columns import verify
from services.eda import EDA
from services.cleaning import Cleaning

from services.visualisation import plot_pie_chart, plot_doughnut_chart, plot_bar_chart, style_axis_ticks, display_and_close

dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Agricultural%20Production.csv"


# Set up Streamlit layout to use full screen width
st.set_page_config(layout="wide")

# App Title
st.title("CROP YIELD PREDICTION APP")

# Introductory markdown explaining the app's purpose
st.markdown(welcome_note)

# Sidebar navigation options for different stages of the app
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Step:", sidenav)

# Initialize Streamlit session state variables to persist data across steps
for key in ['df', 'cleaned_df', 'target', 'model_type', 'model', 'report']:
    if key not in st.session_state:
        st.session_state[key] = None

# Define the required columns expected in the uploaded dataset


# Step 1: Upload or Load Dataset
if options == "Upload Data":
    st.header("Upload or Select Dataset")

    df = None  # Initialize an empty DataFrame variable to avoid reference issues

    # Provide user with two options: upload own dataset or use a default one
    data_source = st.radio("Choose a data source:", ['Upload your dataset', 'Use default dataset'])

    # CASE 1: Uploading a custom dataset
    if data_source == 'Upload your dataset':
        df = upload_dataset()

    # CASE 2: Load default dataset from project directory
    else:
        df = load_dataset()

        # Provide sample structure for download to guide new users
        st.markdown(f"Don't have a dataset? [Download Example CSV]({dataset_url})")
        
        # Provide downloadable blank template CSV with only column headers
        st.download_button(
            label="Download Example Dataset",
            data=pd.DataFrame(columns=expected_columns).to_csv(index=False),
            file_name='example_crop_data.csv',
            mime='text/csv'
        )

    # Proceed only if dataset is loaded and verified
    if df is not None and (df := verify(df)) is not None:
        eda = EDA(df)

        eda.preview()
        eda.overview()
        eda.types()
        eda.description()
        eda.duplicates()
        eda.missing_values()       
            
  #.............................................      

# Step 2: Data Cleaning
elif options == "Data Cleaning":
    st.header("Data Cleaning")

    if "df" in st.session_state and st.session_state.df is not None:
        # Load original dataset
        original_df = st.session_state.df
        cleaned_df = original_df.copy()

        # Perform cleaning
        cleaning = Cleaning(cleaned_df)
        cleaning.handle_cleandata()
        cleaning.summary()

        # Get cleaned result from the class (important!)
        cleaned_df = cleaning.df

        st.success("Data cleaning completed successfully!")

        # Display comparison
        st.subheader("Data Preview (Before vs After Cleaning)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Data (First 5 Rows)**")
            st.dataframe(original_df.head())
        with col2:
            st.markdown("**Cleaned Data (First 5 Rows)**")
            st.dataframe(cleaned_df.head())

        # Cleaning summary
        st.subheader("Cleaning Summary")
        st.write(f"**Original shape:** {original_df.shape}")
        st.write(f"**Cleaned shape:** {cleaned_df.shape}")
        st.write(f"**Rows removed:** {original_df.shape[0] - cleaned_df.shape[0]}")
        st.write(f"**Columns removed:** {original_df.shape[1] - cleaned_df.shape[1]}")

        # Missing values comparison
        orig_missing = original_df.isna().sum().sum()
        clean_missing = cleaned_df.isna().sum().sum()
        st.write(f"**Missing values before:** {orig_missing}")
        st.write(f"**Missing values after:** {clean_missing}")

        # Store cleaned data in session
        st.session_state.cleaned_df = cleaned_df
    else:
        st.warning("Please upload a dataset first in the 'Upload Data' section.")

    
# STEP 3: Data Visualization
elif options == "Visualization":
    st.header("Data Visualization")

    # Ensure dataframe exista in session state
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df

        if 'Crop' in df.columns:
            st.subheader("Crop Distribution")
            # Count the occurences of each crop
            crops=df['Crop'].value_counts()
            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Slightly explode each slice for visual seperation
            explode = [0.05] * len(crops)
            # Plot the pie chart
            ax.pie(crops, labels=crops.index,
                   autopct='%1.1f%%',
                   startangle=140,
                   explode=explode, 
                   textprops={'fontsize': 20})
            # Add title
            ax.set_title('Crop Distribution (Pie)', fontsize=20)
            # Display the chart in streamlit
            


            # Bar Chart
            # Convert to DataFrame for seaborn
            crop_df = crops.reset_index()
            crop_df.columns = ['Crop','Frequency']
            # Create the bar plot
            fig, ax = plt.subplots(figsize=(8, 8))

            plot_bar_chart(data=crop_df, x='Crop', y='Frequency', hue=crops.index, ax=ax)
            # Style the plot
            # style_axis_ticks(ax, axis='x', rotation=80, fontsize=14)
            # style_axis_ticks(ax, axis='y', fontsize=14)
            # ax.set_xlabel('Crop', fontsize=16)
            # ax.set_ylabel('Frequency', fontsize=16)
            # ax.set_title('Crop Frequency Bar Chart', fontsize=17)
            # Layout and display
            plt.tight_layout()
            display_and_close(fig)

        # st.subheader("Area Disribution")
        # # Set figure size: wide vertically for long lists of areas
        # fig, ax = plt.subplots(figsize=(7, 13))
        # # Create a countplot showing the number of occurrences for each 'Area'
        # sns.countplot(data=df,y='Area',palette='husl',hue='Area')
        # # Adjust y-axis tick font size for better readability
        # style_axis_ticks(ax, axis='y', fontsize=8)
        # # Add labels and title for clarity
        # ax.set_xlabel('Frequency', fontsize=12)
        # ax.set_ylabel('Area', fontsize=12)
        # ax.set_title('Distribution of Records by Area', fontsize=14)
        # # Adjust layout to prevent overlapping
        # plt.tight_layout()
        # # Display the plot
        # display_and_close(fig)

        # # Top 20 Areas
        # st.subheader("Top 20 Areas by Frequency")
        # # Get top 20 most frequent Areas
        # top_areas = df['Area'].value_counts().head(20)
        # # Create a new figure
        # fig, ax = plt.subplots(figsize=(8, 8))
        # # Plot a bar chart of the top 20 Areas
        # sns.barplot(x=top_areas.index, y=top_areas.values, palette='coolwarm',hue=top_areas.index)
        # # Rotate x-axis labels and set font size for readability
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=80, fontsize=12)
        # # Set axis labels and title
        # ax.set_xlabel('Area', fontsize=14)
        # ax.set_ylabel('Frequency', fontsize=14)
        # ax.set_title('Top 20 Areas', fontsize=15)
        # # Optimize layout
        # plt.tight_layout()
        # # Show the plot
        # st.pyplot(fig)
        
        # # Least 20 Areas
        # st.subheader("Least 20 Areas by Frequency")
        # # Get top 20 most frequent Areas
        # least_areas = df['Area'].value_counts().tail(20)
        # # Create a new figure
        # fig, ax = plt.subplots(figsize=(8, 8))
        # # Plot a bar chart of the top 20 Areas
        # sns.barplot(x=least_areas.index, y=least_areas.values, palette='viridis',hue=least_areas.index)
        # # Rotate x-axis labels and set font size for readability
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=80, fontsize=12)
        # # Set axis labels and title
        # ax.set_xlabel('Area', fontsize=14)
        # ax.set_ylabel('Frequency', fontsize=14)
        # ax.set_title('Least 20 Areas', fontsize=14)
        # # Optimize layout
        # plt.tight_layout()
        # # Show the plot
        # st.pyplot(fig)  

        # # Yield Trends Over Time 
        # st.subheader("Yearly Yield Trends")
        # # Group data by 'Year' and calculate the mean of 'hg/ha_yield'
        # yearly_yield = df.groupby('Year')['Hg/ha_yield'].mean()
        # # Create a line plot of average yield over time
        # fig, ax = plt.subplots(figsize=(8, 5))
        # yearly_yield.plot(ax=ax, color='orange', linewidth=2, marker='o')
        # # Set chart title and axis labels
        # ax.set_title('Average Crop Yield per Year (hg/ha)', fontsize=16)
        # ax.set_xlabel('Year', fontsize=14)
        # ax.set_ylabel('Average Yield (hg/ha)', fontsize=14)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        # # Add grid for better visual guidance
        # ax.grid(True, linestyle='--', alpha=0.6)
        # # Optimize layout
        # plt.tight_layout()
        # # Display the plot
        # st.pyplot(fig)

        # # Maximum Yield per Crop
        # st.subheader("Max Yield per Crop by Area")
        # # Group by Crop and get the row with the highest yield per crop
        # best_crop = df.loc[df.groupby('Crop')['Hg/ha_yield'].idxmax()][['Crop', 'Area', 'Hg/ha_yield']]
        # # Create a bar plot of maximum yield per crop
        # fig, ax = plt.subplots(figsize=(8, 8))
        # sns.barplot(data=best_crop,x='Area', y='Hg/ha_yield', hue='Crop')
        # # Set the title and axis labels
        # ax.set_title('Maximum Yield per Crop by Area (hg/ha)', fontsize=16)
        # ax.set_xlabel('Area', fontsize=14)
        # ax.set_ylabel('Maximum Yield (hg/ha)', fontsize=14)
        # # Adjust tick label font sizes
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        # # Resize and position the hue legend
        # ax.legend(title='Crop', title_fontsize=12, fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
        # # Tidy up layout
        # plt.tight_layout()
        # # Display the plot
        # st.pyplot(fig)

        # # Total Yield by Crop
        # st.subheader("Total Yield by Crop Type")
        # # Group by crop and calculate **total** yield
        # yield_per_crop = df.groupby('Crop')['Hg/ha_yield'].sum()
        # # Plot using Seaborn
        # fig, ax= plt.subplots(figsize=(10, 6))
        # sns.barplot(x=yield_per_crop.values, y=yield_per_crop.index, color='orange', ax=ax)
        # # Set the title and axis labels
        # ax.set_title('Total Crop Yield (hg/ha)',  fontsize=16)
        # ax.set_xlabel('Yield (hg/ha)', fontsize=14)
        # ax.set_ylabel('Crop', fontsize=14)
        # # Optimize layout
        # plt.tight_layout()
        # st.pyplot(fig)

        # # Average Yield by Area
        # st.subheader("Average Yield by Area")
        # # Group by Area and compute average yield
        # area_avg = df.groupby('Area')['Hg/ha_yield'].mean().reset_index()
        # area_avg.columns = ['Area', 'Hg/ha_yield']
        # # Find the country/area with the best average yield
        # best_area = area_avg.loc[area_avg['Hg/ha_yield'].idxmax()]
        # st.write("Country/Area with the best average yield:")
        # st.write(best_area)
        # # Plot average yield per area
        # fig, ax = plt.subplots(figsize=(7, 13))
        # sns.barplot(data=area_avg, x='Hg/ha_yield', y='Area', hue='Area', palette="viridis")
        # # Set the title and axis labels
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        # ax.set_xlabel('Average Yield (hg/ha)', fontsize=12)
        # ax.set_ylabel('Area', fontsize=12)
        # ax.set_title('Average Crop Yield by Area', fontsize=14)
        # # Optimize layout
        # plt.tight_layout()
        # st.pyplot(fig)

        # # Rainfall Trends
        # st.subheader("Average Rainfall per Year")
        # # Group data by 'Year' and calculate the mean of 'hg/ha_yield'
        # rainfall_year = df.groupby('Year')['Average_rain_fall_mm_per_year'].mean()
        # # Create a line plot of average yield over time
        # fig, ax = plt.subplots(figsize=(8, 5))
        # rainfall_year.plot(color='red', linewidth=2, marker='o')
        # # Set chart title and axis labels
        # ax.set_title('Average Rainfall per Year', fontsize=16)
        # ax.set_xlabel('Year', fontsize=14)
        # ax.set_ylabel('Average Rainfall (mm)', fontsize=14)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        # # Add grid for better visual guidance
        # ax.grid(True, linestyle='--', alpha=0.6)
        # # Optimize layout
        # plt.tight_layout()
        # # Display the plot
        # st.pyplot(fig)

        # # Top 10 Areas by Rainfall
        # st.subheader("Top 10 Areas by Avg Rainfall")
        # # Group by Area and calculate average rainfall
        # top_rain = df.groupby('Area')['Average_rain_fall_mm_per_year'].mean()
        # # Sort in descending order and get the top 10
        # top10_avg_rainfall = top_rain.sort_values(ascending=False).head(10)
        # # Convert to DataFrame for better handling
        # top10_df = top10_avg_rainfall.reset_index()
        # top10_df.columns = ['Area', 'Average_Rainfall']
        # # Plot
        # fig, ax = plt.subplots(figsize=(9,7))
        # sns.barplot(data=top10_df, x='Average_Rainfall', y='Area', 
        #             palette='dark:orange', hue='Area')
        # # Chart styling
        # ax.set_title('Top 10 Countries by Average Rainfall', fontsize=18)
        # ax.set_xlabel('Average Rainfall (mm/year)', fontsize=15)
        # ax.set_ylabel('Country/Area', fontsize=15)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # plt.tight_layout()
        # st.pyplot(fig)

        # # Rainfall vs Yield
        # st.subheader("Rainfall vs Crop Yield")
        # # Plot
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.scatterplot(data=df, x='Average_rain_fall_mm_per_year', 
        #                 y='Hg/ha_yield', hue='Crop', alpha=0.6, ax=ax)
        # ax.set_title('Effect of Rainfall on Crop Yield', fontsize=16)
        # ax.set_xlabel('Average Rainfall (mm/year)', fontsize=14)
        # ax.set_ylabel('Yield (hg/ha)', fontsize=14)
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # ax.legend(title='Country/Area', bbox_to_anchor=(1.05, 1), loc='upper left')
        # # plt.tight_layout()
        # st.pyplot(fig)                                                 

        # # Trend by Selected Crop 
        # st.subheader("Trend Analysis by Crop")        
        # # Filter the DataFrame by the selected crop
        # selected_crop = st.selectbox("Select a Crop", sorted(df['Crop'].unique()))
        # df_crop = df[df['Crop'] == selected_crop]
        
        # crop_metrics = {'Hg/ha_yield': ('Yield (hg/ha)', 'blue'),
        #     'Pesticides_tonnes': ('Pesticides (tonnes)', 'black'),
        #     'Average_rain_fall_mm_per_year': ('Rainfall (mm)', 'red'),
        #     'Avg_temp': ('Avg Temperature (°C)', 'orange')}
        
        # # Loop through each metric and create a time series plot
        # for col, (label, color) in crop_metrics.items():
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     # Plot mean values grouped by year
        #     df_crop.groupby('Year')[col].mean().plot(ax=ax, color=color, marker='o')
        #     # Add plot formatting
        #     ax.set_title(f"{label} over Years for {selected_crop}", fontsize=16)
        #     ax.set_xlabel("Year", fontsize=14)
        #     ax.set_ylabel(label, fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     ax.legend()
        #     ax.grid(True)
        #     # Display the plot
        #     plt.tight_layout()
        #     # Display the plot in Streamlit
        #     st.pyplot(fig)    
    
        # # Country Specific Yield Trend 
        # st.subheader("Country-Specific Yield Trends")
        # selected_country = st.selectbox("Select a Country", sorted(df['Area'].unique()))
        # filtered = df[df['Area'] == selected_country]
        # fig, ax = plt.subplots()
        # sns.lineplot(data=filtered, x='Year', y='Hg/ha_yield', hue='Crop', marker='o', ax=ax)
        # ax.set_title(f"Crop Yield Over Time in {selected_country}", fontsize=16)
        # ax.set_xlabel("Year", fontsize=14)
        # ax.set_ylabel("Yield (hg/ha)", fontsize=14)
        # for label in ax.get_xticklabels():
        #     label.set_fontsize(14)
        # for label in ax.get_yticklabels():
        #     label.set_fontsize(14)
        # st.pyplot(fig)

        # # Pesticide Usage
        # st.subheader("Pesticide Usage Overview")
        # with st.expander("Top Countries by Pesticide Use"):
        #     top_pesticide = df.groupby('Area')['Pesticides_tonnes'].sum().sort_values(ascending=False).head(10)
        #     fig, ax = plt.subplots()
        #     sns.barplot(x=top_pesticide.values, y=top_pesticide.index, 
        #                 palette="Reds_r", ax=ax, hue= top_pesticide.index)
        #     ax.set_title("Top 10 Countries by Pesticide Use", fontsize=16)
        #     ax.set_xlabel("Total Pesticides Used (tonnes)", fontsize=14)
        #     ax.set_ylabel("Country", fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)        

        # with st.expander("Global Pesticide Use Over Time"):
        #     pesticide_trend = df.groupby('Year')['Pesticides_tonnes'].sum()
        #     fig, ax = plt.subplots()
        #     sns.lineplot(x=pesticide_trend.index, y=pesticide_trend.values, marker='o', ax=ax)
        #     ax.set_title("Pesticide Use Over Time", fontsize=16)
        #     ax.set_xlabel("Year", fontsize=14)
        #     ax.set_ylabel("Global Pesticide Use Over Time", fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)

        # # Crop with Most Pesticide
        # with st.expander("Crops With Most Pesticide Used"):
        #     crop_pesticide = df.groupby('Crop')['Pesticides_tonnes'].sum().sort_values(ascending=False).head(10)
        #     fig, ax = plt.subplots()
        #     sns.barplot(x=crop_pesticide.values, y=crop_pesticide.index, palette="YlOrBr", 
        #                 ax=ax, hue=crop_pesticide.index)
        #     ax.set_title("Top 10 Crops by Pesticide Use", fontsize=16)
        #     ax.set_xlabel('Total Pesticides Used (tonnes)', fontsize=14)
        #     ax.set_ylabel('Crop', fontsize=14)
        #     plt.tight_layout()
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)
        
        # # Yield & Environment
        # st.header("Yield vs Environment")
        # with st.expander("Effect of Pesticide on Crop Yield"):
        #     fig, ax = plt.subplots()
        #     sns.scatterplot(data=df, x='Pesticides_tonnes', y='Hg/ha_yield', hue='Crop', ax=ax)
        #     ax.set_title("Pesticide vs Crop Yield", fontsize=16)
        #     ax.set_xlabel("Pesticides (tonnes)", fontsize=14)
        #     ax.set_ylabel("Yield (hg/ha)",fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)
        
        # with st.expander("Effect of Temperature on Yield"):
        #     fig, ax = plt.subplots()
        #     sns.scatterplot(data=df, x='Avg_temp', y='Hg/ha_yield', hue='Crop', ax=ax)
        #     ax.set_title("Temperature vs Yield", fontsize=16)
        #     ax.set_xlabel("Temperature (°C)", fontsize=14)
        #     ax.set_ylabel("Yield (hg/ha)", fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)

        # # Rainfall & Temperature
        # st.header("Rainfall and Temperature")
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader("Average Temperature by Year")
        #     avg_temp_year = df.groupby('Year')['Avg_temp'].mean()
        #     fig, ax = plt.subplots()
        #     sns.lineplot(x=avg_temp_year.index, y=avg_temp_year.values, marker='o', ax=ax)
        #     ax.set_title("Avg. Temperature Over Years", fontsize=16)
        #     ax.set_xlabel("Year", fontsize=14)
        #     ax.set_ylabel("Temperature (°C)", fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)
        
        # with col2:
        #     st.subheader("Average Rainfall by Year")
        #     avg_rain_year = df.groupby('Year')['Average_rain_fall_mm_per_year'].mean()
        #     fig, ax = plt.subplots()
        #     sns.lineplot(x=avg_rain_year.index, y=avg_rain_year.values, marker='o', ax=ax)
        #     ax.set_title("Avg. Rainfall Over Years", fontsize=16)
        #     ax.set_xlabel("Year", fontsize=14)
        #     ax.set_ylabel("Rainfall (mm)", fontsize=14)
        #     for label in ax.get_xticklabels():
        #         label.set_fontsize(14)
        #     for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        #     st.pyplot(fig)

        # # Pesticide vs Yield Scatterplot
        # st.header("Pesticide Use vs Crop Yield")
        # avg_data_country = df.groupby('Area')[['Pesticides_tonnes', 'Hg/ha_yield']].mean().dropna()
        # fig, ax = plt.subplots()
        # sns.scatterplot(data=avg_data_country, x='Pesticides_tonnes', y='Hg/ha_yield')
        # ax.set_title('Average Pesticide Use vs Crop Yield by Country')
        # ax.set_xlabel('Average Pesticide Use (tonnes)')
        # ax.set_ylabel('Average Yield (hg/ha)')
        # st.pyplot(fig)

        # # Top 10 Countries by Average Temperature
        # st.header("Top 10 Hottest Countries (Avg. Temp)")
        # # Average temperature by country (Top 10 warmest)
        # fig, ax = plt.subplots()
        # avg_temp_country = df.groupby('Area')['Avg_temp'].mean().sort_values(ascending=False).head(10)
        # sns.barplot(x=avg_temp_country.values, y=avg_temp_country.index, palette="coolwarm", hue=avg_temp_country.index)
        # ax.set_title('Top 10 Countries by Average Temperature')
        # ax.set_xlabel('Average Temperature (°C)')
        # ax.set_ylabel('Country')
        # st.pyplot(fig)

        # # Set seaborn style
        # sns.set(style='whitegrid', palette='Set2')
        # # Rainfall vs Pesticide Use
        # st.header("Rainfall vs Pesticide Use by Crop")
        # # Scatterplot: Relationship between Rainfall and Pesticide Use
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.scatterplot(data=df, x='Average_rain_fall_mm_per_year', y='Pesticides_tonnes', hue='Crop', alpha=0.7)
        # ax.set_title('Rainfall vs Pesticide Use by Crop')
        # ax.set_xlabel('Average Rainfall (mm/year)')
        # ax.set_ylabel('Pesticide Use (tonnes)')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # st.pyplot(fig)
        
        # # KDE + Barplot for Rainfall, Pesticides, Temperature, and Yield
        # st.header("Distributions: Rainfall, Pesticides, Temperature, and Yield")        
        # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        # sns.histplot(df['Average_rain_fall_mm_per_year'], kde=True, ax=axes[0, 0], color='blue')
        # axes[0, 0].set_title('Average Rainfall Distribution')
        # sns.histplot(df['Pesticides_tonnes'], kde=True, ax=axes[0, 1], color='green')
        # axes[0, 1].set_title('Pesticide Usage Distribution')
        # sns.histplot(df['Avg_temp'], kde=True, ax=axes[1, 0], color='orange')
        # axes[1, 0].set_title('Average Temperature Distribution')
        # sns.histplot(df['Hg/ha_yield'], kde=True, ax=axes[1, 1], color='purple')
        # axes[1, 1].set_title('Crop Yield Distribution')
        # plt.tight_layout()
        # st.pyplot(fig)
        
        # # Boxplot: Temperature Needed by Various Crops
        # st.header("Temperature Distribution by Crop")
        # fig, ax = plt.subplots(figsize=(12, 6))
        # sns.boxplot(data=df, x='Crop', y='Avg_temp', palette='coolwarm', hue='Crop')
        # ax.set_title('Temperature Distribution for Each Crop')
        # ax.set_xlabel('Crop')
        # ax.set_ylabel('Average Temperature (°C)')
        # for label in ax.get_xticklabels():
        #     label.set_fontsize(14)
        # for label in ax.get_yticklabels():
        #         label.set_fontsize(14)
        # st.pyplot(fig)
        
        # # Average Yield per Year by Crop
        # st.header("Average Crop Yield per Year") 
        # fig, ax = plt.subplots(figsize=(14, 7))
        # sns.lineplot(data=df, x='Year', y='Hg/ha_yield', hue='Crop', estimator='mean', errorbar=None)
        # ax.set_title('Average Crop Yield per Year')
        # ax.set_xlabel('Year')
        # ax.set_ylabel('Yield (hg/ha)')
        # ax.legend(title='Crop', bbox_to_anchor=(1.05, 1), loc='upper left')
        # st.pyplot(fig)
    else:
        # Message shown if user tries to access this step before uploading data
        st.warning("Please upload a dataset first in the 'Upload Data' section.")

elif options == "Visualization" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# STEP 4: Preprocessing
elif options == "Preprocessing":
    st.header("Data Preprocessing")

    # Ensure the cleaned dataframe exists in session state
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()

        
        st.subheader("Encoding Categorical Features")
        # Function to encode categorical features using LabelEncoder
        def encode_categorical(dataframe, columns):
            encoder = LabelEncoder()
            for col in columns:
                dataframe[col] = encoder.fit_transform(dataframe[col])
            return dataframe

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include="object").columns.tolist()

        # Encode them
        df = encode_categorical(df, categorical_columns)
        st.success("Categorical features encoded successfully!")
        st.dataframe(df.head())

        
        st.subheader("Correlation Analysis")
        # Compute correlation matrix
        correlation_matrix = df.corr()
        # Visualize correlation matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Heatmap of Correlation Matrix', fontsize=18)
        st.pyplot(fig)

        # Function to find highly correlated feature pairs above a threshold
        def find_high_correlation_pairs(corr_matrix, threshold=0.5):
            corr_pairs = []
            columns = corr_matrix.columns
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    value = corr_matrix.iloc[i, j]
                    if abs(value) >= threshold:
                        corr_pairs.append((columns[i], columns[j], value))
            return corr_pairs

        # Check for highly correlated feature pairs (above or below ±0.5)
        high_corr_columns = find_high_correlation_pairs(df.corr(), threshold=0.5)
        st.subheader("Highly Correlated Feature Pairs ≥ ±0.5)")

        if high_corr_columns:
            for col1, col2, corr_val in high_corr_columns:
                st.write(f"Correlation between `{col1}` and `{col2}` is `{corr_val:.2f}`")
        else:
            st.info("No highly correlated feature pairs found.")

        # Function to drop one column from each highly correlated pair
        def drop_highly_correlated_features(dataframe, corr_matrix, threshold=0.5):
            to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        # Drop the second column in the pair
                        to_drop.add(col2)
            dataframe = dataframe.drop(columns=list(to_drop))
            return dataframe, list(to_drop)

        # Drop correlated features
        df, dropped_cols = drop_highly_correlated_features(df, correlation_matrix, threshold=0.5)

        # Display dropped columns if any
        if dropped_cols:
            st.subheader("Dropped Highly Correlated Columns")
            st.write(f"Columns dropped due to high correlation (>|0.5|): `{', '.join(dropped_cols)}`")
        else:
            st.info("No highly correlated features were dropped.")

        st.subheader("Target and Feature Separation")
        # Ensure target column exists
        if 'Hg/ha_yield' in df.columns:
            st.write("Target Column Selected: `'Hg/ha_yield'`")

            # Split into features and target
            X = df.drop('Hg/ha_yield', axis=1)
            Y = df['Hg/ha_yield']

            st.write("Feature Columns:")
            st.write(X.columns)
        else:
            st.error("Target column `'Hg/ha_yield'` not found. Please check your dataset.")
            st.stop()

        
        st.subheader("Feature Scaling (Standard Normalization)")
         # Function to normalize features
        def normalize_features(features):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            return pd.DataFrame(scaled, columns=features.columns)

        # Apply normalization
        X = normalize_features(X)
        st.success("Features normalized successfully!")

        # Store processed features and target in session state
        st.session_state.X_processed = X
        st.session_state.Y_processed = Y

        st.success("Data preprocessing complete!")

    else:
        # Data not available warning
        st.warning("Please upload a dataset first in the 'Upload Data' section.")

# Fallback: Prevents error if accessed before cleaning
elif options == "Preprocessing" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

        
        
# # STEP 5: Model Training
# st.subheader("🤖 Model Development")

# test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
# random_state = st.number_input("Random state (for reproducibility)", value=42)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=int(random_state))

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# # STEP 6: Evaluation
# st.subheader("📋 Model Evaluation")
# st.write("Accuracy:", accuracy_score(y_test, y_pred))
# st.text("Classification Report:")
# st.text(classification_report(y_test, y_pred))

# # Predict with user input
# st.subheader("📝 Make a Prediction")
# input_data = {}
# for col in df.drop(columns=[target_col]).columns:
#     value = st.text_input(f"Enter value for {col}")
#     input_data[col] = value

# if st.button("Predict"):
#     input_df = pd.DataFrame([input_data])

#     for col in input_df.columns:
#         if input_df[col].dtype == 'object':
#             input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])

#     input_df_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_df_scaled)
#     st.success(f"🎉 Predicted class: {prediction[0]}")
