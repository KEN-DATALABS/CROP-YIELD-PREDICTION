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
from const.markdowns import welcome_note, area_distribution_insight
from helpers.verify_columns import verify
from services.eda import EDA
from services.cleaning import Cleaning

from services.visualisation import pie_plot, bar_plot, count_plot, line_plot, scatter_plot

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

    if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        st.write(df.head())

        if 'Crop' in df.columns:
            with st.expander("Crop Distribution"):
                crops = df['Crop'].value_counts()
                top_crop = crops.idxmax()
                least_crop = crops.idxmin()
                top_count = crops.max()
                least_count = crops.min()

                col1, col2 = st.columns(2)

                with col1:
                    pie_plot(values=crops.values, labels=crops.index.tolist(), 
                             title="Crop Distribution (Pie)", figsize=(8, 8))

                with col2:
                    crop_df = crops.reset_index()
                    crop_df.columns = ['Crop', 'Frequency']

                    bar_plot(df=crop_df,x='Crop',y='Frequency',title='Crop Frequency Bar Chart',
                             xlabel='Crop',ylabel='Frequency',palette='Spectral',figsize=(8, 8),rotation_x=80)

                # Explanatory markdown (placed after both plots)
                st.markdown(f"""
                ### Key Insights:
                - **{top_crop}** is the most frequent crop with **{top_count} entries**.
                - **{least_crop}** is the least frequent crop with **{least_count} entries**.
                - The pie chart provides proportional context, while the bar chart gives precise frequency comparisons.
                """)

        with st.expander("Area Distribution"):
            col1, col2 = st.columns(2)
            with col1:
                # Add labels and title for clarity
                title= 'Distribution of Records by Area'
                xlabel= 'Frequency'
                ylabel= 'Area'
                count_plot(df,column = 'Area', xlabel= xlabel,title= title, palette= 'husl', figsize= (7, 15))
            
            with col2:
                # Group by Area and compute average yield
                area_avg = df.groupby('Area')['Hg/ha_yield'].mean().reset_index()
                area_avg.columns = ['Area', 'Hg/ha_yield']
                title='Average Crop Yield by Area (hg/ha)'
                # Plot using bar_plot function
                bar_plot(df=area_avg,x='Hg/ha_yield',y='Area',title= title,xlabel='Average Yield (hg/ha)',
                    ylabel='Area',hue='Area',palette='viridis',figsize=(7, 15))
        
            col1, col2 = st.columns(2)
            with col1:
                # Top 20 Areas
                st.subheader("Top 20 Areas by Frequency")
                # Get top 20 most frequent areas
                top_areas = df['Area'].value_counts().head(20).reset_index()
                top_areas.columns = ['Area', 'Frequency']
                # Plot using your bar_plot function
                bar_plot(df=top_areas,x='Area',y='Frequency',title='Top 20 Areas',xlabel='Area',
                        ylabel='Frequency',palette='coolwarm',figsize=(10, 6),rotation_x=80)
                
            with col2:
                # Least 20 Areas
                st.subheader("Least 20 Areas by Frequency")
                # Get least 20 most frequent Areas
                least_areas = df['Area'].value_counts().tail(20).reset_index()
                least_areas.columns = ['Area', 'Frequency']
                # Plot using your bar_plot function
                bar_plot(df=least_areas,x='Area',y='Frequency',title='Least 20 Areas',xlabel='Area',
                        ylabel='Frequency',palette='viridis',figsize=(10, 6),rotation_x=80)
            
            # Find the area with the best average yield
            best_area = area_avg.loc[area_avg['Hg/ha_yield'].idxmax()]
            st.write("Country/Area with the best average yield:")
            st.write(best_area)
            # Display insights
            st.markdown(area_distribution_insight.format(top_area=top_areas.iloc[0]['Area'],top_count=top_areas.iloc[0]['Frequency'],
           least_area=least_areas.iloc[-1]['Area'],least_count=least_areas.iloc[-1]['Frequency']))
        
        # Yield Trends Over Time
        with st.expander("Yearly Yield Trends"):
            col1, col2 = st.columns(2)
            with col1:
                # Group data by 'Year' and calculate the mean of 'hg/ha_yield'
                yearly_yield = df.groupby('Year')['Hg/ha_yield'].mean()
                title='Average Crop Yield per Year (hg/ha)'
                ylabel= 'Average Yield (hg/ha)'
                line_plot(yearly_yield, title= title, xlabel= 'Year', ylabel= ylabel, 
                        color= 'orange', marker= "o", figsize= (8, 5))
            
                # Group data by 'Year' and calculate the mean of 'Pesticides_tonnes'
                yearly_pesticide = df.groupby('Year')['Pesticides_tonnes'].mean()
                title='Average Pesticide Use per Year (tonnes)'
                ylabel= 'Average Pesticide Use (tonnes)'
                line_plot(yearly_pesticide, title= title, xlabel= 'Year', ylabel= ylabel, 
                        color= 'green', marker= "o", figsize= (8, 5))
                
            with col2:
                avg_temp_year = df.groupby('Year')['Avg_temp'].mean()
                title= "Avg. Temperature Over Years"
                line_plot(data= avg_temp_year, x=avg_temp_year.index, y=avg_temp_year.values, 
                          title= title, xlabel= "Year", ylabel= "Temperature (°C)", figsize= (8, 5), color= 'blue')
            
                avg_rain_year = df.groupby('Year')['Average_rain_fall_mm_per_year'].mean()
                title= "Avg. Rainfall Over Years"
                line_plot(data= avg_rain_year, x=avg_rain_year.index,title= title, 
                          xlabel= "Year", ylabel= "Rainfall (mm)", figsize= (8, 5), color= 'purple')


        # Country Specific Yield Trend 
        with st.expander("Country-Specific Yield Trends"):
            selected_country = st.selectbox("Select a Country", sorted(df['Area'].unique()))
            filtered = df[df['Area'] == selected_country]
            title= f"Crop Yield Over Time in {selected_country}"
            line_plot(data= filtered, x='Year', y='Hg/ha_yield', hue='Crop', title= title, 
                    xlabel= "Year", ylabel= "Yield (hg/ha)", color= "blue")
        
        
        with st.expander("Environmental Insights (Rainfall, Pesticide, Temperature)"):
            col1, col2 = st.columns(2)
            #Column 1: Rainfall + Temperature
            with col1:
                # Top 10 Areas by Rainfall
                st.markdown("### Top 10 Countries by Average Rainfall")
                top_rain = df.groupby('Area')['Average_rain_fall_mm_per_year'].mean().sort_values(ascending=False).head(10).reset_index()
                top_rain.columns = ['Area', 'Average_Rainfall']
                title='Top 10 Countries by Average Rainfall (mm/year)'
                bar_plot(df=top_rain,x='Average_Rainfall',y='Area',title=title,xlabel='Avg Rainfall (mm/year)',
                         ylabel='Country/Area',hue='Area',palette='dark:orange',figsize=(9, 7))

                # Top 10 Hottest Countries (Average Temp)
                st.markdown("### Top 10 Hottest Countries (Avg. Temperature)")
                avg_temp = df.groupby('Area')['Avg_temp'].mean().sort_values(ascending=False).head(10).reset_index()
                avg_temp.columns = ['Area', 'Average_Temperature']
                title='Top 10 Countries by Average Temperature'
                bar_plot(df=avg_temp,x='Average_Temperature',y='Area',title=title,xlabel='Average Temperature (°C)',
                         ylabel='Country',hue='Area',palette="coolwarm",figsize=(9, 6))

            # Column 2: Pesticide Usage
            with col2:
                st.markdown("### Top 10 Countries by Pesticide Use")
                top_pesticide = df.groupby('Area')['Pesticides_tonnes'].sum().sort_values(ascending=False).head(10).reset_index()
                top_pesticide.columns = ['Area', 'Pesticides_Used']
                title="Top 10 Countries by Pesticide Use"
                xlabel="Total Pesticides Used (tonnes)"
                bar_plot(df=top_pesticide,x='Pesticides_Used',y='Area',title=title,xlabel=xlabel,
                         ylabel="Country",hue='Area',palette="Reds_r",figsize=(9, 6))
        
        with st.expander("Trend Analysis by Crop"):        
            # Select a crop
            selected_crop = st.selectbox("Select a Crop", sorted(df['Crop'].unique()))
            df_crop = df[df['Crop'] == selected_crop]
            
            crop_metrics = {
                'Hg/ha_yield': ('Yield (hg/ha)', 'blue'),
                'Pesticides_tonnes': ('Pesticides (tonnes)', 'black'),
                'Average_rain_fall_mm_per_year': ('Rainfall (mm)', 'red'),
                'Avg_temp': ('Avg Temperature (°C)', 'orange')}
            # Get list of metric items
            metric_items = list(crop_metrics.items())
            # Loop through metrics in pairs of 2 to make 2 columns per row
            for i in range(0, len(metric_items), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(metric_items):
                        with cols[j]:
                            col_name, (label, color) = metric_items[i + j]
                            df_metric = df_crop.groupby('Year')[col_name].mean()
                            title = f"{label} over Years for {selected_crop}"
                            line_plot(data=df_metric,title=title,xlabel="Year",
                                      ylabel=label,color=color,figsize=(8, 5))


        with st.expander("Crop Yield Insights (Max & Total Yield)"):
            col1, col2 = st.columns(2)

            # Column 1: Maximum Yield per Crop by Area
            with col1:
                st.markdown("### Maximum Yield per Crop by Area")
                best_crop = df.loc[df.groupby('Crop')['Hg/ha_yield'].idxmax()][['Crop', 'Area', 'Hg/ha_yield']]
                title='Maximum Yield per Crop by Area (hg/ha)'
                ylabel='Maximum Yield (hg/ha)'
                bar_plot(df=best_crop,x='Area',y='Hg/ha_yield',title=title,xlabel='Area',ylabel=ylabel,
                         hue='Crop',palette='tab10',figsize=(10, 6),rotation_x=45,legend_title='Crop',
                         legend_loc='upper right',legend_bbox=(1.15, 1))

            # Column 2: Total Yield by Crop
            with col2:
                st.markdown("### Total Yield by Crop Type")
                yield_per_crop = df.groupby('Crop')['Hg/ha_yield'].sum().reset_index()
                yield_per_crop.columns = ['Crop', 'Total_Yield']
                title='Total Crop Yield (hg/ha)'
                ylabel='Total Yield (hg/ha)'
                bar_plot(df=yield_per_crop,x='Crop',y='Total_Yield',title=title,
                         xlabel='Crop',ylabel=ylabel,palette='autumn',figsize=(10, 6),rotation_x=45)

        # Crops With Most Pesticide Used
        with st.expander("Crops With Most Pesticide Used"):
            # Group by Crop and sum pesticide usage
            crop_pesticide = df.groupby('Crop')['Pesticides_tonnes'].sum().sort_values(ascending=False).head(10).reset_index()
            crop_pesticide.columns = ['Crop', 'Pesticides_Used']
            title="Top 10 Crops by Pesticide Usage"
            # Plot using the reusable bar_plot function
            bar_plot(df=crop_pesticide,x='Pesticides_Used',y='Crop',title=title,xlabel="Total Pesticides Used (tonnes)",
                    ylabel="Crop",hue='Crop',palette="YlOrBr",figsize=(9, 6))


        #Yield vs Environment
        with st.expander("Yield vs Environment Factors"):
            col1, col2 = st.columns(2)
            with col1:
                title='Pesticide vs Crop Yield'
                xlabel='Pesticides (tonnes)'
                scatter_plot(df=df,x='Pesticides_tonnes',y='Hg/ha_yield',title=title,xlabel=xlabel,
                             ylabel='Yield (hg/ha)',hue='Crop',alpha=0.7)

                scatter_plot(df=df,x='Avg_temp',y='Hg/ha_yield',title='Temperature vs Crop Yield',
                             xlabel='Temperature (°C)',ylabel='Yield (hg/ha)',hue='Crop')

            with col2:
                # Rainfall vs Yield
                scatter_plot(df=df,x='Average_rain_fall_mm_per_year',y='Hg/ha_yield',title='Effect of Rainfall on Crop Yield',
                             xlabel='Average Rainfall (mm/year)',ylabel='Yield (hg/ha)',hue='Crop',alpha=0.6)


        # Country-Level Resource Usage
        with st.expander("Country-level Averages (Pesticide vs Yield)"):
            avg_data_country = df.groupby('Area')[['Pesticides_tonnes', 'Hg/ha_yield']].mean().dropna().reset_index()
            title='Avg Pesticide Use vs Crop Yield by Country'
            scatter_plot(df=avg_data_country,x='Pesticides_tonnes',y='Hg/ha_yield',title=title,
                         xlabel='Avg Pesticide Use (tonnes)',ylabel='Avg Yield (hg/ha)')


        # Rainfall vs Pesticide Use
        with st.expander("Rainfall vs Pesticide Use by Crop"):
            title='Rainfall vs Pesticide Use by Crop'
            xlabel='Average Rainfall (mm/year)'
            x='Average_rain_fall_mm_per_year'
            scatter_plot(df=df,x=x,y='Pesticides_tonnes',title=title,
                         xlabel=xlabel,ylabel='Pesticide Use (tonnes)',hue='Crop',alpha=0.7)

        with st.expander("Distributions: Rainfall, Pesticides, Temperature, and Yield"):
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            # Rainfall
            sns.histplot(df['Average_rain_fall_mm_per_year'], kde=True, ax=axes[0, 0], color='blue')
            axes[0, 0].set_title('Average Rainfall Distribution', fontsize=16)
            axes[0, 0].set_xlabel('Rainfall (mm/year)', fontsize=15)
            axes[0, 0].set_ylabel('Frequency', fontsize=14)
            # Pesticides
            sns.histplot(df['Pesticides_tonnes'], kde=True, ax=axes[0, 1], color='green')
            axes[0, 1].set_title('Pesticide Usage Distribution', fontsize=16)
            axes[0, 1].set_xlabel('Pesticides (tonnes)', fontsize=15)
            axes[0, 1].set_ylabel('Frequency', fontsize=14)
            # Temperature
            sns.histplot(df['Avg_temp'], kde=True, ax=axes[1, 0], color='orange')
            axes[1, 0].set_title('Average Temperature Distribution', fontsize=16)
            axes[1, 0].set_xlabel('Temperature (°C)', fontsize=15)
            axes[1, 0].set_ylabel('Frequency', fontsize=14)
            # Yield
            sns.histplot(df['Hg/ha_yield'], kde=True, ax=axes[1, 1], color='purple')
            axes[1, 1].set_title('Crop Yield Distribution', fontsize=16)
            axes[1, 1].set_xlabel('Yield (hg/ha)', fontsize=15)
            axes[1, 1].set_ylabel('Frequency', fontsize=14)
            # Update tick label fonts for all subplots
            for ax in axes.flat:
                for label in ax.get_xticklabels():
                    label.set_fontsize(12)
                    # label.set_rotation(30)
                for label in ax.get_yticklabels():
                    label.set_fontsize(12)
            plt.tight_layout()
            st.pyplot(fig)

        # Boxplot: Temperature Needed by Various Crops
        with st.expander("Temperature Distribution by Crop"):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df, x='Crop', y='Avg_temp', palette='coolwarm', hue='Crop')
            ax.set_title('Temperature Distribution for Each Crop', fontsize=16)
            ax.set_xlabel('Crop', fontsize=12)
            ax.set_ylabel('Average Temperature (°C)', fontsize=12)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(10)
            for label in ax.get_yticklabels():
                label.set_fontsize(11)
            st.pyplot(fig)
        
        # Average Yield per Year by Crop
        with st.expander("Average Crop Yield per Year"):
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.lineplot(data=df, x='Year', y='Hg/ha_yield', hue='Crop', estimator='mean', errorbar=None)
            ax.set_title('Average Crop Yield per Year', fontsize=16)
            ax.set_xlabel('Year', fontsize=15)
            ax.set_ylabel('Yield (hg/ha)', fontsize=15)
            ax.legend(title='Crop', bbox_to_anchor=(1.05, 1), loc='upper left')
            for label in ax.get_xticklabels():
                label.set_fontsize(13)
            for label in ax.get_yticklabels():
                label.set_fontsize(13)
            st.pyplot(fig)
    else:
        # Message shown if user tries to access this step before uploading data
        st.warning("Please upload a dataset first in the 'Upload Data' section.")

elif options == "Visualization" and st.session_state.cleaned_df is None:
    st.warning("Please clean your data first in the 'Data Cleaning' section.")

# # STEP 4: Preprocessing
# elif options == "Preprocessing":
#     st.header("Data Preprocessing")

#     # Ensure the cleaned dataframe exists in session state
#     if st.session_state.cleaned_df is not None:
#         df = st.session_state.cleaned_df.copy()


#         st.subheader("Encoding Categorical Features")
#         # Function to encode categorical features using LabelEncoder
#         def encode_categorical(dataframe, columns):
#             encoder = LabelEncoder()
#             for col in columns:
#                 dataframe[col] = encoder.fit_transform(dataframe[col])
#             return dataframe

#         # Identify categorical columns
#         categorical_columns = df.select_dtypes(include="object").columns.tolist()

#         # Encode them
#         df = encode_categorical(df, categorical_columns)
#         st.success("Categorical features encoded successfully!")
#         st.dataframe(df.head())


#         st.subheader("Correlation Analysis")
#         # Compute correlation matrix
#         correlation_matrix = df.corr()
#         # Visualize correlation matrix
#         fig, ax = plt.subplots(figsize=(8, 8))
#         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
#         ax.set_title('Heatmap of Correlation Matrix', fontsize=18)
#         st.pyplot(fig)

#         # Function to find highly correlated feature pairs above a threshold
#         def find_high_correlation_pairs(corr_matrix, threshold=0.5):
#             corr_pairs = []
#             columns = corr_matrix.columns
#             for i in range(len(columns)):
#                 for j in range(i + 1, len(columns)):
#                     value = corr_matrix.iloc[i, j]
#                     if abs(value) >= threshold:
#                         corr_pairs.append((columns[i], columns[j], value))
#             return corr_pairs

#         # Check for highly correlated feature pairs (above or below ±0.5)
#         high_corr_columns = find_high_correlation_pairs(df.corr(), threshold=0.5)
#         st.subheader("Highly Correlated Feature Pairs ≥ ±0.5)")

#         if high_corr_columns:
#             for col1, col2, corr_val in high_corr_columns:
#                 st.write(f"Correlation between `{col1}` and `{col2}` is `{corr_val:.2f}`")
#         else:
#             st.info("No highly correlated feature pairs found.")

#         # Function to drop one column from each highly correlated pair
#         def drop_highly_correlated_features(dataframe, corr_matrix, threshold=0.5):
#             to_drop = set()
#             for i in range(len(corr_matrix.columns)):
#                 for j in range(i + 1, len(corr_matrix.columns)):
#                     col1 = corr_matrix.columns[i]
#                     col2 = corr_matrix.columns[j]
#                     if abs(corr_matrix.iloc[i, j]) > threshold:
#                         # Drop the second column in the pair
#                         to_drop.add(col2)
#             dataframe = dataframe.drop(columns=list(to_drop))
#             return dataframe, list(to_drop)

#         # Drop correlated features
#         df, dropped_cols = drop_highly_correlated_features(df, correlation_matrix, threshold=0.5)

#         # Display dropped columns if any
#         if dropped_cols:
#             st.subheader("Dropped Highly Correlated Columns")
#             st.write(f"Columns dropped due to high correlation (>|0.5|): `{', '.join(dropped_cols)}`")
#         else:
#             st.info("No highly correlated features were dropped.")

#         st.subheader("Target and Feature Separation")
#         # Ensure target column exists
#         if 'Hg/ha_yield' in df.columns:
#             st.write("Target Column Selected: `'Hg/ha_yield'`")

#             # Split into features and target
#             X = df.drop('Hg/ha_yield', axis=1)
#             Y = df['Hg/ha_yield']

#             st.write("Feature Columns:")
#             st.write(X.columns)
#         else:
#             st.error("Target column `'Hg/ha_yield'` not found. Please check your dataset.")
#             st.stop()


#         st.subheader("Feature Scaling (Standard Normalization)")
#          # Function to normalize features
#         def normalize_features(features):
#             scaler = StandardScaler()
#             scaled = scaler.fit_transform(features)
#             return pd.DataFrame(scaled, columns=features.columns)

#         # Apply normalization
#         X = normalize_features(X)
#         st.success("Features normalized successfully!")

#         # Store processed features and target in session state
#         st.session_state.X_processed = X
#         st.session_state.Y_processed = Y

#         st.success("Data preprocessing complete!")

#     else:
#         # Data not available warning
#         st.warning("Please upload a dataset first in the 'Upload Data' section.")

# # Fallback: Prevents error if accessed before cleaning
# elif options == "Preprocessing" and st.session_state.cleaned_df is None:
#     st.warning("Please clean your data first in the 'Data Cleaning' section.")



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
# st.success(f"🎉 Predicted class: {prediction[0]}")
