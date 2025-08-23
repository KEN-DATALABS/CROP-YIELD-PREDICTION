import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helpers.read_data import upload_dataset, load_dataset
from const.features import expected_columns, sidenav
from const.markdowns import welcome_note, area_distribution_insight
from helpers.verify_columns import verify
from services.eda import EDA
from services.cleaning import Cleaning
from services.processing import encode_categorical, find_high_correlation_pairs, drop_highly_correlated_features, process_data
from services.visualisation import pie_plot, bar_plot, count_plot, line_plot, scatter_plot
from services.visualisation import plot_model_performance, plot_feature_importance, add_bar_labels, plot_metric_comparison, plot_before_after_comparison
from services.modelling import run_all_transfer_experiments

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
states= ['df', 'cleaned_df', 'target', 'data_splits', 'source_models', 'transfer_results', 'transfer_models','preprocessors']
for key in states:
    if key not in st.session_state:
        st.session_state[key] = None



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
            mime='text/csv')

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
            
            crop_metrics = {'Hg/ha_yield': ('Yield (hg/ha)', 'blue'),
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


# --- Preprocessing Section ---
elif options == "Preprocessing":
    st.header("Data Preprocessing")
    
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
        
        try:
            # Encode categorical features
            st.subheader("Encoding Categorical Features")
            categorical_columns = ['Area', 'Crop']
            label_encoder_area = LabelEncoder()
            label_encoder_crop = LabelEncoder()
            df['Area'] = label_encoder_area.fit_transform(df['Area'])
            df['Crop'] = label_encoder_crop.fit_transform(df['Crop'])
            st.session_state.preprocessors = {
                'label_encoder_area': label_encoder_area,
                'label_encoder_crop': label_encoder_crop}
            
            st.success("Categorical features encoded successfully!")
            st.dataframe(df.head())
            
            # Correlation Analysis
            st.subheader("Correlation Analysis")
            correlation_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
            ax.set_title('Heatmap of Correlation Matrix', fontsize=18)
            st.pyplot(fig)
            
            high_corr_columns = find_high_correlation_pairs(correlation_matrix, threshold=0.5)
            st.subheader("Highly Correlated Feature Pairs ≥ ±0.5")
            if high_corr_columns:
                for col1, col2, corr_val in high_corr_columns:
                    st.write(f"Correlation between `{col1}` and `{col2}` is `{corr_val:.2f}`")
            else:
                st.info("No highly correlated feature pairs found.")
            
            df, dropped_cols = drop_highly_correlated_features(df, correlation_matrix, threshold=0.5)
            if dropped_cols:
                st.subheader("Dropped Highly Correlated Columns")
                st.write(f"Columns dropped due to high correlation (>|0.5|): `{', '.join(dropped_cols)}`")
            else:
                st.info("No highly correlated features were dropped.")
            
            # Target and Feature Separation
            st.subheader("Target and Feature Separation")
            if 'Hg/ha_yield' in df.columns:
                st.write("Target Column Selected: `'Hg/ha_yield'`")
                X = df.drop('Hg/ha_yield', axis=1)
                Y = df['Hg/ha_yield']
                st.session_state.target = Y
                st.write("Feature Columns:")
                st.write(X.columns)
            else:
                st.error("Target column `'Hg/ha_yield'` not found. Please check your dataset.")
                st.stop()
            
            # Scale features and create splits
            numerical_columns = ['Average_rain_fall_mm_per_year', 'Pesticides_tonnes', 'Avg_temp', 'Year']
            st.write("Processing data splits...")
            try:
                splits = process_data(X, Y, numerical_columns)
            except Exception as e:
                st.error(f"Error in process_data: {e}")
                st.stop()
            
            # Validate splits
            if not splits:
                st.error("No splits created. Check your data and preprocessing steps.")
                st.stop()
            for split_name in ['seen', 'unseen_countries', 'unseen_crops', 'unseen_both']:
                if split_name not in splits:
                    st.error(f"Split '{split_name}' missing. Check process_data implementation.")
                    st.stop()
                if not len(splits[split_name]['X_train']) or not len(splits[split_name]['X_test']):
                    st.error(f"Split '{split_name}' has empty train or test set. Check data distribution (e.g., Year range).")
                    st.stop()
                if 'scaler' not in splits[split_name]:
                    st.error(f"Scaler missing in split '{split_name}'. Check scale_features_and_create_weights.")
                    st.stop()
            if 'transfer_learning' not in splits or not len(splits['transfer_learning']['X_adapt']):
                st.error("Transfer learning split missing or empty. Check prepare_transfer_learning_data.")
                st.stop()
            
            # Store splits and update preprocessors
            st.session_state.data_splits = splits
            st.session_state.preprocessors['scaler'] = splits['seen']['scaler']
            
            # Save preprocessors to disk
            joblib.dump(st.session_state.preprocessors['label_encoder_area'], 'label_encoder_area.pkl')
            joblib.dump(st.session_state.preprocessors['label_encoder_crop'], 'label_encoder_crop.pkl')
            joblib.dump(st.session_state.preprocessors['scaler'], 'scaler.pkl')
            
            # Display split information
            st.subheader("Data Splits Created")
            for split_name, data in splits.items():
                if split_name != 'transfer_learning':
                    st.write(f"**{split_name}**: Train size: {len(data['X_train'])}, Test size: {len(data['X_test'])}")
                else:
                    st.write(f"**{split_name}**: Adaptation size: {len(data['X_adapt'])}, Final test size: {len(data['X_test_unseen_final'])}")
            
            st.success("Data processing complete! All splits and preprocessors saved.")
        
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            st.write("Please check your data and try again.")
            st.stop()
    
    else:
        st.warning("Please upload and clean your data first in the 'Upload Data' section.")
        st.stop()
                
# --- Modeling Section ---
elif options == "Modeling":
    st.subheader("Model Training and Transfer Learning")
    
    if 'data_splits' not in st.session_state or not st.session_state.data_splits:
        st.warning("Please process data first in the Preprocessing section.")
        st.stop()
    
    # Get data splits
    try:
        splits = st.session_state.data_splits
        X_train_final = splits['unseen_countries']['X_train']
        y_train_final = splits['unseen_countries']['y_train']
        sample_weights1 = splits['unseen_countries']['sample_weights']
        X_test_unseen = splits['unseen_countries']['X_test']
        y_test_unseen = splits['unseen_countries']['y_test']
        X_adapt = splits['transfer_learning']['X_adapt']
        y_adapt = splits['transfer_learning']['y_adapt']
        X_test_unseen_final = splits['transfer_learning']['X_test_unseen_final']
        y_test_unseen_final = splits['transfer_learning']['y_test_unseen_final']
    except KeyError:
        st.error("Data splits are incomplete. Please reprocess data in the Preprocessing section.")
        st.stop()
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", len(X_train_final))
    with col2:
        st.metric("Adaptation Samples", len(X_adapt))
    with col3:
        st.metric("Test Samples", len(X_test_unseen_final))
    
    # Model parameters
    st.subheader("Model Parameters")
    
    rf_params = {
        'n_estimators': 50,
        'max_depth': 10,
        'min_samples_split': 2,
        'random_state': 42}
    
    xgb_params = {
        'n_estimators': 50,
        'max_depth': 6,
        'learning_rate': 0.5,
        'random_state': 42}
    
    st.write("**Random Forest Parameters**")
    st.write(f"- n_estimators: {rf_params['n_estimators']}")
    st.write(f"- max_depth: {rf_params['max_depth']}")
    st.write(f"- min_samples_split: {rf_params['min_samples_split']}")
    st.write("**XGBoost Parameters**")
    st.write(f"- n_estimators: {xgb_params['n_estimators']}")
    st.write(f"- max_depth: {xgb_params['max_depth']}")
    st.write(f"- learning_rate: {xgb_params['learning_rate']}")
    
    # Try loading source models
    source_models = {}
    try:
        source_models['Random Forest'] = joblib.load('rf_model.pkl')
        source_models['XGBoost'] = joblib.load('xgb_model.pkl')
        st.session_state.source_models = source_models
        st.info("Loaded pre-trained source models from disk.")
    except FileNotFoundError:
        st.warning("Source model files not found. Please train models below.")
    
    # Train source models
    if st.button("Train Source Models", type="primary"):
        st.info("Training source models with best parameters...")
        
        # Train Random Forest
        st.write("Training Random Forest...")
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train_final, y_train_final, sample_weight=sample_weights1)
        source_models['Random Forest'] = rf_model
        joblib.dump(rf_model, 'rf_model.pkl')
        st.success("Random Forest trained and saved successfully!")
        
        # Train XGBoost
        st.write("Training XGBoost...")
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_final, y_train_final, sample_weight=sample_weights1)
        source_models['XGBoost'] = xgb_model
        joblib.dump(xgb_model, 'xgb_model.pkl')
        st.success("XGBoost trained and saved successfully!")
        
        # Store source models
        st.session_state.source_models = source_models
        st.success("All models trained and saved successfully!")
        
        # Show source model performance
        st.subheader("Source Model Performance")
        for model_name, model in source_models.items():
            y_pred = model.predict(X_test_unseen)
            r2 = r2_score(y_test_unseen, y_pred)
            mae = mean_absolute_error(y_test_unseen, y_pred)
            mape = mean_absolute_percentage_error(y_test_unseen, y_pred) * 100
            st.write(f"**{model_name}**: R² = {r2:.4f}, MAE = {mae:.2f}, MAPE = {mape:.2f}%")
    
    # Run transfer learning experiments
    if 'source_models' in st.session_state and st.session_state.source_models:
        st.subheader("Transfer Learning Experiments")
        if st.button("Run Transfer Learning Experiments", type="primary"):
            st.info("Running transfer learning experiments. This may take a while...")
            
            models_config = [
                ('Random Forest', source_models['Random Forest'], RandomForestRegressor(**rf_params)),
                ('XGBoost', source_models['XGBoost'], XGBRegressor(**xgb_params))]
            
            results_df, transfer_models = run_all_transfer_experiments(
                models_config, 
                X_train_final, y_train_final, 
                X_adapt, y_adapt, 
                X_test_unseen, y_test_unseen,
                X_test_unseen_final, y_test_unseen_final, 
                sample_weights1)
            
            # Save transfer models
            for model_name, model in transfer_models.items():
                joblib.dump(model, f'transfer_{model_name.lower().replace(" ", "_")}_model.pkl')
            
            # Store results and models
            st.session_state.transfer_results = results_df
            st.session_state.transfer_models = transfer_models
            
            # Display results
            st.subheader("Transfer Learning Results")
            st.dataframe(results_df.style.format({
                'No Transfer R²': '{:.4f}',
                'Transfer R²': '{:.4f}',
                'R² Improvement (pp)': '{:.2f}',
                'R² Improvement (%)': '{:.2f}',
                'No Transfer MAE': '{:.2f}',
                'Transfer MAE': '{:.2f}',
                'MAE Improvement': '{:.2f}',
                'MAE Reduction (%)': '{:.2f}',
                'No Transfer MAPE (%)': '{:.2f}',
                'Transfer MAPE (%)': '{:.2f}',
                'MAPE Improvement (%)': '{:.2f}',
                'MAPE Reduction (%)': '{:.2f}'}))
            
            # Show best model
            best_model_idx = results_df['Transfer R²'].idxmax()
            best_model = results_df.loc[best_model_idx, 'Model']
            st.success(f"Best performing model: {best_model} (R²: {results_df.loc[best_model_idx, 'Transfer R²']:.4f})")
            
            # Visualizations
            with st.expander("Model Performance Visualizations", expanded=True):
                st.subheader("Model Performance Comparison")
                fig_r2_improvement = plot_metric_comparison(
                    results_df, 'R² Improvement (%)', 
                    'Percentage R² Improvement from Transfer Learning', 
                    'R² Improvement (%)', 'viridis')
                st.pyplot(fig_r2_improvement)
                
                fig_mae_reduction = plot_metric_comparison(
                    results_df, 'MAE Reduction (%)', 
                    'Percentage MAE Reduction from Transfer Learning', 
                    'MAE Reduction (%)', 'viridis')
                st.pyplot(fig_mae_reduction)
                
                fig_mape_reduction = plot_metric_comparison(
                    results_df, 'MAPE Reduction (%)', 
                    'Percentage MAPE Reduction from Transfer Learning', 
                    'MAPE Reduction (%)', 'viridis')
                st.pyplot(fig_mape_reduction)
                
                st.subheader("Before and After Transfer Learning")
                fig_r2_comparison = plot_before_after_comparison(
                    results_df, 'No Transfer R²', 'Transfer R²',
                    'R² Comparison: Transfer Learning vs No Transfer',
                    'R² Score')
                st.pyplot(fig_r2_comparison)
                
                fig_mae_comparison = plot_before_after_comparison(
                    results_df, 'No Transfer MAE', 'Transfer MAE',
                    'MAE Comparison: Transfer Learning vs No Transfer',
                    'MAE')
                st.pyplot(fig_mae_comparison)
                
                fig_mape_comparison = plot_before_after_comparison(
                    results_df, 'No Transfer MAPE (%)', 'Transfer MAPE (%)',
                    'MAPE Comparison: Transfer Learning vs No Transfer',
                    'MAPE (%)')
                st.pyplot(fig_mape_comparison)
            
            # Feature Importance
            with st.expander("Feature Importance Analysis", expanded=False):
                st.subheader("Feature Importance")
                feature_names = X_train_final.columns.tolist() + ['source_pred']
                for model_name in transfer_models:
                    fig_importance = plot_feature_importance(
                        transfer_models[model_name], feature_names, model_name)
                    st.pyplot(fig_importance)
            
            # Individual Model Performance
            with st.expander("Individual Model Performance", expanded=False):
                st.subheader("Detailed Model Performance")
                for model_name in transfer_models:
                    X_test_transfer = X_test_unseen_final.copy()
                    source_model = source_models[model_name]
                    X_test_transfer['source_pred'] = source_model.predict(X_test_unseen_final)
                    y_pred = transfer_models[model_name].predict(X_test_transfer)
                    fig_performance = plot_model_performance(
                        y_test_unseen_final, y_pred, model_name)
                    st.pyplot(fig_performance)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="transfer_learning_results.csv",
                mime="text/csv")
    else:
        st.error("No source models found. Please train source models first.")

# --- Prediction Section ---
elif options == "Prediction":
    st.subheader("Crop Yield Prediction")
    
    if st.session_state.preprocessors is None or not st.session_state.preprocessors:
        st.warning("Please process data first in the Preprocessing section to initialize preprocessors.")
        st.stop()
    
    # Load transfer models
    transfer_models = {}
    try:
        transfer_models['Random Forest'] = joblib.load('transfer_random_forest_model.pkl')
        transfer_models['XGBoost'] = joblib.load('transfer_xgboost_model.pkl')
        st.info("Loaded pre-trained transfer models from disk.")
    except FileNotFoundError:
        st.warning("Transfer models not found. Please run transfer learning experiments in the Modeling section.")
        st.stop()
    
    # Load preprocessors
    label_encoder_area = st.session_state.preprocessors['label_encoder_area']
    label_encoder_crop = st.session_state.preprocessors['label_encoder_crop']
    scaler = st.session_state.preprocessors['scaler']
    
    # Display model performance (placeholders, update with actual results)
    st.header("Model Performance Metrics (Transfer Learning)")
    col1, col2 = st.columns(2)
  
    with col1:
        st.subheader("Random Forest")
        st.write("Transfer R²: 0.9256")
        st.write("Transfer MAE: 14113.68")
        st.write("Transfer MAPE: 44.55%")
    with col2:
        st.subheader("XGBoost")
        st.write("Transfer R²: 0.9250")
        st.write("Transfer MAE: 13226.37")
        st.write("Transfer MAPE: 27.31%")
    
    # File upload for predictions
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Area', 'Crop', 'Year', 'Average_rain_fall_mm_per_year', 'Pesticides_tonnes', 'Avg_temp']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                st.stop()
            
            st.write("Uploaded data preview:")
            st.dataframe(df.head())
            
            # Preprocess data
            df_encoded = df.copy()
            try:
                df_encoded['Area'] = label_encoder_area.transform(df_encoded['Area'])
                df_encoded['Crop'] = label_encoder_crop.transform(df_encoded['Crop'])
            except ValueError:
                st.error("Uploaded data contains unseen Area or Crop values. Ensure values match training data.")
                st.stop()
            
            numerical_columns = ['Average_rain_fall_mm_per_year', 'Pesticides_tonnes', 'Avg_temp', 'Year']
            df_encoded[numerical_columns] = scaler.transform(df_encoded[numerical_columns])
            
            # Add source predictions for transfer learning
            source_models = st.session_state.source_models if 'source_models' in st.session_state else {}
            if not source_models:
                try:
                    source_models['Random Forest'] = joblib.load('rf_model.pkl')
                    source_models['XGBoost'] = joblib.load('xgb_model.pkl')
                    st.session_state.source_models = source_models
                except FileNotFoundError:
                    st.error("Source models not found. Please train source models in the Modeling section.")
                    st.stop()
            
            X = df_encoded[required_columns]
            predictions = {}
            for model_name in transfer_models:
                X_transfer = X.copy()
                X_transfer['source_pred'] = source_models[model_name].predict(X)
                predictions[model_name] = transfer_models[model_name].predict(X_transfer)
            
            # Add predictions to dataframe
            df['RF_Predicted_Yield'] = predictions['Random Forest']
            df['XGB_Predicted_Yield'] = predictions['XGBoost']
            
            # Display predictions
            st.header("Predictions")
            st.write("Predicted crop yields (hg/ha):")
            st.dataframe(df[['Area', 'Crop', 'Year','RF_Predicted_Yield', 'XGB_Predicted_Yield']])
            
            # Download predictions
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="crop_yield_predictions.csv",
                mime="text/csv")
        
        except Exception as e:
            st.error(f"Error processing data or making predictions: {e}")
    
    else:
        st.info("Please upload a CSV file to make predictions.")