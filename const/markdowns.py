welcome_note = """
Welcome to the Crop Yield Prediction App!  
This tool guides you from dataset upload to model prediction through interactive analysis, cleaning, and visualization.  
Upload your dataset or use a sample to begin your journey!
"""
area_distribution_insight="""
### Key Insights:
- The **Top 20 Areas** by frequency reveal the regions with the **highest number of records**, indicating either a higher concentration of agricultural activity or better data reporting in those regions.
- The **most frequent area** is **{top_area}**, with **{top_count} records**, suggesting it plays a major role in the dataset's overall trends.
- On the other hand, the **Least 20 Areas** highlight regions with **very few records**, which could imply underreporting, sparse agricultural presence, or data collection issues.
- Some of the **least represented areas**, like **{least_area}**, have as low as **{least_count} entries**, which could limit statistical significance when analyzing those locations individually.
- These frequency imbalances should be taken into account during model training and analysis, especially when assessing model performance across geographic regions.
**Recommendation**: Consider strategies such as data augmentation or targeted data collection in underrepresented areas to improve overall dataset balance.
"""