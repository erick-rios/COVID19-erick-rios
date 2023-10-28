
import pandas as pd
import build_features

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df.drop(columns=["patient_type","other_disease","clasiffication_final"])
df.head()

outlierColumns = [df.columns[1]]

outliersRemovedDf = df.copy()

for value in df["death"].unique():
    for column in outlierColumns:
        dataset = build_features.mark_outliers_chauvenet(df[df["death"] == value], column)
        build_features.plot_binary_outliers(dataset, "age", "age_outlier", reset_index = True)
        # Replace values marked as outliers with NaN
        outliersRemovedDf.loc[outliersRemovedDf["death"] == value, column] = dataset[column]
        
        numberOutliers = len(df[df["death"] == value]) - len(dataset[column].dropna())
        print(f"Removed {numberOutliers} outliers from {column} for death={value}")



# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliersRemovedDf.to_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
outliersRemovedDf.info()
