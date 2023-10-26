import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import LocalOutlierFactor 
import build_features# pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("/home/soundskydriver/Documents/COVID19-erick-rios/data/interim/01_data_processed.pkl")
df.columns

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
# Agrupa los datos por sexo y muerte y crea los boxplots
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()
    


# Insert Chauvenet's function
def chauvenet(data, col, alpha=0.05):
    data = data.copy()
    
    z_scores = abs((data[col] - data[col].mean()) / data[col].std())
    probabilities = 2 * (1 - norm.cdf(z_scores))
    is_outlier = probabilities < alpha
    
    # Create a new column 'column_outlier' with True/False values for outliers
    data[f'{col}_outlier'] = is_outlier
    
    return data 


dataset = build_features.clean_table(df, column)
plot_binary_outliers(dataset, "age", "age_outlier", reset_index = True)



col = ['age']
outliersRemovedDf = df.copy()

for value in df["death"].unique():
    for column in col:
        dataset = build_features.clean_table(df[df["death"] == value], column)
        plot_binary_outliers(dataset, "age", "age_outlier", reset_index = True)
        # Replace values marked as outliers with NaN
        outliersRemovedDf.loc[outliersRemovedDf["death"] == value, column] = dataset[column]
        
        numberOutliers = len(df[df["death"] == value]) - len(dataset[column].dropna())
        print(f"Removed {numberOutliers} outliers from {column} for death={value}")




# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliersRemovedDf.to_pickle("../../data/interim/02_outliers_removed_iqr.pkl")
outliersRemovedDf.info()
