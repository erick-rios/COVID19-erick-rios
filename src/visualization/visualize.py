import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import  display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"]==1]
plt.plot(set_df["death"])
# --------------------------------------------------------------
# Plot all labels to see the relation between age and death
# --------------------------------------------------------------
for label in df["set"].unique():
    subset = df[df["set"] == label]
    value_counts = subset["death"].value_counts().sort_index()
    value_counts.plot(kind="bar", label=f"Label {label}")
    plt.xlabel("Death")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

for label in df["sex"].unique():
    subset = df[df["set"] == label]
    value_counts = subset["death"].value_counts().sort_index()
    value_counts.plot(kind="bar", label=f"Label {label}")
    plt.xlabel("Death")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------