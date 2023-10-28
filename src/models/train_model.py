import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

df_train = df.drop("age", axis = 1)

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
X = df_train.drop("death", axis = 1)
Y = df_train["death"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y, test_size=0.3, random_state=42, stratify=Y
    )
fig, ax = plt.subplots(figsize=(10,5))
df_train["death"].value_counts().plot(
    kind="bar", ax = ax, color = "lightblue", label = "Total"
)
Y_train.value_counts().plot(
    kind="bar", ax = ax, color = "dodgerblue", label = "Train"
)
Y_test.value_counts().plot(
    kind="bar", ax = ax, color = "royalblue", label = "Test"
)
plt.legend()
plt.show()

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
