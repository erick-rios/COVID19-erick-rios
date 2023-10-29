import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import multiprocessing



# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df = pd.read_pickle("/home/soundskydriver/Documents/COVID19-erick-rios/data/interim/02_outliers_removed_chauvenet.pkl")

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
# Correlation using LogisticRegression
# --------------------------------------------------------------
model = LogisticRegression()
model.fit(X, Y)  # X es tu conjunto de características, y es la variable objetivo

coeficients = model.coef_[0]  # Obtén los coeficientes del modelo

feature_coef = pd.DataFrame({'Feature': X.columns, 'Coefficient': coeficients})



# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
feature_set_1 = df.columns.drop(["pregnant", "age","death"])
learner = ClassificationAlgorithms()

max_features= 12
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, Y_train
)

plt.figure(figsize = (10,5))
plt.plot(np.arange(1, max_features+1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks((1,max_features+1,1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    selected_features
]

feature_names = [
    "Feature Set 1",
    "Selected Features",
]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i + 1)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            Y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(Y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, Y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(Y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(Y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(Y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, Y_train, selected_test_X)

    performance_test_nb = accuracy_score(Y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

score_df.sort_values(ascending=False, by="accuracy")

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
plt.figure(figsize=(10,10))
sns.barplot(x = "model", y = "accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

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
