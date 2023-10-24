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

category_df = df.query("set==3").query("sex == 1").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["diabetes"])["death"].value_counts().plot(kind="bar", label = "Death sorted by age and sex and diabetes")
ax.set_xlabel("death")
ax.set_ylabel("frecuency")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
category_df_male = df.query("set==4").query("sex == 2").reset_index()

fig, ax = plt.subplots()
category_df_male.groupby(["diabetes"])["death"].value_counts().plot(kind="bar", label = "Death sorted by age and sex and diabetes")
ax.set_xlabel("death")
ax.set_ylabel("frecuency")
plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
set = 4
sex = 1
male = 2
columns = ["diabetes"]
combined_plot_df = df.query(f"set == {set}").query(f"sex == {sex}").reset_index(drop=True)
combined_plot_df_male = df.query(f"set == {set}").query(f"sex == {male}").reset_index(drop=True)

# Combina las columnas de la leyenda en una sola columna llamada 'Legend'
combined_plot_df['Legend'] = combined_plot_df[columns].astype(str).agg('-'.join, axis=1)
combined_plot_df_male['Legend'] = combined_plot_df_male[columns].astype(str).agg('-'.join, axis=1)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

combined_plot_df.groupby('Legend')["death"].value_counts().unstack().plot(kind="bar", ax=ax[0])
combined_plot_df_male.groupby('Legend')["death"].value_counts().unstack().plot(kind="bar", ax=ax[1])

ax[0].set_xticklabels([])  # Elimina las etiquetas del eje x en el primer histograma
ax[1].set_xticklabels([])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.9, 0.9, 0.1), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.9, 0.9, 0.1), ncol=1, fancybox=True, shadow=True)

ax[1].set_xlabel("Death")
ax[1].set_ylabel("Frequency")
ax[0].set_ylabel("Frequency")

plt.savefig(f"../../reports/figures/historigram{set}.png")
plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------