import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import  display
import os

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

columnsToPlot = df.columns.drop(["other_disease","sex","death", "set","age","patient_type", "pregnant","clasiffication_final"])



# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
sets   = df["set"].unique()
sexes  = df["sex"].unique()
ages = {1: '0-20', 2: '20-40',3:'40-60',4: '60+'}
preconditions ={
    "diabetes": "Diabetes",
    "copd": "Chronic Obstructive Pulmonary",
    "asthma": "Asthma",
    "inmsupr": "Inmunosuppressed",
    "hipertension": "Hipertension",
    "cardiovascular": "Heart or Blood Vessels",
    "renal_chronic": "Chronic Renal",
    "obesity": "Obesity",
    "pneumonia": "Pneumonia",
    "tobacco": "Tobacco User",
    
}

for set in sets:
    for column in columnsToPlot:
            
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
        
        combined_plot_df = df.query(f"set == {set}").query(f"sex == {sexes[0]}").reset_index(drop=True)
        combined_plot_df_female = df.query(f"set == {set}").query(f"sex == {sexes[1]}").reset_index(drop=True)
        
        combined_plot_df['death'] = combined_plot_df['death'].replace({1: 'Death', 2: 'No death'})
        combined_plot_df_female['death'] = combined_plot_df['death'].replace({1: 'Death', 2: 'No death'})
        
        
        combined_plot_df.groupby(column)["death"].value_counts().unstack().plot(kind="bar", ax=ax[0])
        combined_plot_df_female.groupby(column)["death"].value_counts().unstack().plot(kind="bar", ax=ax[1])
        
        title = f'Age: {ages[set]}, Sex: Female, Precondition: {preconditions[column]}'
        titleTwo = f'Age: {ages[set]}, Sex: Male, Precondition: {preconditions[column]}'
        
        ax[0].set_xticklabels([f" {preconditions[column]}", f"No  {preconditions[column]}"], rotation = 0)  # Elimina las etiquetas del eje x en el primer histograma
        ax[1].set_xticklabels([f" {preconditions[column]}", f"No  {preconditions[column]}"], rotation = 0)
        
        ax[0].set_title(title)
        ax[1].set_title(titleTwo)
        
        ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.9, 0.9, 0.1), ncol=3, fancybox=True, shadow=True)
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.9, 0.9, 0.1), ncol=1, fancybox=True, shadow=True)    
        
        ax[1].set_xlabel(preconditions[column])
        ax[1].set_ylabel("Frequency")
        ax[0].set_ylabel("Frequency")
        
        if not os.path.exists(f"../../reports/figures/{preconditions[column]}"):
            os.makedirs(f"../../reports/figures/{preconditions[column]}")
        
        plt.savefig(f"../../reports/figures/{preconditions[column]}/historigram_{ages[set]}.png")
        plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------