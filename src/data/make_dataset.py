## -*- coding: utf-8 -*-
#import click
#import logging
#from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
#
#
#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
#def main(input_filepath, output_filepath):
#    """ Runs data processing scripts to turn raw data from (../raw) into
#        cleaned data ready to be analyzed (saved in ../processed).
#    """
#    logger = logging.getLogger(__name__)
#    logger.info('making final data set from raw data')
#
#
#if __name__ == '__main__':
#    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#    logging.basicConfig(level=logging.INFO, format=log_fmt)
#
#    # not used in this stub but often useful for finding various files
#    project_dir = Path(__file__).resolve().parents[2]
#
#    # find .env automagically by walking up directories until it's found, then
#    # load up the .env entries as environment variables
#    load_dotenv(find_dotenv())
#
#    main()

import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("../../data/raw/*.csv")

def read_data_from_files(files):
    """_summary_

    Args:
        files (_type_): _description_

    Returns:
        _type_: _description_
    """
    covid_df = pd.DataFrame()
    
    for f in files:
        df = pd.read_csv(f)
        df.columns = [col.lower().strip().replace('\n', '') for col in df.columns]

        df["death"] = [2 if each=="9999-99-99" else 1 for each in df.date_died]
        df["pregnant"] = df["pregnant"].replace({ 97: 2, 98: 2})
        df["set"] = pd.cut(df["age"], bins=[0, 20, 40, 60, float('inf')], labels=[1, 2, 3, 4], right=False)
        # Lista de las columnas a las que deseas aplicar las condiciones
        columnas = ["pneumonia", "diabetes", "copd", "asthma", "inmsupr",
                "hipertension", "other_disease", "cardiovascular", 
                "obesity", "renal_chronic", "tobacco"]

        # Recorre las columnas y aplica las condiciones
        for columna in columnas:
            df = df[(df[columna] == 1) | (df[columna] == 2)]

        covid_df = df[['sex', 'patient_type', 'age', 'pregnant', 'diabetes', 'copd', 'asthma', 'inmsupr',
       'hipertension', 'other_disease', 'cardiovascular', 'obesity', 'pneumonia',
       'renal_chronic', 'tobacco', 'clasiffication_final', 'death','set']]
        
    return covid_df


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data = read_data_from_files(files)
data["pregnant"].unique()
data["pneumonia"].unique()
data.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data.to_pickle("../../data/interim/01_data_processed.pkl")