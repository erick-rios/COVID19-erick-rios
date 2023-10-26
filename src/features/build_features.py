import numpy as np
from numbers import Number
from scipy.integrate import quad
from typing import List
import pandas as pd


def get_outliers(data: List[float]) -> List[Number]:
    N = len(data)
    x_n = np.mean(data)
    s_x = np.std(data, ddof=1)

    def t(x):
        return np.abs(x - x_n)/s_x

    def G(x):
        return 1 / np.sqrt(2 * np.pi) * np.e**(-x**2 / 2)

    def prob(t):
        return quad(G, -t, t)[0]

    def n(x):
        return N*(1-prob(t(x)))

    outliers = []
    for i in range(len(data)):
        n_i = n(data[i])
        if n_i < 0.5:
            outliers.append(data[i])

    return outliers


def clean_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    val_col_name = value

    # Initialize result DataFrame. Outliers will be removed from this table.
    clean_df = df.copy()

    # Group by all columns except column of values
    columns = df.columns.values.tolist()
    columns.remove(val_col_name)
    grouped_df = df.groupby(columns)

    # Save outlier rows (index list)
    to_be_removed = []

    for key, group in grouped_df:

        # Get outliers of the group
        outliers = get_outliers(group[val_col_name].to_list())

        # Find index for each outlier
        for outlier in outliers:

            # Initialize query
            query = f'`{val_col_name}` == {outlier}'

            # Build query adding a condition for each column value
            for column_i in range(len(columns)):

                # Match value and column
                col_name = columns[column_i]
                col_value = key[column_i]
                col_type = group[col_name].dtypes

                # Add quotes if value is not numeric (string or text)
                if col_type == 'object':
                    query += f' and `{col_name}` == "{col_value}"'
                else:
                    query += f' and `{col_name}` == {col_value}'

            # Run query and get a single index
            i = df.query(query).index.values[0]

            # Save index
            to_be_removed.append(i)

    # Remove outliers looping through the index list in decreasing way
    # (otherwise the integrity of the indices would be compromised)
    to_be_removed.sort()
    i = len(to_be_removed) - 1
    while i >= 0:
        clean_df.drop(to_be_removed[i], inplace=True)
        i -= 1

    clean_df.reset_index(drop=True, inplace=True)

    return clean_df

def add_mean_and_std(df: pd.DataFrame, value: str) -> pd.DataFrame:
    columns = list(df.columns)
    columns.remove(value)
    grouped_df = df.groupby(columns, as_index=False)
    grouped_df = grouped_df.agg({value: ['mean', 'std']})
    grouped_df.columns = columns + ['mean', 'std']

    return grouped_df