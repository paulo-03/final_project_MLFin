"""
Python script to process the data. Indeed, a lot of characteristic values are missing. We propose to retrieve them using
the 'missingpy' package and its MissForest algorithm based on Random Forest.
"""

from missingpy import MissForest
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
from helpers import load_data_df, select_time_window, data_per_month

INFORMATION_ATTRIBUTE = ['permno', 'date']


def impute_data(data: pd.DataFrame, seed=42) -> pd.DataFrame:
    """Main function to impute the missing values using MissForest"""
    # Initialize the imputer
    imputer = MissForest(missing_values=np.nan,
                         criterion=('squared_error', 'gini'),
                         max_features='sqrt',
                         random_state=seed,
                         n_estimators=10)
    # Cache the information attribute to not use them in the imputation
    data_info = data[INFORMATION_ATTRIBUTE]
    data_to_impute = data.drop(columns=INFORMATION_ATTRIBUTE)
    char_str = data_to_impute.columns.tolist()
    # Perform the imputation
    data_imputed = pd.DataFrame(data=imputer.fit_transform(data_to_impute), columns=char_str)
    # Concatenate the information attribute to the imputed data
    data_imputed = pd.concat([data_info, data_imputed], axis=1).reset_index(drop=True)

    return data_imputed
