"""
Python script to process the data. Indeed, a lot of characteristic values are missing. We propose to retrieve them using
the 'missingpy' package and its MissForest algorithm based on Random Forest.
"""

from missingpy import MissForest
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
from helpers import load_data_df, select_time_window, data_per_month

INFORMATION_ATTRIBUTE = ['permno', 'date', 'Price']


def impute_data(file_path: str, from_=None, to_=None, seed=42) -> pd.DataFrame:
    """Main function to impute the missing values using MissForest"""
    # First load the data
    print("Loading the data... (estimated time: ~2min)")
    data = load_data_df(file_path=file_path)
    # Keep only the data into the desire time window
    data = select_time_window(data=data, from_=from_, to_=to_)
    print("Data loaded successfully !\n")
    # Create a list of Dataframe. Every Dataframe regroup all record of a current month
    data = data_per_month(data=data)
    # Perform the impute for all the dataframe
    data_imputed = []
    for data_month in tqdm(data, desc='Status of imputation: '):
        # Initialize the imputer
        imputer = MissForest(missing_values=np.nan,
                             criterion=('squared_error', 'gini'),
                             max_features='sqrt',
                             random_state=seed)
        # Cache the information attribute to not use them in the imputation
        data_info = data_month[INFORMATION_ATTRIBUTE]
        data_to_impute = data_month.drop(columns=INFORMATION_ATTRIBUTE)
        char_str = data_to_impute.columns.tolist()
        # Perform the imputation
        data_imputed_not_full = pd.DataFrame(data=imputer.fit_transform(data_to_impute), columns=char_str)
        # Concatenate the information attribute to the imputed data
        data_imputed_full = pd.concat([data_info, data_imputed_not_full], axis=1).reset_index(drop=True)
        # Store the fully imputed dataframe
        data_imputed.append(data_imputed_full)

    # Finally, concatenate all imputed data to reform the initial dataframe
    print("Performing the concatenation...")
    data_imputed = pd.concat(data_imputed).reset_index(drop=True)
    print("Impute process finished !")

    return data_imputed
