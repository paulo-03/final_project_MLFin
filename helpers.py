"""
Python script with all utils function across notebooks.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm


def load_data_df(file_path: str) -> pd.DataFrame:
    """Load the .csv data located in path variable into pandas DataFrame"""
    # Read the .csv file
    data = pd.read_csv(filepath_or_buffer=file_path)
    # Convert their special date format to a more clear one
    data['yyyymm'] = pd.to_datetime(data['yyyymm'], format='%Y%m')
    data = data.rename(columns={'yyyymm': 'date'})  # rename the column
    # Delete all rows with year 2024 because of lack of data
    data = data[data['date'].dt.year != 2024]

    return data


def select_time_window(data: pd.DataFrame, from_: int, to_: int):
    """Filter the dataframe to keep only rows within the time window selected"""
    data = data[(data['date'].dt.year >= from_) & (data['date'].dt.year <= to_)]
    return data


def data_per_month(data: pd.DataFrame) -> pd.DataFrame:
    """Return a listh of dataframe including all records of same month/year"""
    # Start by listing all unique date in the dataframe
    date_list = data['date'].unique()
    # Start grouping the records
    data_months = []
    for date in tqdm(date_list, desc='Grouping all observations per date: '):
        data_months.append(data[data['date'] == date])

    return data_months
