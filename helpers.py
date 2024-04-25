"""
Python script with all utils function across notebooks.
"""
import pandas as pd


def load_data_df(file_path: str):
    """Load the .csv data located in path variable into pandas DataFrame"""
    # Read the .csv file
    data = pd.read_csv(filepath_or_buffer=file_path)
    # Convert their special date format to a more clear one
    data['yyyymm'] = pd.to_datetime(data['yyyymm'], format='%Y%m')
    data = data.rename(columns={'yyyymm': 'date'})  # rename the column

    return data
