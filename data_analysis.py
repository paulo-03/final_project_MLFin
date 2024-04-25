"""
Python script with useful functions for data analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt


def asset_price_plot(data: pd.DataFrame, permno_id: int):
    """Plot the price evolution of selected permno_id"""
    asset = data[data['permno'] == permno_id]

    # Plot the price evolution in function of time
    plt.plot(asset['date'], asset['Price'])

    # Customize the plot if needed
    plt.title(f'(Log)Price Over Time (permno: {permno_id})')  # Set the title of the plot
    plt.xlabel('Date')  # Set the label for the x-axis
    plt.ylabel('Price')  # Set the label for the y-axis
    plt.grid(True)  # Enable grid lines
    plt.show()


def assets_information(data: pd.DataFrame):
    """Given the overall dataset, it returns the number of asset and some information about it."""
    # Select only attributes that we are interested here
    data = data[['permno', 'date', 'Price']]
    # Group by 'permno' and aggregate
    asset_info = data.groupby('permno').agg(
        data_quantity=('date', 'count'),
        date_from=('date', 'min'),
        date_to=('date', 'max'),
        ratio_price_nan=('Price', lambda x: x.isna().sum() / len(x))
    ).reset_index()
    # Calculate the date window
    asset_info['date_window'] = (asset_info['date_to'] - asset_info['date_from'])
    # Reorder the columns
    asset_info = asset_info[['permno', 'data_quantity', 'date_from', 'date_to', 'date_window', 'ratio_price_nan']]
    # Print the number of different asset and the window time of our dataset
    asset_number = data['permno'].nunique()
    oldest_date = data['date'].min()
    latest_date = data['date'].max()
    print("%" * 10, "Asset Information", "%" * 10,
          f"\nNumber of assets: {asset_number}",
          f"\nOldest date in dataset: {oldest_date}",
          f"\nLatest date in dataset: {latest_date}",
          "\n%" + "%" * 38)

    return asset_info


def attributes_type(data: pd.DataFrame):
    """Separate attributes into numerical and categorical predictors."""
    numerical_attributes = data.select_dtypes(include=['number']).columns
    categorical_attributes = data.select_dtypes(exclude=['number']).columns
    # Print the result of such quick analysis
    print("%" * 11, "Attributes Type", "%" * 11,
          f"\nNumber of numerical attributes: {len(numerical_attributes)}",
          f"\nNumber of categorical attributes: {len(categorical_attributes)}",
          "\n%" + "%" * 38)

    return numerical_attributes, categorical_attributes


def numerical_attributes_description(data: pd.DataFrame, column_name: list):
    """Compute a brief statistical description of numerical attributes"""
    # Initialize dictionaries to store information
    numerical_info = {'attribute': [], 'min': [], 'max': [], 'mean': [], 'nan_ratio': []}
    # For numerical attributes
    for col in column_name:
        nan_ratio = data[col].isnull().mean()
        numerical_info['attribute'].append(col)
        numerical_info['min'].append(data[col].min())
        numerical_info['max'].append(data[col].max())
        numerical_info['mean'].append(data[col].mean())
        numerical_info['nan_ratio'].append(nan_ratio)
    # Convert dict to pd.Dataframe
    numerical_info_df = pd.DataFrame(numerical_info)
    return numerical_info_df
