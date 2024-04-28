"""
Python script with useful functions for data analysis.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def asset_price_plot(data: pd.DataFrame, permno_id: int):
    """Plot the price evolution of selected permno_id"""
    asset = data[data['permno'] == permno_id]

    # Plot the price evolution in function of time
    plt.plot(asset['date'], asset['Price'])

    # Customize the plot if needed
    plt.title(f'(Log)Price Over Time (permno: {permno_id})')  # Set the title of the plot
    plt.xlabel('Date')  # Set the label for the x-axis
    plt.xticks(rotation=45)
    plt.ylabel('Price')  # Set the label for the y-axis
    plt.grid(True)  # Enable grid lines
    plt.show()


def assets_information(data: pd.DataFrame):
    """Given the overall dataset, it returns the number of asset and some information about it."""
    # Group by 'permno' and aggregate
    asset_info = data.groupby('permno').agg(
        data_quantity=('date', 'count'),
        date_from=('date', 'min'),
        date_to=('date', 'max')
    ).reset_index()
    # Calculate the date window
    asset_info['month_window'] = (asset_info['date_to'].dt.year - asset_info['date_from'].dt.year) * 12 + (
            asset_info['date_to'].dt.month - asset_info['date_from'].dt.month)
    # Compute the ratio of Nan values in the data for each asset. Allows us to see if a specific asset has too many Nan
    ratio_nan_attributes = data.groupby('permno').apply(lambda x: x.isnull().mean()).drop(columns=["date", "permno"])
    asset_info['ratio_nan_price'] = ratio_nan_attributes['Price'].reset_index(drop=True)
    asset_info['ratio_nan_overall'] = ratio_nan_attributes.mean(axis=1).reset_index(drop=True)
    # Print the number of different asset and the window time of our dataset
    asset_number = data['permno'].nunique()
    oldest_date = data['date'].min()
    latest_date = data['date'].max()

    print("%" * 10, "Asset Information", "%" * 10,
          f"\nNumber of assets: {asset_number}",
          f"\nOldest date in dataset: {oldest_date}",
          f"\nLatest date in dataset: {latest_date}",
          "\n%" + "%" * 38)

    return asset_info, ratio_nan_attributes


def attributes_type(file_path: str):
    """Separate attributes/predictors into continuous and discrete predictors."""
    # Read the .csv file
    data_description = pd.read_csv(filepath_or_buffer=file_path)
    # Select only predictors to analyze them
    data_description = data_description[data_description['Cat.Signal'] == 'Predictor']
    # Separate continuous and discrete
    continuous_attributes = data_description[data_description['Cat.Form'] == 'continuous']['Acronym']
    discrete_attributes = data_description[data_description['Cat.Form'] == 'discrete']['Acronym']
    # Print the result of such quick analysis
    print("%" * 10, "Attributes Type", "%" * 10,
          f"\nNumber of continuous attributes: {len(continuous_attributes)}",
          f"\nNumber of discrete attributes: {len(discrete_attributes)}",
          "\n%" + "%" * 36)

    return continuous_attributes, discrete_attributes


def continuous_attributes_description(data: pd.DataFrame, column_name: list):
    """Compute a brief statistical description of continuous attributes"""
    # Initialize dictionaries to store information
    continuous_info = {'attribute': [], 'min': [], 'max': [], 'mean': [], 'nan_ratio': []}
    # For numerical attributes
    for col in column_name:
        nan_ratio = data[col].isnull().mean()
        continuous_info['attribute'].append(col)
        continuous_info['min'].append(data[col].min())
        continuous_info['max'].append(data[col].max())
        continuous_info['mean'].append(data[col].mean())
        continuous_info['nan_ratio'].append(nan_ratio)
    # Convert dict to pd.Dataframe
    continuous_info_df = pd.DataFrame(continuous_info)
    # Plot the Nan ratio over the attributes
    plt.figure(figsize=(80, 10))
    sns.barplot(data=continuous_info_df.sort_values(by='nan_ratio', ascending=True), x='attribute', y='nan_ratio')
    plt.xticks(rotation=45)
    plt.show()

    return continuous_info_df


def discrete_attributes_description(data: pd.DataFrame, column_name: list):
    """Compute a brief statistical description of discrete attributes"""
    # Initialize dictionaries to store information
    discrete_info = {'attribute': [], 'unique_values': [], 'min': [], 'max': [], 'mean': [], 'nan_ratio': []}
    # For numerical attributes
    for col in column_name:
        nan_ratio = data[col].isnull().mean()
        discrete_info['attribute'].append(col)
        discrete_info['unique_values'].append(data[col].nunique())
        discrete_info['min'].append(data[col].min())
        discrete_info['max'].append(data[col].max())
        discrete_info['mean'].append(data[col].mean())
        discrete_info['nan_ratio'].append(nan_ratio)
    # Convert dict to pd.Dataframe
    discrete_info_df = pd.DataFrame(discrete_info)
    # Plot the Nan ratio over the attributes
    plt.figure(figsize=(20, 5))
    sns.barplot(data=discrete_info_df.sort_values(by='nan_ratio', ascending=True), x='attribute', y='nan_ratio')
    plt.xticks(rotation=45)
    plt.show()

    return discrete_info_df


def evolution_over_time(data: pd.DataFrame):
    """Compute few metrics evolution over time."""
    #  Create a year column to ease the group by
    data['year'] = data['date'].dt.year
    # Group by 'year' and aggregate
    data_evolution = data.groupby('year').agg(
        asset_number=('permno', lambda x: x.nunique()),
        data_quantity=('date', 'count'),
    ).reset_index()
    # Compute the ratio of Nan values in the data for each year. Allows us to see if a specific year has too many Nan
    ratio_nan_attributes = data.groupby('year').apply(lambda x: x.isnull().mean()).drop(columns=["date", "permno"])
    data_evolution['ratio_nan_price'] = ratio_nan_attributes['Price'].reset_index(drop=True)
    data_evolution['ratio_nan_overall'] = ratio_nan_attributes.mean(axis=1).reset_index(drop=True)
    # Finally plot all results to visualize them
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    sns.lineplot(data=data_evolution, x='year', y='asset_number', ax=axs[0][0])
    sns.lineplot(data=data_evolution, x='year', y='data_quantity', ax=axs[0][1])
    sns.lineplot(data=data_evolution, x='year', y='ratio_nan_price', ax=axs[1][0])
    sns.lineplot(data=data_evolution, x='year', y='ratio_nan_overall', ax=axs[1][1])
    return data_evolution


def select_attributes_nan(attributes_infos: pd.DataFrame, max_nan_ratio: float):
    """Select only attributes that have at most "max_nan_ratio" of nan values over de dataset."""
    return attributes_infos[attributes_infos['nan_ratio'] <= max_nan_ratio]


def _asset_number_over_time(data: pd.DataFrame):
    """Compute the number of asset evolution over years"""
    return data[['date', 'year']].groupby('year').count().reset_index()
