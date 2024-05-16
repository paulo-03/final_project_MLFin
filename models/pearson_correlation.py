"""
This python script propose a class containing all useful method in a pearson correlation.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PearsonCorrelation:
    def __init__(self, data: pd.DataFrame):
        self.correlation_matrix = data.drop(columns=['permno', 'date']).corr()

    def get_sorted_correlation_pairs(self, ascending=False, top_k_pairs: int = None) -> pd.DataFrame:
        # Reset the index
        correlation_matrix_reset = self.correlation_matrix.reset_index()

        # Melt the DataFrame
        melted = correlation_matrix_reset.melt(id_vars='index')

        # Create a new column that contains the pair of columns
        melted['pair'] = list(zip(melted['index'], melted['variable']))

        # Set the new column as the index and select the value column
        flattened = melted.set_index('pair')['value'].rename('correlation')

        # Print the flattened Series
        correlation_pairs = flattened.sort_values(ascending=ascending)[flattened != 1].iloc[::2].reset_index()
        # If k_first is provided, return only the first k_first elements
        if top_k_pairs is not None:
            return correlation_pairs.head(top_k_pairs)

        return correlation_pairs

    def _plot_mean_correlation(self):
        means = self.correlation_matrix.mean().sort_values()
        # plot means
        plt.figure(figsize=(20, 10))
        sns.barplot(x=means.index, y=means.values)
        plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
        plt.title('Mean Correlation')
        plt.show()

    def _plot_median_correlation(self):
        medians = self.correlation_matrix.median().sort_values()
        # plot medians
        plt.figure(figsize=(20, 10))
        sns.barplot(x=medians.index, y=medians.values)
        plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
        plt.title('Median Correlation')
        plt.show()

    def _plot_correlation_matrix_heatmap(self):
        plt.figure(figsize=(20, 20))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='Reds', fmt=".2f", linewidths=0.5)
        plt.title('Correlation')
        plt.show()

    def plot(self, plot_type: str = 'heatmap'):
        if plot_type == 'heatmap':
            self._plot_correlation_matrix_heatmap()
        elif plot_type == 'mean':
            self._plot_mean_correlation()
        elif plot_type == 'median':
            self._plot_median_correlation()
        else:
            raise ValueError('Plot type not supported')
