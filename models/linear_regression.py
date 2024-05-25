"""
This python script propose a class containing all useful method in a linear regression.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import LassoCV

PASTEL_BLUE = '#aec6cf'
PASTEL_GREEN = '#77dd77'


class OLS:
    def __init__(self, predictors: pd.DataFrame, label: pd.Series):
        self.pred_name = predictors.columns.tolist()
        self.X = np.array(predictors)
        self.y = np.array(label)
        self.weights = None
        self.reg = None
        self.r_square = None

    def fit(self):
        self.reg = LinearRegression().fit(X=self.X, y=self.y)
        self.r_square = self.reg.score(self.X, self.y)
        self.weights = self.reg.coef_

    def show_weights(self, top_k: int = 20):
        """Allows to show nicely plotted the non-zero weights chosen by the Lasso method."""
        # Make sure the user has already performed the fitting step
        if self.weights is None:
            raise ValueError("Please perform the regression by calling '.fit' method before calling the '.show_weights"
                             "method.")

        # Use argsort to get the indices that would sort the array
        sorted_indices = np.argsort(np.abs(self.weights))
        # Select the indices of the top k values
        top_k_indices = sorted_indices[-top_k:][::-1]
        # Get the top k values and names
        top_k_values = self.weights[top_k_indices]
        top_k_names = [self.pred_name[i] for i in top_k_indices]

        # Determine colors based on the sign of the weights
        colors = [PASTEL_BLUE if value > 0 else PASTEL_GREEN for value in top_k_values]

        # Plot the absolute values of the weights
        plt.figure(figsize=(10, 4))
        plt.bar(x=top_k_names, height=np.abs(top_k_values), color=colors)
        plt.title(f'Top {top_k} Predictors From linear Regression')
        plt.xlabel('Predictor Name')
        plt.xticks(rotation=90)
        plt.ylabel('Absolute Weight')

        # Create custom legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=PASTEL_BLUE, lw=4),
                        Line2D([0], [0], color=PASTEL_GREEN, lw=4)]
        plt.legend(custom_lines, ['Positive Weights', 'Negative Weights'])

        plt.show()


class OLSLasso(OLS):
    def __init__(self,
                 predictors: pd.DataFrame,
                 label: pd.Series,
                 alpha: float = None,
                 seed: int = 42):
        super().__init__(predictors, label)
        self.alpha = alpha
        self.seed = seed
        self.r_square = None
        self.cpu_num = os.cpu_count() // 2  # set to "-1" if you want to use all cpu cores available

    def fit(self):
        """Method to train the data on the training data using the given alpha or the chosen one by the CV method."""
        if self.alpha is not None:
            self.reg = Lasso(alpha=self.alpha, random_state=self.seed).fit(X=self.X, y=self.y, njobs=-2)
            self.r_square = self.reg.score(self.X, self.y)
            self.weights = self.reg.coef_
        else:
            self.fit_alpha_cv()

    def fit_alpha_cv(self, from_: float = 0.001, to_: float = 0.05, val_number: int = 100):
        """Method that will try the val_number alphas between from_ and to_ values and test the Lasso regression using
        cross-validation to choose the best alpha possible."""
        # Initiate and fit model to perform the cross_validation
        alphas = np.linspace(from_, to_, val_number)
        self.reg = LassoCV(alphas=alphas,
                           n_jobs=self.cpu_num,
                           random_state=self.seed).fit(X=self.X, y=self.y)
        # Retrieve important information and plot the CV progress
        self.alpha = self.reg.alpha_
        self.weights = self.reg.coef_
        self.r_square = self.reg.score(self.X, self.y)
        alphas_mse_avg = self.reg.mse_path_.mean(axis=1)[::-1]

        # Find the index and the corresponded alpha value of the smallest value in alphas_mse_avg
        min_index = np.argmin(alphas_mse_avg)
        min_alpha = alphas[min_index]

        # Plot results
        plt.plot(alphas, alphas_mse_avg, color=PASTEL_BLUE)
        plt.axvline(x=min_alpha, color=PASTEL_GREEN, linestyle='--', label=f'Min MSE at alpha={min_alpha:.4f}')
        plt.title(r'5-Fold Cross Validation to Find The Best $\alpha$')
        plt.xlabel('alpha')
        plt.ylabel('mean mse over folds')
        plt.legend()
        plt.show()

    def show_weights(self, top_k: int = 20):
        """Allows to show nicely plotted the non-zero weights chosen by the Lasso method."""
        if self.weights is None:
            raise ValueError(
                "Please perform the regression by calling '.fit' method before calling the '.show_weights' method.")

        non_zero_weights = self.weights != 0
        if np.sum(non_zero_weights) == 0:
            raise ValueError("No non-zero weights found. Try a smaller alpha.")

        sorted_indices = np.argsort(np.abs(self.weights[non_zero_weights]))
        top_k_indices = sorted_indices[-top_k:][::-1]
        top_k_values = self.weights[non_zero_weights][top_k_indices]
        top_k_names = np.array(self.pred_name)[non_zero_weights][top_k_indices]

        # Determine colors based on the sign of the weights
        colors = [PASTEL_BLUE if value > 0 else PASTEL_GREEN for value in top_k_values]

        # Plot the absolute values of the weights
        plt.figure(figsize=(10, 4))
        plt.bar(x=top_k_names, height=np.abs(top_k_values), color=colors)
        plt.title(f'Non-Zero Predictors From Lasso Regression ({len(self.weights[non_zero_weights])} '
                  f'selected predictors)')
        plt.xlabel('Predictor Name')
        plt.xticks(rotation=90)
        plt.ylabel('Absolute Weight')

        # Create custom legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=PASTEL_BLUE, lw=4),
                        Line2D([0], [0], color=PASTEL_GREEN, lw=4)]
        plt.legend(custom_lines, ['Positive Weights', 'Negative Weights'])

        plt.show()
