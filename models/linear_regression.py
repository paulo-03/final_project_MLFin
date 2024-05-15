"""
This python script propose a class containing all useful method in a linear regression.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

PASTEL_BLUE = '#aec6cf'
PASTEL_GREEN = '#77dd77'


class OLS:
    def __init__(self,
                 predictors: pd.DataFrame,
                 label: pd.Series):
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
        plt.title(f'Top {top_k} Predictors From Lasso Regression')
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

    def alpha_cross_validation(self, from_: float = 0.01, to_: float = 10, val_number: int = 10):
        """Method that will try the val_number alphas between from_ and to_ values and test the Lasso regression using
        cross-validation to choose the best alpha possible."""
        # Initialize all the alphas to be cross-validated and the Lasso regressor
        alphas = np.linspace(from_, to_, val_number)
        lasso = Lasso(random_state=self.seed)
        # Initiate the mean and std scores list for all alphas
        mean_scores = []
        std_scores = []
        # Perform the cross-validation for all alphas
        for alpha in alphas:
            lasso.alpha = alpha
            scores = cross_val_score(lasso, self.X, self.y, cv=5)
            mean_scores.append(scores.mean())
            std_scores.append(scores.std())
        # Convert for ease of usage the list in array
        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)

        # Plotting the mean scores with 95% CI
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mean_scores, label='Mean CV Score')
        plt.fill_between(alphas, mean_scores - 1.96 * std_scores, mean_scores + 1.96 * std_scores, color='b', alpha=0.2,
                         label='95% CI')
        plt.title('Lasso Cross-Validation Scores for Different Alphas')
        plt.xlabel('Alpha')
        plt.ylabel('Mean CV Score')
        plt.legend()
        plt.show()

        best_alpha_index = np.argmax(mean_scores)
        self.alpha = alphas[best_alpha_index]
        print(f"Best alpha found: {self.alpha:.3f}. Simply run '.fit()' method to use this alpha value.")

    def fit(self):
        if self.alpha is None:
            raise ValueError(
                "Alpha value not set. Please set alpha or use alpha_cross_validation to find the best alpha.")
        self.reg = Lasso(alpha=self.alpha, random_state=self.seed).fit(X=self.X, y=self.y)
        self.r_square = self.reg.score(self.X, self.y)
        self.weights = self.reg.coef_

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
