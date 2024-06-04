"""
This python script propose a class containing all useful method in a random forest.
"""
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.tree import plot_tree


class RandomForest:
    def __init__(self, predictors, label):
        self.X = predictors
        self.y = label

        self.data_shape = self.X.shape
        self.n_estimators = 50
        self.max_depth = 3
        self.max_features = 1.0
        self.cpu_num = os.cpu_count() // 2  # set to "-1" if you want to use all cpu cores available
        self.r_square = None
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           max_features=self.max_features, random_state=0, n_jobs=self.cpu_num)

    def hyperparameter_tuning_with_crossvalidation(self, n_estimators: list = None,
                                                   max_depth: list = None,
                                                   max_features: list = None, cv_splits: int = 5):
        if n_estimators is None:
            n_estimators = [50, 300]
        if max_depth is None:
            max_depth = [None, 3]
        if max_features is None:
            max_features = [1.0, 'sqrt']

        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features
        }
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
        grid_search = GridSearchCV(self.model, param_grid, cv=kf, scoring='r2', n_jobs=self.cpu_num)
        grid_search.fit(self.X, self.y)

        # plot parameters score
        results = pd.DataFrame(grid_search.cv_results_)
        results = results[['param_n_estimators', 'param_max_depth', 'param_max_features', 'mean_test_score']]
        results = results.sort_values(by='mean_test_score', ascending=False)
        results = results.reset_index(drop=True)
        print(results)
        return grid_search.best_params_

    def set_model(self, n_estimators: list = 50, max_depth: list = 3, max_features: list = 1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                           random_state=0, n_jobs=self.cpu_num)

    def fit_predict_and_print_score(self):
        self.model.fit(self.X, self.y)
        print(f'With the following data:\n'
              f'- Data shape: {self.data_shape}\n'
              f'and the following hyperparameters:\n'
              f'- n_estimators: {self.n_estimators}\n'
              f'- max_depth: {self.max_depth}\n'
              f'- max_features: {self.max_features}')
        print('The R2 score is:', self.model.score(self.X, self.y))

    def plot_feature_importance(self):
        feature_importance = pd.Series(self.model.feature_importances_, index=self.X.columns).sort_values(
            ascending=False).head(20)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=feature_importance.index, y=feature_importance.values, color='lightblue')
        plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
        plt.title('Top 20 feature importance')
        plt.show()

    def plot_decision_tree(self, num_trees_to_plot: int = 3):
        for tree_in_forest in range(num_trees_to_plot):
            plt.figure(figsize=(20, 10))
            plot_tree(self.model.estimators_[tree_in_forest],
                      filled=True,
                      rounded=True,
                      feature_names=self.X.columns.tolist())
            plt.title(f'Tree number: {tree_in_forest + 1}')
            plt.show()

    def cross_val_score(self, n_jobs, cv: int = 5, scoring: str = 'r2'):
        return cross_val_score(self.model, self.X, self.y, cv=cv, n_jobs=n_jobs, scoring=scoring)
