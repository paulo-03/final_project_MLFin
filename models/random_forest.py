"""
This python script propose a class containing all useful method in a random forest.
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.tree import plot_tree


class RandomForest:
    def __init__(self, data: pd.DataFrame, target: str = 'return'):
        self.target = target

        # data['date'] = pd.to_datetime(data['date'])
        self.X = data.drop(columns=[target])
        self.y = data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0, shuffle=True)

        self.data_shape = data.shape
        self.n_estimators = 50
        self.max_depth = 3
        self.max_features = 1.0
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           max_features=self.max_features, random_state=0)

    def hyperparameter_tuning_with_crossvalidation(self, n_jobs, verbose, n_estimators: list = None,
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
        grid_search = GridSearchCV(self.model, param_grid, cv=kf, scoring='r2', n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(self.X, self.y)
        return grid_search.best_params_

    def set_model(self, n_jobs, verbose, n_estimators: list = 50, max_depth: list = 3, max_features: list = 1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                           random_state=0, verbose=verbose, n_jobs=n_jobs)

    def fit_predict_and_print_score(self):
        self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(self.X_test)
        print(f'With the following data:\n'
              f'- Data shape: {self.data_shape}\n'
              f'- Target: {self.target}\n'
              f'and the following hyperparameters:\n'
              f'- n_estimators: {self.n_estimators}\n'
              f'- max_depth: {self.max_depth}\n'
              f'- max_features: {self.max_features}')
        print('The R2 score is:', r2_score(self.y_test, prediction))
        print('The MSE is:', mean_squared_error(self.y_test, prediction))

    def plot_feature_importance(self):
        feature_importance = pd.Series(self.model.feature_importances_, index=self.X.columns).sort_values(
            ascending=False)
        plt.figure(figsize=(20, 10))
        sns.barplot(x=feature_importance.index, y=feature_importance.values)
        plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
        plt.title('Median Correlation')
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
