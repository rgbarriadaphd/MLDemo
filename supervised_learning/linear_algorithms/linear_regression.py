"""
# Author = ruben
# Date: 24/9/24
# Project: MLDemo
# File: linear_regression.py

Description: Simple linear regression
"""
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class BaselineLogisticRegression:
    """Baseline Logistic Regression model from scikit learn"""
    def __init__(self):
        """Instance model """
        self._model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model on input data X and targets y"""
        X = X.reshape(-1, 1)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray, y: np.ndarray) -> float:
        """Predicts target values and returns the root mean squared error for provided samples."""
        X = X.reshape(-1, 1)
        y_predictions = self._model.predict(X)
        return sqrt(mean_squared_error(y, y_predictions))

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for the input samples X."""
        X = X.reshape(-1, 1)
        return self._model.predict(X)

    def get_model(self) -> dict:
        """Returns the current model parameters."""
        return {'w': self._model.coef_[0],
                'b': self._model.intercept_}


class CustomLogisticRegression:
    """Custom Logistic Regression model with gradient descent for parameter optimization."""

    def __init__(self, n_iter: int, lr: float):
        """Initializes the model with the specified number of iterations and learning rate."""
        self._model = self._init_model()
        self._n_iter = n_iter
        self._lr = lr

    def _init_model(self) -> dict:
        """Initializes model parameters."""
        return {'w': 0.0, 'b': 0.0}

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Calculates the predicted values for input data X, based on current model parameters."""
        return self._model['w'] * X + self._model['b']

    def _compute_cost(self, X: np.ndarray, y: np.ndarray)  -> float:
        """Computes the mean squared error cost function. Returns Mean squared error of predictions.

        .. math::
            J = \\frac{1}{m} \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2

        """
        y_prediction = self._predict(X)
        return np.mean((y - y_prediction) ** 2)

    def _update_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Updates model parameters using gradient descent.

        .. math::
                w = w - \\alpha \\frac{\\partial J}{\\partial w}
        .. math::
                b = b - \\alpha \\frac{\\partial J}{\\partial b}
        """
        y_pred = self._predict(X)

        # Compute gradients
        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)

        # Update parameters
        self._model['w'] = self._model['w'] - self._lr * dw
        self._model['b'] = self._model['b'] - self._lr * db

    def fit(self, X: np.ndarray, y: np.ndarray, verbosity: bool = False) -> None:
        """Trains the model on input data X and targets y for a given number of iterations."""
        for i in range(self._n_iter):
            loss = self._compute_cost(X, y)
            if verbosity and i % 100 == 0:
                print(f'Iteration: {i}, Loss: {loss} | model: {self._model}')
            self._update_params(X, y)

    def predict(self, X, y):
        """Predicts target values and returns the root mean squared error for provided samples."""

        y_pred = self._predict(X)
        return sqrt(mean_squared_error(y, y_pred))

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for the input samples X."""
        return self._predict(X)

    def get_model(self) -> dict:
        """Returns the current model parameters."""
        return self._model


import matplotlib.pyplot as plt
import numpy as np
def plot_model_comparison(train_df: pd.DataFrame,
                          baseline_model: dict, custom_model: dict, baseline_error: float, custom_error: float,
                          baseline_predictions: np.ndarray, custom_predictions: np.ndarray,
                          X_test_plot: pd.DataFrame, y_test_plot: pd.DataFrame) -> None:
    # create suplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Extract train data
    x_train = train_df['TV'].values
    y_train = train_df['Sales'].values

    # Extract model lines
    x_range = np.linspace(min(x_train), max(x_train), 100)
    baseline_line = baseline_model["w"] * x_range + baseline_model['b']
    custom_line = custom_model["w"] * x_range + custom_model['b']

    # set colors
    assert len(baseline_predictions) == len(custom_predictions)
    n_points = len(baseline_predictions)
    colors = plt.cm.get_cmap('tab10', n_points)

    # Baseline model
    axs[0].scatter(x_train, y_train, label='Train Points', color='black')
    axs[0].plot(x_range, baseline_line, label='Baseline Model', color='blue')
    for i, (x_test, y_real, y_pred) in enumerate(zip(X_test_plot, y_test_plot, baseline_predictions)):
        axs[0].scatter(x_test, y_pred, color=colors(i), label=f'Test {i+1} Prediction')
        axs[0].plot([x_test, x_test], [y_real, y_pred], color=colors(i), linestyle='--')  # Línea entre real y predicho
        print([x_test, x_test], [y_real, y_pred])
    axs[0].set_title(f'Baseline Model\nError: {baseline_error:.4f}')
    axs[0].legend()

    # Custom model
    axs[1].scatter(x_train, y_train, label='Train Points', color='black')
    axs[1].plot(x_range, custom_line, label='Custom Model', color='green')
    for i, (x_test, y_real, y_pred) in enumerate(zip(X_test_plot, y_test_plot, custom_predictions)):
        axs[1].scatter(x_test, y_pred, color=colors(i), label=f'Test {i+1} Prediction')
        axs[1].plot([x_test, x_test], [y_real, y_pred], color=colors(i), linestyle='--')  # Línea entre real y predicho

    axs[1].set_title(f'Custom Model\nError: {custom_error:.4f}')
    axs[1].legend()

    # Compare modele lines
    axs[2].scatter(x_train, y_train, label='Train Points', color='black')
    axs[2].plot(x_range, baseline_line, label='Baseline Model', color='blue')
    axs[2].plot(x_range, custom_line, label='Custom Model', color='green')

    axs[2].set_title('Comparison of Models')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load dataset
    df = pd.read_csv('../dataset/tvmarketing.csv')

    # split dataset into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df['TV'].values
    y_train = train_df['Sales'].values
    X_test = test_df['TV'].values
    y_test = test_df['Sales'].values

    # Baseline model
    baseline_regression = BaselineLogisticRegression()
    baseline_regression.fit(X_train, y_train)
    baseline_error = baseline_regression.predict(X_test, y_test)
    print(f"Baseline error: {baseline_error}")

    # Custom model
    custom_regression = CustomLogisticRegression(n_iter=1000, lr=0.00001)
    custom_regression.fit(X_train, y_train)
    custom_error = custom_regression.predict(X_test, y_test)
    print(f"Custom error: {custom_error}")

    # Random select test samples:
    n_samples = 5
    test_plot_df = test_df.sample(n_samples)
    X_test_plot = test_plot_df['TV'].values
    y_test_plot = test_plot_df['Sales'].values

    # Get baseline predictions subset and model
    baseline_predictions = baseline_regression.predict_samples(X_test_plot)
    baseline_model = baseline_regression.get_model()

    # Get custom predictions subset and model
    custom_predictions = custom_regression.predict_samples(X_test_plot)
    custom_model = custom_regression.get_model()

    print(custom_predictions)
    print(y_test_plot)

    # Compare models
    plot_model_comparison(train_df,
                          baseline_model, custom_model,
                          baseline_error, custom_error,
                          baseline_predictions, custom_predictions,
                          X_test_plot, y_test_plot)
