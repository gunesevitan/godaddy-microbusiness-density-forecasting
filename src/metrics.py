import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def symmetric_mean_absolute_percentage_error(y_true, y_pred):

    """
    Calculate symmetric mean absolute percentage error from given ground-truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Array of ground-truth values

    y_pred: array-like of shape (n_samples)
        Array of prediction values

    Returns
    -------
    smape: float
        Symmetric mean absolute percentage error
    """

    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    return smape


def regression_scores(y_true, y_pred):

    """
    Calculate regression scores from given ground-truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Array of ground-truth values

    y_pred: array-like of shape (n_samples)
        Array of prediction values

    Returns
    -------
    scores: dict
        Dictionary of calculated scores
    """

    scores = {
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'smape': symmetric_mean_absolute_percentage_error(y_true, y_pred)
    }

    return scores
