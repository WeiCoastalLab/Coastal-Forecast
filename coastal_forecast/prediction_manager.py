# Created by Andrew Davison
# Will be used to call the model and make predictions
import pickle
import warnings

import numpy as np
from tensorflow.keras import backend
from tensorflow.python.keras.models import load_model

from coastal_forecast import data_manager as dm


def hello():
    return "Hello from Component prediction_manager"


def rmse(y_true, y_pred):
    """
    Calculates RMSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def mse(y_true, y_pred):
    """
    Calculates MSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: MSE.
    """
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


def r_square(y_true, y_pred):
    """
    Calculates R^2.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: R^2.
    """
    ssr = backend.sum(backend.square(y_true, y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - ssr / (sst + backend.epsilon())


def get_prediction(n_input, n_output):
    dataset = dm.fetch_data('41013')
    print('\nBack in get_prediction()...')
    print(dataset.info())
    print(dataset.head(5))
    print(dataset.head(-5))

    dependencies = {'r_square': r_square,
                    'rmse': rmse,
                    'mse': mse}

    model = load_model('../model/model.h5', custom_objects=dependencies)
    print(model.summary())
    data = dataset.drop('Time', axis=1)
    test = data.to_numpy()
    with open('../model/scalers.pkl', 'rb') as f:
        scaler_input, scaler_target = pickle.load(f)
    data_scaled = scaler_input.fit_transform(test[:, :3])
    target_scaled = scaler_target.fit_transform(test[:, 3:])
    data_scaled = np.column_stack((data_scaled, target_scaled))
    print(data.head())
    pass


if __name__ == '__main__':
    n_input, n_output = 9, 3
    get_prediction()
