# Created by Andrew Davison
# Will be used to call the model and make predictions
import pickle

import numpy as np
import pandas as pd
from numpy import array, split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend
from tensorflow.python.keras.models import load_model, Sequential

from coastal_forecast.data_manager import fetch_data
from coastal_forecast.results_manager import plot_results


def hello():
    return "Hello from Component prediction_manager"


# Custom loss functions, credit: https://github.com/keras-team/keras/issues/7947
# root mean squared error (rmse) for regression
def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates RMSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates MSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: MSE.
    """
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates R^2.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: R^2.
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - ssr / (sst + backend.epsilon())


def r_square_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates R^2 loss.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: 1 - R^2
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - (1 - ssr / (sst + backend.epsilon()))


def forecast(model: Sequential, history: list, n_inputs: int) -> np.array:
    """
    Conducts forecasting with the trained ML model on test data.

    :param model: trained ML model to be used.
    :param history: compiled historical data from test set.
    :param n_inputs: number of inputs required.
    :return: A prediction y_hat.
    """
    # flatten training_data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

    # retrieve last observations for input
    input_x = data[-n_inputs:, :]
    n_features = input_x.shape[1]

    # reshape into [sample, time-step, row, cols, features]
    input_x = input_x.reshape((1, 1, 1, len(input_x), n_features))
    y_hat = model.predict(input_x, verbose=0)

    return y_hat


def get_prediction(station_id: str, n_inputs: int, n_outputs: int) -> None:
    """
    Scrapes new data from NOAA station and runs a new short term prediction for display in the application.

    :param station_id: string of NOAA station identification number.
    :param n_inputs: number of inputs for trained model.
    :param n_outputs: number of outputs from trained model.
    :return: None
    """
    dataset = fetch_data(station_id)
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
    data_scaled = scaler_input.fit_transform(test[:, :-n_outputs])
    target_scaled = scaler_target.fit_transform(test[:, -n_outputs:])
    data_scaled = np.column_stack((data_scaled, target_scaled))
    print(data.head())
    n_times = (dataset.shape[0] // n_outputs) - 1
    index_inuse = n_times * n_outputs
    data_split = array(split(data_scaled[-index_inuse:], n_times))
    test_1st = data_split[:len(data_split) - 10]
    test_2nd = data_split[-10:]
    actual_test = dataset.iloc[-10 * n_outputs:]

    history = [x for x in test_1st]
    predictions = []
    for i in range(len(test_2nd)):
        y_hat_sequence = forecast(model, history, n_input)
        predictions.append(y_hat_sequence)
        history.append(test_2nd[i, :])
    predictions = array(predictions)

    post_processing(test_2nd, predictions, scaler_target, actual_test, station_id, n_inputs, n_outputs)
    

def post_processing(y_true: np.array, y_pred: np.array, scalar_target: StandardScaler,
                    actual_test: pd.DataFrame, station_id: str, n_inputs: int,
                    n_outputs: int, training: bool = False) -> (np.array, np.array) or None:
    """
    Conducts post processing of prediction data and sends to plot results in results_manager.

    :param y_true: array of ground truth values.
    :param y_pred: array of predicted values.
    :param scalar_target: StandardScaler for inverse transformation of data
    :param actual_test: dataframe of truth values.
    :param station_id: string of NOAA station identification number.
    :param n_inputs: number of inputs used for model.
    :param n_outputs: number of outputs from model.
    :param training: boolean if results are from a training run.
    :return: tuple of rescaled ground truth and predicted arrays if training, otherwise None.
    """
    truth_squeezed = np.squeeze(y_true)
    pred_squeezed = np.squeeze(y_pred)
    truth_2d = truth_squeezed.reshape((truth_squeezed.shape[0] * truth_squeezed.shape[1], truth_squeezed.shape[2]))
    pred_2d = pred_squeezed.reshape((pred_squeezed.shape[0] * pred_squeezed.shape[1], pred_squeezed.shape[2]))
    
    truth_2d = scalar_target.inverse_transform(truth_2d[:, -n_outputs:])
    pred_2d = scalar_target.inverse_transform(pred_2d)

    actual_test = actual_test.assign(WVHT_LSTM=pred_2d[:, 0])
    actual_test = actual_test.assign(APD_LSTM=pred_2d[:, 1])
    actual_test = actual_test.assign(MWD_LSTM=pred_2d[:, 2])
    actual_test = actual_test.reset_index()
    if training is True:
        plot_results(actual_test, station_id, f'../model/{station_id}_pred_results.png', n_inputs, n_outputs, training)
        return truth_2d, pred_2d
    else:
        plot_results(actual_test, station_id, f'static/{station_id}_system_prediction_.png', n_inputs, n_outputs)


if __name__ == '__main__':
    n_input, n_output = 9, 3
    get_prediction('41013', n_input, n_output)
