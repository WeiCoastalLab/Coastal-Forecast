# Created by Andrew Davison
import numpy as np
import pandas as pd
from numpy import array, split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend
from tensorflow.keras.models import load_model, Sequential

from coastal_forecast.data_manager import fetch_data
from coastal_forecast.results_manager import plot_results


# Custom loss functions, credit: https://github.com/keras-team/keras/issues/7947
# root mean squared error (rmse) for regression
def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates RMSE
    :param y_true: observed response
    :param y_pred: predicted response
    :return: RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates MSE
    :param y_true: observed response
    :param y_pred: predicted response
    :return: MSE
    """
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates R^2
    :param y_true: observed response
    :param y_pred: predicted response
    :return: R^2
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - ssr / (sst + backend.epsilon())


def r_square_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculates R^2 loss
    :param y_true: observed response
    :param y_pred: predicted response
    :return: 1 - R^2
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - (1 - ssr / (sst + backend.epsilon()))


def forecast(model: Sequential, history: list, n_inputs: int) -> np.array:
    """
    Conducts forecasting with the trained ML model on test data
    :param model: trained ML model to be used
    :param history: compiled historical data from test set
    :param n_inputs: number of inputs required
    :return: A prediction y_hat
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


def scale_data(data: np.array, n_outputs: int) -> (np.array, StandardScaler):
    """
    Scales the data using a Standard Scaler. Scales predictors and targets separately
    :param data: array of data to be scaled
    :param n_outputs: number of outputs from model
    :return: tuple of scaled array and target scaler
    """
    # instantiate scaler instances
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # scale predictors and targets
    data_scaled = input_scaler.fit_transform(data[:, :-n_outputs])
    target_scaled = target_scaler.fit_transform(data[:, -n_outputs:])

    return np.column_stack((data_scaled, target_scaled)), target_scaler


def get_prediction(station_id: str, n_inputs: int, n_outputs: int) -> None:
    """
    Scrapes new data from NOAA station and runs a new short term prediction for display in the application
    :param station_id: string of NOAA station identification number
    :param n_inputs: number of inputs for trained model
    :param n_outputs: number of outputs from trained model
    :return: None
    """
    dependencies = {'r_square': r_square,
                    'rmse': rmse,
                    'mse': mse}

    model = load_model(f'./model/{station_id}_model.h5', custom_objects=dependencies)

    print(f'Making new system prediction for station {station_id}:')

    dataset = fetch_data(station_id)
    data = dataset.drop('Time', axis=1)
    data = data.to_numpy()

    # scale the data
    data_scaled, target_scaler = scale_data(data, n_outputs)

    n_times = (dataset.shape[0] // n_outputs) - 1
    index_inuse = n_times * n_outputs
    data_split = array(split(data_scaled[-index_inuse:], n_times))
    test_1st = data_split[:len(data_split) - 24]
    test_2nd = data_split[-24:]
    ground_truth = dataset.iloc[-24 * n_outputs:]

    history = [x for x in test_1st]
    predictions = []
    for i in range(len(test_2nd)):
        y_hat_sequence = forecast(model, history, n_inputs)
        predictions.append(y_hat_sequence)
        history.append(test_2nd[i, :])
    predictions = array(predictions)

    post_processing(test_2nd, predictions, target_scaler, ground_truth, station_id, n_inputs, n_outputs)


def post_processing(y_true: np.array, y_pred: np.array, scalar_target: StandardScaler,
                    ground_truth: pd.DataFrame, station_id: str, n_inputs: int,
                    n_outputs: int, training: bool = False) -> (np.array, np.array) or None:
    """
    Conducts post-processing of prediction data and sends to plot results in results_manager
    :param y_true: array of ground truth values
    :param y_pred: array of predicted values
    :param scalar_target: StandardScaler for inverse transformation of data
    :param ground_truth: dataframe of truth values
    :param station_id: string of NOAA station identification number
    :param n_inputs: number of inputs used for model
    :param n_outputs: number of outputs from model
    :param training: boolean if results are from a training run
    :return: tuple of rescaled ground truth and predicted arrays if training, otherwise None
    """
    truth_squeezed = np.squeeze(y_true)
    pred_squeezed = np.squeeze(y_pred)
    truth_2d = truth_squeezed.reshape((truth_squeezed.shape[0] * truth_squeezed.shape[1], truth_squeezed.shape[2]))
    pred_2d = pred_squeezed.reshape((pred_squeezed.shape[0] * pred_squeezed.shape[1], pred_squeezed.shape[2]))

    truth_2d = scalar_target.inverse_transform(truth_2d[:, -n_outputs:])
    pred_2d = scalar_target.inverse_transform(pred_2d)

    # add predictions to ground truth dataframe
    ground_truth = ground_truth.assign(WVHT_LSTM=pred_2d[:, 0])
    ground_truth = ground_truth.assign(APD_LSTM=pred_2d[:, 1])
    ground_truth = ground_truth.assign(MWD_LSTM=pred_2d[:, 2])
    ground_truth = ground_truth.reset_index()
    if training is True:
        plot_results(ground_truth, station_id, f'../model/training_results/{station_id}_results.png',
                     n_inputs, n_outputs)
        return truth_2d, pred_2d
    else:
        plot_results(ground_truth, station_id,
                     f'coastal_forecast/static/{station_id}_system_prediction.png', n_inputs, n_outputs)
