import pickle
import timeit
from math import sqrt

import numpy as np
import pandas as pd
from numpy import array, split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers import ConvLSTM2D, Flatten, RepeatVector, LSTM, TimeDistributed, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

from coastal_forecast.prediction_manager import mse, r_square, rmse, forecast, post_processing


def prep_data(filepath: str) -> pd.DataFrame:
    """
    Reads in clean data file and prepares it for the ML model, with the required columns.

    :param filepath: filepath to the data set.
    :return: Dataframe of prepared data.
    """
    data = read_csv(filepath, header=0, infer_datetime_format=True, parse_dates=['Time'], index_col=['Time'])
    drop_cols = ['GST', 'DPD']
    data.drop(drop_cols, inplace=True, axis=1)

    return data


def split_dataset(dataset: pd.DataFrame, station_id: str, n_outputs: int) \
        -> (np.array, np.array, pd.DataFrame, StandardScaler):
    """
    Splits dataset into a standard-scaled train and test set.

    :param dataset: dataframe of data
    :param station_id: string of NOAA station identification number.
    :param n_outputs: number of required outputs for the ML model.
    :return: tuple of train set, test set, dataframe of ground truths, and target scaler.
    """
    print("Splitting data...")
    data = dataset.to_numpy()

    # define the scalar
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()
    data_scaled = scaler_input.fit_transform(data[:, :-n_outputs])
    target_scaled = scaler_target.fit_transform(data[:, -n_outputs:])
    with open(f'../model/scalers.pkl', 'wb') as scaler_f:
        pickle.dump([scaler_input, scaler_target], scaler_f)
    data_scaled = np.column_stack((data_scaled, target_scaled))

    # split into train and test
    split_idx = 38304
    n_times = (dataset.shape[0] - split_idx) // n_outputs
    split_idx_up = split_idx + n_times * n_outputs
    train, test = data_scaled[:split_idx], data_scaled[split_idx:split_idx_up]
    data_test = dataset.iloc[split_idx:split_idx_up]
    train, test = array(split(train, len(train) / n_outputs)), array(split(test, len(test) / n_outputs))

    return train, test, data_test, scaler_target


def evaluate_forecasts(actual: np.array, predicted: np.array) -> (float, list[float]):
    """
    Evaluates performance of ML model.

    :param actual: array of inverse scaled observed responses.
    :param predicted: array of inverse scaled predicted responses.
    :return: tuple of a list of RMSE scores and overall RMSE.
    """
    print("Evaluating forecasts...")
    scores = []

    # calculate an RMSE score for each
    for i in range(actual.shape[1]):
        # calculate rmse
        root_mse = mean_squared_error(actual[:, i], predicted[:, i], squared=False)
        # store
        scores.append(root_mse)

    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores


def summarize_scores(name: str, score: float, scores: list[float]) -> None:
    """
    Summarizes scores for easy user reading.

    :param name: name of the type of model used.
    :param score: overall RMSE.
    :param scores: list of RMSE scores.
    :return: None
    """
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('{}: [{:.3f}] {}'.format(name, score, s_scores))


def to_supervised(train: np.array, n_inputs: int, n_outputs: int) -> (np.array, np.array):
    """
    Creates training sets of predictor variables and response variables, incremented by a single time step.

    :param train: training dataset to be split.
    :param n_inputs: number of inputs for the model.
    :param n_outputs: number of outputs for the model.
    :return: tuple of arrays of predictor and response variables.
    """
    # flatten training_data
    data_tn = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))

    train_x, train_y = [], []
    in_start = 0

    # step over training data one step at a time
    for _ in range(len(data_tn)):
        in_end = in_start + n_inputs
        out_end = in_end + n_outputs

        # check for enough training_data for instance
        if out_end <= len(data_tn):
            train_x.append(data_tn[in_start:in_end, :])
            train_y.append(data_tn[in_end:out_end, -n_outputs:])
        # move along one step
        in_start += 1

    return array(train_x), array(train_y)


def build_model(train_x: np.array, train_y: np.array, n_inputs: int, n_outputs: int, station_id: str) -> Sequential:
    """
    Builds and trains ConvLSTM2D ML model. Model is saved in a model folder and training history is saved to
    a training data folder.

    :param train_x: training predictor variables dataset to be used in the model training.
    :param train_y: training response variables dataset to be used in the model training.
    :param n_inputs: number of inputs required.
    :param n_outputs: number of outputs required.
    :param station_id: string of NOAA station identification number.
    :return: The trained Sequential ML model.
    """
    print("Building model...")

    # identify model hyper-parameters and features
    verbose, epochs, batch_size = 0, 50, 64
    n_features = train_x.shape[2]

    # reshape into subsequences [samples, time-steps, rows, cols, features]
    train_x = train_x.reshape((train_x.shape[0], 1, 1, n_inputs, n_features))

    # split training data into training and validation sets
    train_X, val_X, train_Y, val_Y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)  # noqa

    # build the model
    model = Sequential()  # this is NOW the same model as used in the manuscript
    model.add(ConvLSTM2D(50, (1, 3), activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l=0.01),
                         input_shape=(1, 1, n_inputs, n_features),
                         name="ConvLSTM2D_Layer"))
    model.add(Flatten(name="Flatten_Op"))
    model.add(RepeatVector(n_outputs, name="RepeatVector_Op"))
    model.add(LSTM(50, activation='relu',
                   return_sequences=True,
                   name="LSTM_Layer"))
    model.add(TimeDistributed(Dense(30, activation='relu',
                                    name="Dense_Layer"),
                              name="TimeDistributed_Op1"))
    model.add(TimeDistributed(Dense(n_outputs, name="Output_Layer")))
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt,
                  metrics=['accuracy', mse, r_square, rmse])

    # old model architecture
    # model.add(ConvLSTM2D(10, (1, 9), activation='relu',
    #                      kernel_initializer='he_normal',
    #                      kernel_regularizer=regularizers.l2(l=0.01),
    #                      input_shape=(1, 1, n_inputs, n_features),
    #                      name="ConvLSTM2D_Layer"))
    # model.add(Flatten(name="Flatten_Op"))
    # model.add(RepeatVector(n_outputs, name="RepeatVector_Op"))
    # model.add(LSTM(20, activation='relu',
    #                return_sequences=True,
    #                name="LSTM_Layer"))
    # model.add(TimeDistributed(Dense(10, activation='relu',
    #                                 name="Dense_Layer"),
    #                           name="TimeDistributed_Op1"))
    # model.add(TimeDistributed(Dense(n_outputs, name="Output_Layer")))
    # opt = Adam(learning_rate=0.001)
    # model.compile(loss='mse', optimizer=opt,
    #               metrics=['accuracy', mse, r_square, rmse])
    model.summary()

    # fit the model and capture history of performance
    print("Fitting model...")
    fit_start = timeit.default_timer()
    history = model.fit(train_X, train_Y, epochs=epochs,
                        batch_size=batch_size, verbose=verbose,
                        validation_data=(val_X, val_Y))
    fit_end = timeit.default_timer() - fit_start
    print('Training time: {:.2f} seconds'.format(fit_end))

    # save history
    with open(f'../training_data/training_history/{station_id}_training_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(f'../model/{station_id}_model.h5')

    return model


def run_predictions(model: Sequential, train: np.array, test: np.array, n_inputs: int) -> np.array:
    """
    Compiles predictions through trained ML model.

    :param model: trained ML model to calculate predictions.
    :param train: numpy array of training data.
    :param test: numpy array of test data.
    :param n_inputs: number of inputs needed for ML model.
    :return: numpy array of predictions obtained from trained model.
    """
    pred_start = timeit.default_timer()
    history = [x for x in train]
    predictions = []

    print('Running predictions...')
    for i in range(len(test)):
        y_hat = forecast(model, history, n_inputs)
        predictions.append(y_hat)
        history.append(test[i, :])
    predictions = array(predictions)
    pred_end = timeit.default_timer() - pred_start
    print('Prediction time: {:.2f} seconds'.format(pred_end))

    return predictions


def train_model(station_id: str, n_inputs: int, n_outputs: int) -> None:
    """
    Controls workflow of training a ML model for a specific NOAA Station information.

    :param station_id: string of NOAA station identification number.
    :param n_inputs: number of inputs to train the model on.
    :param n_outputs: number of outputs to predict.
    :return: None
    """
    # prepare the data for training
    dataset = prep_data(f'../training_data/{station_id}_lt_clean.csv')

    # split the data into train and test sets, will need the unscaled ground truths and scaler for the targets
    train, test, data_test, scaler_target = split_dataset(dataset, station_id, n_outputs)

    # split the training set into predictor and response variable datasets
    train_x, train_y = to_supervised(train, n_inputs, n_outputs)

    # build and fit the ML model
    model = build_model(train_x, train_y, n_inputs, n_outputs, station_id)

    # run predictions on the model
    predictions = run_predictions(model, train, test, n_inputs)

    # post processing to prep for plotting results
    test_2d, pred_2d = post_processing(test, predictions, scaler_target, data_test,
                                       station_id, n_inputs, n_outputs, True)

    # summarize results from training
    score, scores = evaluate_forecasts(test_2d, pred_2d)
    summarize_scores('LSTM', score, scores)


# this will eventually be moved to the system timer file, where we can run annually, for now leave as a runnable script
if __name__ == "__main__":
    stations = ['41013']  # '41008', '41009', '41013', '44013']
    stations = ['41008', '41009', '44013']
    for station in stations:
        train_model(station, 9, 3)
