import pickle
import timeit
from math import sqrt

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import array, split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend, regularizers
from tensorflow.python.keras.layers import ConvLSTM2D, Flatten, RepeatVector, LSTM, TimeDistributed, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

dpi = 300
matplotlib.rc("savefig", dpi=dpi)


# Custom loss functions, credit: https://github.com/keras-team/keras/issues/7947
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    """
    Calculates RMSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
    """
    Calculates MSE.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: MSE.
    """
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    """
    Calculates R^2.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: R^2.
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - ssr / (sst + backend.epsilon())


def r_square_loss(y_true, y_pred):
    """
    Calculates R^2 loss.

    :param y_true: observed response.
    :param y_pred: predicted response.
    :return: 1 - R^2
    """
    ssr = backend.sum(backend.square(y_true - y_pred))
    sst = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - (1 - ssr / (sst + backend.epsilon()))


def prep_data(filepath):
    """
    Reads in clean data file and prepares it for the ML model, with the required columns.

    :param filepath: filepath to the data set.
    :return: Dataframe of prepared data.
    """
    data = read_csv(filepath, header=0, infer_datetime_format=True, parse_dates=['Time'], index_col=['Time'])
    drop_cols = ['GST', 'ATMP', 'DPD', 'WTMP', 'DEWP']
    data.drop(drop_cols, inplace=True, axis=1)
    return data


def split_dataset(data, n_output):
    """
    Splits dataset into a standard-scaled train and test set.

    :param data: dataframe of data
    :param n_output: number of required outputs for the ML model.
    :return: tuple of train set, test set, target scaler, lower split index, upper split index.
    """
    print("Splitting data...")
    data = data.to_numpy()

    # define the scaler
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()
    data_scaled = scaler_input.fit_transform(data[:, :3])
    target_scaled = scaler_target.fit_transform(data[:, 3:6])
    with open('../model/scalers.pkl', 'wb') as scaler_f:
        pickle.dump([scaler_input, scaler_target], scaler_f)
    data_scaled = np.column_stack((data_scaled, target_scaled))

    # split into train and test
    split_idx = 38304
    n_times = (data.shape[0] - split_idx) // n_output
    split_idx_up = split_idx + n_times*n_output
    train, test, data_test = data_scaled[:split_idx], data_scaled[split_idx:split_idx_up], data[split_idx:split_idx_up]
    return array(split(train, len(train) / n_output)), array(split(test, len(test) / n_output)), scaler_target, split_idx, split_idx_up


def evaluate_forecasts(actual, predicted):
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


def summarize_scores(name, score, scores):
    """
    Summarizes scores for easy user reading.

    :param name: name of the type of model used.
    :param score: overall RMSE.
    :param scores: list of RMSE scores.
    :return: None
    """
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('{}: [{:.3f}] {}'.format(name, score, s_scores))


def to_supervised(train, n_input, n_output):
    """
    Creates training sets of predictor variables and response variables, incremented by a single time step.

    :param train: training dataset to be split.
    :param n_input: number of inputs for the model.
    :param n_output: number of outputs for the model.
    :return: tuple of arrays of predictor and response variables.
    """
    # flatten training_data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    train_x, train_y = [], []
    in_start = 0
    # step over training_data one step at a time
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_output
        # check for enough training_data for instance
        if out_end <= len(data):
            train_x.append(data[in_start:in_end, :])
            train_y.append(data[in_end:out_end, 3:6])
        # move along one step
        in_start += 1
    return array(train_x), array(train_y)


def build_model(train, n_input, n_output):
    """
    Builds and trains ConvLSTM2D ML model. Model is saved in a model folder and training history is saved to
    a training data folder.

    :param train: training dataset to be used in the model training.
    :param n_input: number of inputs required.
    :param n_output: number of outputs required.
    :return: The trained ML model.
    """
    print("Building model...")
    train_x, train_y = to_supervised(train, n_input, n_output)
    verbose, epochs, batch_size = 0, 50, 64
    n_features, n_outputs = train_x.shape[2], train_y.shape[1]
    # reshape into subsequences [samples, timesteps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], 1, 1, n_input, n_features))
    # split training data into training and validation sets
    train_X, val_X, train_Y, val_Y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

    # build the model
    model = Sequential()
    model.add(ConvLSTM2D(10, (1, 9), activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(l=0.01),
                         input_shape=(1, 1, n_input, n_features),
                         name="ConvLSTM2D_Layer"))
    model.add(Flatten(name="Flatten_Op"))
    model.add(RepeatVector(n_outputs, name="RepeatVector_Op"))
    model.add(LSTM(20, activation='relu',
                   return_sequences=True,
                   name="LSTM_Layer"))
    model.add(TimeDistributed(Dense(10, activation='relu',
                                    name="Dense_Layer"),
                              name="TimeDistributed_Op1"))
    model.add(TimeDistributed(Dense(n_outputs, name="Output_Layer")))
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt,
                  metrics=['accuracy', mse, r_square, rmse])
    model.summary()

    print("Fitting model...")
    # fit the model and capture history of performance
    fit_start = timeit.default_timer()
    history = model.fit(train_X, train_Y, epochs=epochs,
                        batch_size=batch_size, verbose=verbose,
                        validation_data=(val_X, val_Y))
    fit_end = timeit.default_timer() - fit_start
    print('Training time: {:.2f} seconds'.format(fit_end))
    # save history
    with open('../training_data/training_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save('../model/model.h5')
    return model


def forecast(model, history, n_input):
    """
    Conducts forecasting with the trained ML model on test data.

    :param model: trained ML model to be used.
    :param history: compiled historical data from test set.
    :param n_input: number of inputs required.
    :return: A prediction yhat.
    """
    # flatten training_data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input training_data
    input_x = data[-n_input:, :]
    # reshape into [1, 1, 1, 9, 6]
    input_x = input_x.reshape((1, 1, 1, len(input_x), 6))
    yhat = model.predict(input_x, verbose=0)
    return yhat


def evaluate_model(train, test, n_input, n_output, scaler_target):
    """
    Controls model testing workflow. Compiles a list of historical data (ground truth) and a
    list of predicted data (predictions), conducts post processing and calls evaluation method.

    :param train: training data to be used to train the model.
    :param test: test data to be used to test the trained model.
    :param n_input: number of inputs required.
    :param n_output: number of outputs required.
    :param scaler_target: scaler data to re-scale history and predictions.
    :return: tuple of evaluation scores from evaluation method.
    """
    model = build_model(train, n_input, n_output)
    pred_start = timeit.default_timer()
    history = [x for x in train]
    predictions = []

    print("Running predictions...")
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = array(predictions)
    pred_end = timeit.default_timer() - pred_start
    print('Prediction time: {:.2f} seconds'.format(pred_end))

    # post processing for evaluating the forecasts
    # squeeze and reshape test and prediction data to 2d arrays
    test_squeezed = np.squeeze(test)
    pred_squeezed = np.squeeze(predictions)
    test_squeezed = test_squeezed.reshape((test_squeezed.shape[0]*test_squeezed.shape[1], test_squeezed.shape[2]))
    pred_squeezed = pred_squeezed.reshape((pred_squeezed.shape[0] * pred_squeezed.shape[1], pred_squeezed.shape[2]))
    # perform inverse transform to scale results back to original scale
    test_2d = scaler_target.inverse_transform(test_squeezed[:, 3:6])
    pred_2d = scaler_target.inverse_transform(pred_squeezed)
    score, scores = evaluate_forecasts(test_2d, pred_2d)
    return score, scores, pred_2d


def plot_results(data_test):
    """
    Creates and saves a comparison plot of ground truth and predictions over time.

    :param data_test: dataframe of observed and predicted values
    :return: None
    """
    print("Plotting results...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), facecolor='w', edgecolor='k', sharex=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.15)
    data_test.plot(x='Time', y='WVHT', color='blue', label='NOAA Measurement', ax=ax1)
    data_test.plot(x='Time', y='WVHT_LSTM', color='red', label='LSTM Prediction', ax=ax1)
    ax1.legend();
    ax1.set_xlabel(' ')
    ax1.set_ylabel('WVHT (m)')
    title_WVHT = f'(a) Significant wave height (WVHT) comparison for {n_output}-hour prediction with {n_input}-hour input'
    ax1.set_title(title_WVHT)
    ax1.grid(b=True, which='major', color='#666666', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax1.get_legend().remove()

    data_test.plot(x='Time', y='APD', color='blue', label='NOAA Measurement', ax=ax2, legend=False)
    data_test.plot(x='Time', y='APD_LSTM', color='red', label='LSTM Prediction', ax=ax2, legend=False)
    ax2.legend();
    ax2.set_xlabel(' ')
    ax2.set_ylabel('APD (s)')
    title_APD = f'(b) Averaged wave period (APD) comparison for {n_output}-hour prediction with {n_input}-hour input'
    ax2.set_title(title_APD)
    ax2.grid(b=True, which='major', color='#666666', linestyle='-')
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax2.get_legend().remove()

    data_test.plot(x='Time', y='MWD', color='blue', label='NOAA Measurement', ax=ax3)
    data_test.plot(x='Time', y='MWD_LSTM', color='red', label='LSTM Prediction', ax=ax3)
    ax3.legend();
    ax3.set_xlabel('Time')
    title_MWD = f'(c) Mean Wave Direction (MWD) comparison for {n_output}-hour prediction with {n_input}-hour input'
    ax3.set_title(title_MWD)
    ax3.set_ylabel('MWD (degree)')
    ax3.grid(b=True, which='major', color='#666666', linestyle='-')
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax3.legend(loc="lower center", ncol=2)

    plt.savefig('../model/pred_results.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # identify number of inputs and outputs
    n_input, n_output = 9, 3

    # prepare the dataset
    dataset = prep_data('../training_data/41013_lt_clean.csv')  # change this filename when we successfully scrape new

    # split the dataset
    train, test, scaler, split_idx, split_idx_up = split_dataset(dataset, n_output)

    # run the model training and capture results
    score, scores, pred_2d = evaluate_model(train, test, n_input, n_output, scaler)

    # create dataset with ground truth values
    data_test = dataset.iloc[split_idx:split_idx_up]
    # add columns for predicted values
    data_test = data_test.assign(WVHT_LSTM=pred_2d[:, 0])
    data_test = data_test.assign(APD_LSTM=pred_2d[:, 1])
    data_test = data_test.assign(MWD_LSTM=pred_2d[:, 2])
    data_test = data_test.reset_index()
    # plot results
    plot_results(data_test)

    # summarize results for user
    summarize_scores('lstm', score, scores)
