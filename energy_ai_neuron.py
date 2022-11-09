# univariate multi-step lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
import pickle

# univariate multi-step lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

import pandas as pd
import re

dataset = read_csv('updated_data_full_dropped.csv', header=0,
                     infer_datetime_format=True, parse_dates=['dt'],
                     index_col=['dt'])

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[0:len(data)-7*24], data[len(data)-7*24:]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/24))
    test = array(split(test, len(test)/24))
    return train, test
train, test = split_dataset(dataset.values)

# validate train data
print(train.shape)
print(train[0, 12, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 12, 0], test[-1, -1, 0])

# print(train, test)

def show_plot(true, pred, title):
    fig = pyplot.subplots()
    pyplot.plot(true, label='Y_original')
    pyplot.plot(pred, dashes=[4, 3], label='Y_predicted')
    pyplot.xlabel('N_samples', fontsize=12)
    pyplot.ylabel('Instance_value', fontsize=12)
    pyplot.title(title, fontsize=12)
    pyplot.grid(True)
    pyplot.legend(loc='upper right')
    pyplot.show()

# # evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    # plot forecasts vs observations
    for j in range(predicted.shape[1]):
        show_plot(actual[:, j], predicted[:, j], j + 1)
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=24):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)

# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    pickle.dump(model, open(f'NN_model_wo_args.sav', 'wb'))
    return model



# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# evaluate a single model
def evaluate_model(train, test, n_input, model):
    # fit model
    
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 24
model = build_model(train, n_input)
pred = []

def evaluate_model(train, test, n_input, model):
    # fit model
    
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    pred = predictions
   
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    print(pred)
    return score, scores



score, scores = evaluate_model(train, test, n_input, model)

# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
hours = [n for n in range(1,25)]
fig = pyplot.subplots()
pyplot.plot(hours, scores, marker='o', label='lstm')
pyplot.show()

# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
hours = [n for n in range(1,25)]
fig = pyplot.subplots()
# pyplot.plot(hours, scores, marker='o', label='lstm')
print(pred)
print(test[0])

pyplot.plot(test[-1], label='Y_original')
pyplot.plot(pred[-1], dashes=[4, 3], label='Y_predicted')
pyplot.show()