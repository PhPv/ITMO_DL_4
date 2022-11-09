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


# load and clean-up data
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric

# # fill missing values with a value at the same time one day ago
# def fill_missing(values):
# 	one_day = 60 * 24
# 	for row in range(values.shape[0]):
# 		for col in range(values.shape[1]):
# 			if isnan(values[row, col]):
# 				values[row, col] = values[row - one_day, col]

# # load all data
# dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# # mark all missing values
# dataset.replace('?', nan, inplace=True)
# # make dataset numeric
# dataset = dataset.astype('float32')
# # fill missing
# fill_missing(dataset.values)
# # add a column for for the remainder of sub metering
# values = dataset.values
# dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# # save updated dataset
# dataset.to_csv('household_power_consumption.csv')


# # resample minute data to total for each day
# from pandas import read_csv
# # load the new file
# dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# # resample data to daily
# daily_groups = dataset.resample('D')
# daily_data = daily_groups.sum()
# # summarize
# print(daily_data.shape)
# print(daily_data.head())
# # save
# daily_data.to_csv('household_power_consumption_days.csv')


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

# # split a univariate dataset into train/test sets
# def split_dataset(data):
#     # split into standard weeks
#     train, test = data[1:-30*24+1], data[-30*24:-48]
#     # restructure into windows of weekly data
#     train = array(split(train, len(train)/24))
#     test = array(split(test, len(test)/24))
#     return train, test

# def show_plot(true, pred, title):
#     fig = pyplot.subplots()
#     pyplot.plot(true, label='Y_original')
#     pyplot.plot(pred, dashes=[4, 3], label='Y_predicted')
#     pyplot.xlabel('N_samples', fontsize=12)
#     pyplot.ylabel('Instance_value', fontsize=12)
#     pyplot.title(title, fontsize=12)
#     pyplot.grid(True)
#     pyplot.legend(loc='upper right')
#     pyplot.show()

# # # evaluate one or more weekly forecasts against expected values
# def evaluate_forecasts(actual, predicted):
#     scores = list()
#     # calculate an RMSE score for each day
#     for i in range(actual.shape[1]):
#         # calculate mse
#         mse = mean_squared_error(actual[:, i], predicted[:, i])
#         # calculate rmse
#         rmse = sqrt(mse)
#         # store
#         scores.append(rmse)
#     # calculate overall RMSE
#     s = 0
#     for row in range(actual.shape[0]):
#         for col in range(actual.shape[1]):
#             s += (actual[row, col] - predicted[row, col])**2
#     score = sqrt(s / (actual.shape[0] * actual.shape[1]))
#     # plot forecasts vs observations
#     for j in range(predicted.shape[1]):
#         show_plot(actual[:, j], predicted[:, j], j + 1)
#     return score, scores

# # summarize scores
# def summarize_scores(name, score, scores):
#     s_scores = ', '.join(['%.1f' % s for s in scores])
#     print('%s: [%.3f] %s' % (name, score, s_scores))

# # convert history into inputs and outputs
# def to_supervised(train, n_input, n_out=24):
#     # flatten data
#     data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
#     X, y = list(), list()
#     in_start = 0
#     # step over the entire history one time step at a time
#     for _ in range(len(data)):
#         # define the end of the input sequence
#         in_end = in_start + n_input
#         out_end = in_end + n_out
#         # ensure we have enough data for this instance
#         if out_end <= len(data):
#             x_input = data[in_start:in_end, 0]
#             x_input = x_input.reshape((len(x_input), 1))
#             X.append(x_input)
#             y.append(data[in_end:out_end, 0])
#         # move along one time step
#         in_start += 1
#     return array(X), array(y)

# # train the model
# def build_model(train, n_input):
#     # prepare data
#     train_x, train_y = to_supervised(train, n_input)
#     # define parameters
#     verbose, epochs, batch_size = 0, 70, 16
#     n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
#     # define model
#     model = Sequential()
#     model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs))
#     model.compile(loss='mse', optimizer='adam')
#     # fit network
#     model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     return model

# # make a forecast
# def forecast(model, history, n_input):
#     # flatten data
#     data = array(history)
#     data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
#     # retrieve last observations for input data
#     input_x = data[-n_input:, 0]
#     # reshape into [1, n_input, 1]
#     input_x = input_x.reshape((1, len(input_x), 1))
#     # forecast the next week
#     yhat = model.predict(input_x, verbose=0)
#     # we only want the vector forecast
#     yhat = yhat[0]
#     return yhat

# # evaluate a single model
# def evaluate_model(train, test, n_input):
#     # fit model
#     model = build_model(train, n_input)
#     # history is a list of weekly data
#     history = [x for x in train]
#     # walk-forward validation over each week
#     predictions = list()
#     for i in range(len(test)):
#         # predict the week
#         yhat_sequence = forecast(model, history, n_input)
#         # store the predictions
#         predictions.append(yhat_sequence)
#         # get real observation and add to history for predicting the next week
#         history.append(test[i, :])
#     # evaluate predictions days for each week
#     predictions = array(predictions)
#     score, scores = evaluate_forecasts(test[:, :, 0], predictions)
#     return score, scores

import pandas as pd
import re

# импорт данных
data_x = pd.read_csv('Station_1_weather_clear_06_06.csv', sep=';')
data_y = pd.read_csv('fact_year_original.csv', sep=',', index_col=0)

# удаление дупликатов по параметру дататайм, с оставлением последнего (второго) экземпляра
data_x = data_x.drop_duplicates(subset=['dt'], keep = 'last')

# преобразование даты и времени в тип данных дататайм64
data_x.loc[:,'dt'] = pd.to_datetime(data_x['dt'])
data_y.loc[:,'dt'] = pd.to_datetime(data_y['dt'])

# слияние матрицы признаков и матрицы целевых переменных
data = pd.merge(data_y, data_x, on='dt')
print(data.head())
# срез изключающий из выборки последние 10 дней для анализа
#data = pd.merge(data_y, data_x, on='dt').iloc[:-240]

# создание доп матрицы признаков из дататайм
dayofyear = data['dt'].dt.dayofyear
hours = data['dt'].dt.hour
month = data['dt'].dt.month
days_hours = pd.concat([dayofyear, hours, month], axis=1, join='inner', keys=['dayofyear', 'hours', 'month'])
print(days_hours.head())

# удаление явно ненужных столбцов из матрицы признаков
data = data.drop(['id', 'gtpp', 'load_time', 'predict'], axis=1)
data = pd.concat([data, days_hours], axis=1)
# data = data.drop(data[data['fact'].values <= 10].index)
# зимой станцию заметает снегом. 

data = data.drop(['month'], axis=1)

# очистка данных от лишних символов (не нужно) 
data.replace('[^a-zA-Z0-9]', ' ', regex = True)
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+:[C]:', '', x))

# выделение матриц признаков и целевых переменных (целевая переменная 'fact')
y_train = data[data.columns[0]]
X_train = data[data.columns[1:]]

# формирование набора для обучения и тестирования
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42)




# data.drop(data.tail(19).index,inplace=True) 
data.to_csv('data.csv', index=False)






# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0,
                    infer_datetime_format=True, parse_dates=['datetime'],
                    index_col=['datetime'])
# split into train and test
dataset = read_csv('data.csv', header=0,
                     infer_datetime_format=True, parse_dates=['dt'],
                     index_col=['dt'])

train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 24
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
fig = pyplot.subplots()
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()