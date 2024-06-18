import os
import argparse
import tensorflow as tf
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM,SimpleRNN
import numpy as np
from datetime import timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import secrets

import sys

# Get the selected option from command-line arguments
import os

# Set TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class StockPrediction:
    def __init__(self, ticker, start_date, validation_date, project_folder, epochs, time_steps, token, batch_size):
        self._ticker = ticker
        self._start_date = start_date
        self._validation_date = validation_date
        self._project_folder = project_folder
        self._epochs = epochs
        self._time_steps = time_steps
        self._token = token
        self._batch_size = batch_size

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, value):
        self._ticker = value

    def get_start_date(self):
        return self._start_date

    def set_start_date(self, value):
        self._start_date = value

    def get_validation_date(self):
        return self._validation_date

    def set_validation_date(self, value):
        self._validation_date = value

    def get_project_folder(self):
        return self._project_folder

    def set_project_folder(self, value):
        self._project_folder = value

    def get_epochs(self):
        return self._epochs

    def get_time_steps(self):
        return self._time_steps

    def get_token(self):
        return self._token

    def get_batch_size(self):
        return self._batch_size

class LongShortTermMemory:
    def __init__(self, project_folder, lstm_units=(50, 50, 50, 50), dropout_rates=(0.2, 0.2, 0.5, 0.5)):
        self.project_folder = project_folder
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates

    def get_defined_metrics(self):
        defined_metrics = [tf.keras.metrics.MeanSquaredError(name='MSE')]
        return defined_metrics

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
        return callback

    def create_model(self, x_train,model_type):
        if(model_type=="LSTM"):
            model = Sequential()

            for units, rate in zip(self.lstm_units, self.dropout_rates):
                model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(rate))

            model.add(LSTM(units=self.lstm_units[-1]))
            model.add(Dropout(self.dropout_rates[-1]))

            model.add(Dense(units=1))
            model.summary()
            return model
        else:
            model = Sequential()
            model.add(SimpleRNN(units=self.lstm_units[0], return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(self.dropout_rates[0]))
            for units, rate in zip(self.lstm_units[1:], self.dropout_rates[1:]):
                model.add(SimpleRNN(units=units, return_sequences=True))
                model.add(Dropout(rate))
            model.add(SimpleRNN(units=self.lstm_units[-1]))
            model.add(Dropout(self.dropout_rates[-1]))
            model.add(Dense(units=1))
            model.summary()
            return model
        
class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def get_stock_short_name(self):
        return self._sec.info['shortName']

    def get_min_max(self):
        return self._min_max

    def get_stock_currency(self):
        return self._sec.info['currency']

    def download_transform_to_numpy(self, time_steps, project_folder):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download([self._stock.get_ticker()], start=self._stock.get_start_date(), end=end_date)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))

        training_data = data[data['Date'] < self._stock.get_validation_date()].copy()
        test_data = data[data['Date'] >= self._stock.get_validation_date()].copy()
        training_data = training_data.set_index('Date')
        test_data = test_data.set_index('Date')

        train_scaled = self._min_max.fit_transform(training_data)
        self.__data_verification(train_scaled)

        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = self._min_max.fit_transform(inputs)

        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        original_price = latest_close_price

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            original_price = original_price + (original_price * random_slope)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)

        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []

        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data
                                                                        
class Plotter:
    def __init__(self, blocking, project_folder, short_name, currency, stock_ticker):
        self.blocking = blocking
        self.project_folder = project_folder
        self.short_name = short_name
        self.currency = currency
        self.stock_ticker = stock_ticker

    def plot_histogram_data_split(self, training_data, test_data, validation_date):
        print("plotting Data and Histogram")
        plt.figure(figsize=(12, 5))
        plt.plot(training_data.Close, color='green')
        plt.plot(test_data.Close, color='red')
        plt.ylabel('Price [' + self.currency + ']')
        plt.xlabel("Date")
        plt.legend(["Training Data", "Validation Data >= " + validation_date.strftime("%Y-%m-%d")])
        plt.title(self.short_name)
        plt.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_price.png'))

        fig, ax = plt.subplots()
        training_data.hist(ax=ax)
        fig.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_hist.png'))

        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_loss_and_metrics(self, history, metric_name, file_name):
        print(f"plotting {metric_name}")
        plt.plot(history.history[metric_name], label=metric_name)
        plt.plot(history.history[f'val_{metric_name}'], label=f'val_{metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name}/Validation {metric_name}')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.project_folder, f'{file_name}.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_mse(self, history):
        self.plot_loss_and_metrics(history, 'MSE', 'MSE')

    def plot_loss(self, history):
        self.plot_loss_and_metrics(history, 'loss', 'loss')

    def project_plot_predictions(self, price_predicted, test_data):
        print("plotting predictions")
        plt.figure(figsize=(14, 5))
        plt.plot(price_predicted[self.stock_ticker + '_predicted'], color='red', label='Predicted [' + self.short_name + '] price')
        plt.plot(test_data.Close, color='green', label='Actual [' + self.short_name + '] price')
        plt.xlabel('Time')
        plt.ylabel('Price [' + self.currency + ']')
        plt.legend()
        plt.title('Prediction')
        plt.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_prediction.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

class ReadmeGenerator:
  
    def __init__(self, project_folder, short_name):
        self.project_folder = project_folder
        self.short_name = short_name.strip().replace('.', '').replace(' ', '%20')

    def write(self):
        my_file = open(os.path.join(self.project_folder, 'README.md'), "w+")
        image_names = ['price', 'hist', 'prediction', 'MSE', 'loss']
        for name in image_names:
            my_file.write(f'![]({self.project_folder}/{self.short_name}_{name}.png)\n')


def train_LSTM_network(stock,model_type):
    data = StockData(stock)
    plotter = Plotter(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(stock.get_time_steps(), stock.get_project_folder())
    plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())

    lstm = LongShortTermMemory(stock.get_project_folder())
    model = lstm.create_model(x_train,model_type)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=stock.get_epochs(), batch_size=stock.get_batch_size(), validation_data=(x_test, y_test))
    print("saving weights")
    model.save(os.path.join(stock.get_project_folder(), 'model_weights.keras'))

    plotter.plot_loss_and_metrics(history, 'loss', 'loss')
    plotter.plot_mse(history)

    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
    test_predictions_baseline.index = test_data.index
    plotter.project_plot_predictions(test_predictions_baseline, test_data)

    BASE_URL = "https://raw.githubusercontent.com/raghavendra1207/TimeSeriesAnalysis/main/"  # Replace with your base URL
    generator = ReadmeGenerator(stock.get_project_folder(), data.get_stock_short_name())
    generator.write()

    print("prediction is finished")


import pandas as pd
from datetime import datetime
import sys

# Get the selected option from command-line arguments
if len(sys.argv) > 1:
    selected_option = sys.argv[1]
    if(selected_option=='google'):
        STOCK_TICKER='GOOG'
    elif(selected_option=='Bitcoin'):
        STOCK_TICKER='BTC-USD'
    elif(selected_option=='FTSE100'):
        STOCK_TICKER='^FTSE'
    elif(selected_option=='NIFTY 50'):
        STOCK_TICKER='^NSEI'
    elif(selected_option=='Tesla, Inc.'):
        STOCK_TICKER='TSLA'
else:
    STOCK_TICKER='^NSEI'

# Set your parameters directly in the notebook

STOCK_START_DATE = pd.to_datetime("2017-11-01")
STOCK_VALIDATION_DATE = pd.to_datetime("2021-09-01")
EPOCHS = 10
BATCH_SIZE = 10
TIME_STEPS = 3

TODAY_RUN = datetime.today().strftime("%Y%m%d")
TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
print('Ticker: ' + STOCK_TICKER)
print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
print('Test Run Folder: ' + TOKEN)
# create project run folder
PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
if not os.path.exists(PROJECT_FOLDER):
    os.makedirs(PROJECT_FOLDER)

stock_prediction = StockPrediction(STOCK_TICKER, 
                                   STOCK_START_DATE, 
                                   STOCK_VALIDATION_DATE, 
                                   PROJECT_FOLDER, 
                                   EPOCHS,
                                   TIME_STEPS,
                                   TOKEN,
                                   BATCH_SIZE)
# Execute Deep Learning model
model_type=input("Enter the Method name: ")
train_LSTM_network(stock_prediction,model_type)