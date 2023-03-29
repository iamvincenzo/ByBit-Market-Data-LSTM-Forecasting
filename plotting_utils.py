import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

class Visualizer:
    # , valid_data, train_loss, valid_loss, train_metric, valid_metric, predictions):
    def __init__(self, train_data):
        self.train_data = train_data
        """
        self.valid_data = valid_data
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_metric = train_metric
        self.valid_metric = valid_metric
        self.predictions = predictions
        """

    def plot_training_data(self):               
        # create a line plot using Plotly
        fig = px.line(self.train_data, x=self.train_data.index, y='close', title='Closing Price')
        fig.update_traces(line_color='#5070ff')
        # display the plot
        fig.show()
    
        """ Other methods
        # create plot
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.train_data.index, self.train_data['close'])

        # set x-axis label and ticks
        ax.set_xlabel('Date')
        ax.xaxis.set_tick_params(rotation=45)

        # set y-axis label
        ax.set_ylabel('Close Price')

        # set title
        ax.set_title('Close Price over Time')

        # show plot
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_data['index_col'], self.train_data['close'], label='Training Data')
        # plt.plot(self.valid_data['close'], label='Validation Data')
        # plt.title('Training and Validation Data')
        plt.title('Training Data')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()

        # create a sample DataFrame with a time-based index and a 'close' column
        # plot the 'close' column against time
        self.train_data['close'].plot()

        # set the plot title and axis labels
        plt.title('Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')

        # display the plot
        plt.show()
        """

    """ Other functions
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.valid_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_metric(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_metric, label='Training Metric')
        plt.plot(self.valid_metric, label='Validation Metric')
        plt.title('Training and Validation Metric')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.show()

    def plot_predictions(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.valid_data['Close'], label='Actual Price')
        plt.plot(self.predictions, label='Predicted Price')
        plt.title('Actual vs. Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()
    """






# class Visualizer:
#     def __init__(self, history, predictions):
#         self.history = history
#         self.predictions = predictions

#     def plot_loss(self):
#         loss = self.history['loss']
#         val_loss = self.history['val_loss']
#         epochs = range(1, len(loss) + 1)
#         plt.plot(epochs, loss, 'bo', label='Training loss')
#         plt.plot(epochs, val_loss, 'b', label='Validation loss')
#         plt.title('Training and validation loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.show()

#     def plot_metric(self):
#         metric_name = 'mean_absolute_error'
#         metric = self.history[metric_name]
#         val_metric = self.history['val_' + metric_name]
#         epochs = range(1, len(metric) + 1)
#         plt.plot(epochs, metric, 'bo', label='Training ' + metric_name)
#         plt.plot(epochs, val_metric, 'b', label='Validation ' + metric_name)
#         plt.title('Training and validation ' + metric_name)
#         plt.xlabel('Epochs')
#         plt.ylabel(metric_name)
#         plt.legend()
#         plt.show()

#     def plot_predictions(self):
#         close = np.array(self.predictions['close'])
#         future_close = np.array(self.predictions['future_close'])
#         plt.plot(close, label='Close')
#         plt.plot(future_close, label='Future close')
#         plt.title('Predictions')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.legend()
#         plt.show()
