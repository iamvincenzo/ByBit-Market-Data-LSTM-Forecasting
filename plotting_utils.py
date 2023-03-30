import plotly.express as px
import matplotlib.pyplot as plt

class Visualizer(object):   
    """ Initialize configurations. """
    def __init__(self):
        pass

    """ Helper function. """
    def plot_data(self, data, t):               
        # create a line plot using Plotly
        fig = px.line(data, x=data.index, y='close', title='Closing Price: ' + t)
        fig.update_traces(line_color='#5070ff')
        # display the plot
        fig.show()

    """ Helper function. """
    def plot_loss(self, train_loss, valid_loss):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
        plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('./data/loss_plot.png', bbox_inches='tight')

    """ Other functions
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

# def __init__(self) #, train_data, valid_data, train_loss, valid_loss): #, train_metric, valid_metric, predictions):
#         # self.train_data = train_data
#         # self.valid_data = valid_data
#         # self.train_loss = train_loss
#         # self.valid_loss = valid_loss
#         # self.train_metric = train_metric
#         # self.valid_metric = valid_metric
#         # self.predictions = predictions