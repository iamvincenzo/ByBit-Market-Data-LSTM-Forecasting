import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

    def plot_predictions(self, y_true, predictions, t):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.title('Actual vs. Predicted Price (' + t + ')')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        # plt.show()
        fig.savefig('./data/pred_plot_' + t + '.png', bbox_inches='tight')

    """ Helper function. """
    def plot_loss(self, train_loss, valid_loss):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
        plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss)) + 1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('time')
        plt.ylabel('loss')
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig('./data/loss_plot.png', bbox_inches='tight')

    """ Helper function. """
    def plot_pca(self, data, t):
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data)

        # Plot PCA
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of ByBit data')
        plt.show()



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