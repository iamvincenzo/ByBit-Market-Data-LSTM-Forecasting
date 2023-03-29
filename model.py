import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM1(nn.Module):
    def __init__(self, device, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.device = device
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #? .to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #? .to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

"""
def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('./market-data.csv', index_col='date', parse_dates=True)

    df.drop(['symbol', 'interval', 'open_time',
            'turnover'], inplace=True, axis=1)

    columns_titles = ['open', 'high', 'low', 'volume', 'close']
    df = df.reindex(columns=columns_titles)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    mm = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    train_size = int((len(df) * 0.67))

    X_train = X_ss[:train_size, :]
    y_train = y_mm[:train_size, :]

    X_test = X_ss[train_size:, :]
    y_test = y_mm[train_size:, :]

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape)

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # reshaping to rows, timestamps, features
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1,
                                                            X_train_tensors.shape[1]))

    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1,
                                                          X_test_tensors.shape[1]))

    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

    num_epochs = 500 # number of epochs
    learning_rate = 0.001 # learning-rate
    input_size = 4 # number of features
    hidden_size = 2 # number of features in hidden state
    num_layers = 1 # number of stacked lstm layers
    num_classes = 1 #number of output classes

    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) # our lstm class

    print(lstm1)

    criterion = torch.nn.MSELoss() # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

    lstm1.train()

    for epoch in range(num_epochs):
        outputs = lstm1.forward(X_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward() #calculates the loss of the loss function

        optimizer.step() #improve from loss, i.e backprop

        if epoch % 100 == 0:
            print("Epoch: %d, train-loss: %1.5f" % (epoch, loss.item()))

            lstm1.eval()

            # utilizza X_test_tensors_final per testare il modello
            with torch.no_grad():
                test_outputs = lstm1(X_test_tensors_final)
                loss = criterion(outputs, y_train_tensors)

                print(f'test-loss: {loss.item():.5f}')

            lstm1.train()


    # testing with train-data
    df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
    df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers

    df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
    df_y_mm = Variable(torch.Tensor(df_y_mm))

    # reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

    train_predict = lstm1(df_X_ss) # forward pass
    data_predict = train_predict.data.numpy() # numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)

    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=200, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

    # testing with test data
    lstm1.eval()

    # utilizza X_test_tensors_final per testare il modello
    with torch.no_grad():
        test_outputs = lstm1(X_test_tensors_final)
        loss = criterion(outputs, y_train_tensors)

    print(loss.item())

    # salva i pesi addestrati
    torch.save(lstm1.state_dict(), 'model_weights.pth')

    # carica i pesi addestrati
    lstm1.load_state_dict(torch.load('model_weights.pth'))


    # df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
    # df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers

    # df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
    # df_y_mm = Variable(torch.Tensor(df_y_mm))

    # # reshaping the dataset
    # df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

    train_predict = lstm1(X_test_tensors_final) # forward pass
    data_predict = train_predict.data.numpy() # numpy conversion
    dataY_plot = y_test_tensors.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)

    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=200, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
"""