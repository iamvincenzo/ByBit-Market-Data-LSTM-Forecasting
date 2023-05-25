import torch
import torch.nn as nn
from torch.autograd import Variable 

""" The main difference between the two models is that they use different outputs of the LSTM network:
        - In the first model, LSTM1, the last hidden state hn is used as input for the fully connected layer. 
        The last hidden state hn contains the final state information of the LSTM network after processing the
        entire input sequence;
        
        - In the second model, LSTMModel, the complete output of the LSTM network is used as input for the fully 
        connected layer. Specifically, only the output of the last timestep, out[:, -1, :], is selected and passed 
        to the fully connected layer.
        
    In both cases, the fully connected layer is used to map the output of the LSTM network into a suitable form for 
    the classification task. However, the two models use slightly different approaches to obtain the input of the 
    fully connected layer. """

# non buona perchè devo prevedere un prezzo e non ha senso usare la funzione di attivazione
class LSTM1(nn.Module):
    """ Initialize configurations. """
    def __init__(self, device, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.device = device
        self.num_classes = num_classes # number of classes
        self.num_layers = num_layers # number of recurrent layers
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # hidden state
        self.seq_length = seq_length # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) # lstm
        # because of lstm-out-shape(seq_len, batch_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, 128) # fully connected 1
        self.fc = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        self._reinitialize()

    """ Tensorflow/Keras-like initialization. """
    def _reinitialize(self):    
        print('\nPerforming weights initialization...')
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    """ Method used to train the netwrok. """
    def forward(self, x):
        batch_size = x.size(0)
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, 
                                   self.hidden_size)).to(self.device) # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, 
                                   self.hidden_size)).to(self.device) # internal state
        # Propagate input through LSTM
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 

        # shape [num_layers * batch_size, hidden_size]
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first Dense
        out = self.relu(out) # relu
        out = self.fc(out) # Final Output

        return out
    
# Define the LSTM model
class LSTMModel(nn.Module):
    """ Initialize configurations. """
    def __init__(self, device, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # because of lstm-out-shape(seq_len, batch_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._reinitialize()

    """ Tensorflow/Keras-like initialization. """
    def _reinitialize(self):
        print('\nPerforming weights initialization...')
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    """ Method used to train the netwrok. """    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, 
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, 
                         self.hidden_size).to(self.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # get from all batches, all columns of the last row
        out = self.fc(out[:, -1, :])

        return out
    