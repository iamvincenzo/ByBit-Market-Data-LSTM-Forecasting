import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """ Initialize configurations. """
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

        self.init_weights()

    """ Method used to initialize network's weights. """
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    """ Method used to train the netwrok. """
    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)

        # return predictions[:, -1]
        return predictions 
        
# Define the LSTM model
class LSTMModel2(nn.Module):
    """ Initialize configurations. """
    def __init__(self, device, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel2, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #? .to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #? .to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        """ Se seq_length=60, batch_size=1407, e hidden_size=2, allora la forma dell'output della rete 
            LSTM sarebbe (1407, 60, 2), poiché abbiamo 1407 sequenze, ognuna di lunghezza 60 e ognuna con 
            uno hidden state di dimensione 2. Per l'operazione out[:, -1, :], stiamo selezionando solo 
            l'ultimo timestep per ciascuna sequenza, quindi la forma dell'output sarebbe (1407, 2). 
            Questo significa che stiamo selezionando l'ultimo hidden state per ciascuna delle 1407 sequenze, 
            e passando questi 1407 hidden state come input alla fully connected layer. """
        out = self.fc(out[:, -1, :])

        return out
