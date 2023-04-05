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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        """ Se seq_length=60, batch_size=1407 e hidden_size=2, allora l'hidden state hn 
            restituito dalla chiamata self.lstm(x, (h_0, c_0)) avrà una shape di 
            (num_layers, batch_size, hidden_size) = (1, 1407, 2), dove num_layers è il numero
            di strati della rete LSTM. La successiva riga di codice, hn = hn.view(-1, self.hidden_size), 
            riformatta l'hidden state hn in un tensore di shape (batch_size, hidden_size) = (1407, 2) 
            per essere passato alla fully connected layer. """
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output

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
    