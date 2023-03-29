import torch
import torch.nn as nn
from model import LSTM
import torch.optim as optim

class Solver(object):
    """ Initialize configurations. """
    def __init__(self, x, y):
        super(Solver, self).__init__()
        # Definizione dei dati di addestramento
        # 100 batch di sequenze di lunghezza 10 e 5 features
        self.x = x
        self.y = y

        # Definizione del modello, loss e ottimizzatore
        self.model = LSTM(input_size=5, hidden_size=64, num_layers=2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    """ Training function. """
    def train(self):
        # Addestramento del modello
        self.model.train()

        for epoch in range(100):
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(self.x)

            # Calcolo della loss
            loss = self.criterion(output, self.y)

            # Backward pass e aggiornamento dei pesi
            loss.backward()
            self.optimizer.step()

            # Stampa della loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    """ Evaluation of the model. """
    def test(self):
        self.model.eval()  # Imposta il modello in modalità di valutazione
        with torch.no_grad():
            # Forward pass
            output = self.model(self.x)

            # Calcolo della loss
            criterion = nn.MSELoss()
            loss = criterion(output, self.y)

            # Calcolo della metrica di valutazione (es. MAE, R^2, ecc.)
            metric = torch.sqrt(loss).item()

        return loss.item(), metric
    
    
""" Main function used to run the simulation. """
def main():
    x = torch.randn(100, 10, 5)
    y = torch.randn(100, 1)

    solver = Solver(x, y)

    print('\nTraining:\n')
    solver.train()

    print('\nTesting:\n')
    loss, metric = solver.test()
    print(f'Test Loss: {loss}, Test MAE: {metric}\n')


if __name__ == '__main__':
    main()