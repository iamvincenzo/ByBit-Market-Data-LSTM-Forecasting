import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import LSTM1
from model import LSTMModel
from plotting_utils import Visualizer

class Solver(object):
    """ Initialize configurations. """
    def __init__(self, args, device, input_size, train_dataloader, test_dataloader):
        super(Solver, self).__init__()
        self.args = args
        self.model_name = f'{self.args.model_name}.pth'        
        
        self.device = device
        self.input_size = input_size
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.vz = Visualizer()

        # Model definition
        self.model = LSTM1(self.device, self.args.output_size, self.input_size, 
                           self.args.hidden_size, self.args.num_layers, self.args.seq_len).to(self.device)
        
        # # Model definition
        # self.model = LSTMModel(self.device, self.input_size, self.args.hidden_size, 
        #                        self.args.num_layers, self.args.output_size).to(self.device)

        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model(device)
        
        # Loss definition
        self.criterion = nn.MSELoss()

        # Optimizer definition
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), 
                                       lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.args.lr, betas=(0.9, 0.999))

    """ Helper function used to save the model. """
    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.model.state_dict(), check_path)
        print('\nModel saved!\n')


    """ Helper function used to load the model. """
    def load_model(self, device):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.model.load_state_dict(torch.load(check_path, 
                                            map_location=torch.device(device)))
        print('\nModel loaded!\n')

    """ Training function. """
    def train(self):
        print('\nStarting the training...')

        avg_train_losses = []
        avg_test_losses = []

        # trigger for earlystopping
        earlystopping = False

        self.model.train()

        for epoch in range(self.args.epochs): # loop over the dataset multiple times
            # record the training and test losses for each batch in this epoch
            train_losses = []
            test_losses = []

            loop = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader), leave=True)
            
            # print(f'\nEpoch {epoch + 1}/{self.args.epochs}\n')

            for batch_idx, (X_batch, y_batch) in loop:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device) 

                # Inizializzazione del gradiente
                self.optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(X_batch)

                # Calcolo della loss
                loss = self.criterion(y_pred, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Aggiornamento dei pesi
                self.optimizer.step()

                train_losses.append(loss.item())
        
                if batch_idx % self.args.print_every == self.args.print_every - 1:                    
                    # used to check model improvement
                    self.test(test_losses)

                    batch_avg_train_loss = np.average(train_losses)
                    batch_avg_test_loss = np.average(test_losses)

                    avg_train_losses.append(batch_avg_train_loss)
                    avg_test_losses.append(batch_avg_test_loss)

                    print(f'\nEpoch: {epoch + 1}/{self.args.epochs}, ' + 
                          f'Batch: {batch_idx + 1}/{len(self.train_dataloader)}, ' +
                          f'train-loss: {batch_avg_train_loss:.4f}, ' +
                          f'test-loss: {batch_avg_test_loss:.4f}')
                    
                    print(f'\nGloabl-step: {epoch * len(self.train_dataloader) + batch_idx}')

                    train_losses = []
                    test_losses = []

                    if epoch > self.args.early_stopping:  # Early stopping with a patience of 1 and a minimum of N epochs
                        if avg_test_losses[-1] >= avg_test_losses[-2]:
                            print('\nEarly Stopping Triggered With Patience 1')
                            self.save_model()  # save before stop training
                            earlystopping = True
                    if earlystopping:
                        break

            if earlystopping:
                break
        
        self.plot_results(avg_train_losses, avg_test_losses)
                    
    """ Evaluation of the model. """
    def test(self, test_losses):
        self.model.eval()  # put net into evaluation mode

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            test_loop = tqdm(enumerate(self.test_dataloader),
                             total=len(self.test_dataloader), leave=True)

            for _, (X_test, y_test) in test_loop:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)
                y_test_pred = self.model(X_test.detach())

                test_loss = self.criterion(y_test_pred, y_test).item()     

                test_losses.append(test_loss)           
    
        self.model.train()  # put again the model in trainining-mode

    """ Helper function used to plot some results. """
    def plot_results(self, avg_train_losses, avg_test_losses):
        print('\nPlotting losses...')

        self.vz.plot_loss(avg_train_losses, avg_test_losses)
