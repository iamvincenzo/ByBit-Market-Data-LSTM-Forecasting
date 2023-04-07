import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import LSTMModel
from plotting_utils import Visualizer
from pytorchtools import EarlyStopping

class Solver(object):
    """ Initialize configurations. """
    def __init__(self, args, device, input_size, train_dataloader, val_dataloader, test_dataloader=None):
        super(Solver, self).__init__()
        self.args = args
        self.model_name = f'{self.args.model_name}.pth'        
        
        self.device = device
        self.input_size = input_size
        self.train_dataloader = train_dataloader
        self.val_dataloader= val_dataloader
        self.test_dataloader = test_dataloader

        self.vz = Visualizer()

        self.set_seed(42)

        """ from model import LSTM1
        # Model definition: non va bene con num-layers=4 perchè occorre reshape
        self.model = LSTM1(self.device, self.args.output_size, self.input_size, 
                           self.args.hidden_size, self.args.num_layers, self.args.seq_len).to(self.device)
        """
        
        # Model definition
        self.model = LSTMModel(self.device, self.input_size, self.args.hidden_size, 
                               self.args.num_layers, self.args.output_size).to(self.device)
        
        print(f'\nNetwork:\n\n {self.model}\n')

        for name, p in self.model.named_parameters():
            print('%-32s %s' % (name, tuple(p.shape)))

        # load a pretrained model
        if self.args.resume_train == True or self.args.make_prediction == True:
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

    """ Helper function. """
    def set_seed(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

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

        train_y_trues = np.array([], dtype=np.float64)
        train_preds = np.array([], dtype=np.float64)

        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.args.early_stopping, 
                                       verbose=True, path=check_path)
        early_stp = False

        self.model.train()

        for epoch in range(self.args.epochs): # loop over the dataset multiple times
            # record the training and test losses for each batch in this epoch
            train_losses = []
            test_losses = []

            loop = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader), leave=True)
                        
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

                train_y_trues = np.concatenate((y_batch.detach().numpy(), train_y_trues), axis=None)
                train_preds = np.concatenate((y_pred.detach().numpy(), train_preds), axis=None)
        
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
                    
                    self.plot_results(avg_train_losses, avg_test_losses, train_y_trues, train_preds)
                    
                    # print(f'\nGloabl-step: {epoch * len(self.train_dataloader) + batch_idx}')

                    train_losses = []
                    test_losses = []

                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(batch_avg_test_loss, self.model)

                    # evaluation on test-set
                    if self.test_dataloader is not None:
                         eval_loss, eval_metric = self.evaluate_on_test_set()
                         
                         print(f'\nEpoch: {epoch + 1}/{self.args.epochs}, ' + 
                               f'Batch: {batch_idx + 1}/{len(self.train_dataloader)}, ' +
                               f'eval_loss: {eval_loss:.4f}, ' +
                               f'eval_metric: {eval_metric:.4f}')

                    if early_stopping.early_stop:
                        print('\nEarly stopping...')
                        early_stp = True
                        break

            if early_stp:
                break

            # save at the end of each epoch only if earlystopping = False
            self.save_model()  
        
        print('\nTraining finished...')
        # self.plot_results(avg_train_losses, avg_test_losses, train_y_trues, train_preds) # ???
                    
    """ Evaluation of the model. """
    def test(self, test_losses):
        print('\nStarting the validation...')

        val_y_trues = np.array([], dtype=np.float64)
        val_preds = np.array([], dtype=np.float64)

        self.model.eval()  # put net into evaluation mode
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            test_loop = tqdm(enumerate(self.val_dataloader),
                             total=len(self.val_dataloader), leave=True)

            for _, (X_test, y_test) in test_loop:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)
                y_test_pred = self.model(X_test.detach())

                val_y_trues = np.concatenate((y_test.detach().numpy(), val_y_trues), axis=None)
                val_preds = np.concatenate((y_test_pred.detach().numpy(), val_preds), axis=None)

                test_loss = self.criterion(y_test_pred, y_test)     

                test_losses.append(test_loss.item())

            self.vz.plot_predictions(val_y_trues, val_preds, 'validation')           
    
        self.model.train()  # put again the model in trainining-mode

    """ Helper function used to evaluate the model in test set. """
    def evaluate_on_test_set(self):
        print('\nEvaluation on test set...')

        eval_y_trues = np.array([], dtype=np.float64)
        eval_preds = np.array([], dtype=np.float64)

        self.model.eval()

        with torch.no_grad():
            eval_losses = []
            metrics = []

            eval_loop = tqdm(enumerate(self.test_dataloader),
                             total=len(self.test_dataloader), leave=True)

            for _, (X_eval, y_eval) in eval_loop:
                X_eval = X_eval.to(self.device)
                y_eval = y_eval.to(self.device)
                y_eval_pred = self.model(X_eval.detach())

                eval_y_trues = np.concatenate((y_eval.detach().numpy(), eval_y_trues), axis=None)
                eval_preds = np.concatenate((y_eval_pred.detach().numpy(), eval_preds), axis=None)

                eval_loss = self.criterion(y_eval_pred, y_eval)                
                metric = torch.sqrt(eval_loss)     

                eval_losses.append(eval_loss.item())
                metrics.append(metric.item())

            avg_loss = np.average(eval_losses)
            avg_metric = np.average(metrics)

            self.vz.plot_predictions(eval_y_trues, eval_preds, 'evaluation')
        
        self.model.train()

        return avg_loss, avg_metric

    """ Helper function used to plot some results. """
    def plot_results(self, avg_train_losses, avg_test_losses, y_trues, predictions):
        print('\nPlotting losses...')

        self.vz.plot_loss(avg_train_losses, avg_test_losses)

        self.vz.plot_predictions(y_trues, predictions, 'training')

    """ Helper function to do. """
    def make_prediction(self, X):
        
        print(X)