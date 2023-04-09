import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class CryptoDataset(Dataset):
    """ Initialize configurations. """
    def __init__(self, X_data, y_data, sequence_len):
        self.sequence_len = sequence_len
        self.X_data = X_data
        self.y_data = y_data

    """ Helper function. """
    def __getitem__(self, index):
        X = torch.FloatTensor(self.X_data[index : index + self.sequence_len])
        """ The target shape is [bs, 1] and not [bs, seq_len] because the LSTM model is being trained to 
            predict a single value, i.e., the next closing price of the cryptocurrency, given a sequence 
            of historical prices. """
        # y = torch.FloatTensor(np.array(self.y_data[index + self.sequence_len - 1]))
        y = torch.FloatTensor(np.array(self.y_data[index + self.sequence_len])) # non so quale sia corretto
        
        return X, y

    """ Helper function. """
    def __len__(self):
        return len(self.X_data) - self.sequence_len
    

class GetDataloader(object):
    """ Initialize configurations. """
    def __init__(self, X, y, bs_train, bs_test, max_batch_sz, workers, seq_len, split_perc):
        self.X = X
        self.y = y
        self.bs_train = bs_train
        self.bs_test = bs_test
        self.max_batch_sz = max_batch_sz
        self.workers = workers
        self.seq_len = seq_len
        self.split_perc = split_perc

    """ Helper function. """
    def get_dataloaders(self):
        train_size = int((len(self.X) * self.split_perc))

        # data-preprocessing
        #########################################
        X_train = self.X.iloc[:train_size, :]

        y_train = self.y.iloc[:train_size, :]

        X_test = self.X.iloc[train_size:, :]
        y_test = self.y.iloc[train_size:, :]

        ss = StandardScaler()
        mm = MinMaxScaler()

        X_train_ss = ss.fit_transform(X_train)
        y_train_mm = mm.fit_transform(y_train)

        X_test_ss = ss.transform(X_test)
        y_test_mm = mm.transform(y_test)
        ##########################################

        crypto_train_data = CryptoDataset(X_train_ss, y_train_mm, self.seq_len)       
        crypto_test_data = CryptoDataset(X_test_ss, y_test_mm, self.seq_len)
        
        n_items_train = len(crypto_train_data)
        print(f'\nNumber of items in training-set: {n_items_train}')
                
        n_items_test = len(crypto_test_data)
        print(f'Number of items in test-set: {n_items_test}')

        # hard-coded
        #############################
        if self.max_batch_sz == True:
            self.bs_train = n_items_train
            self.bs_test = n_items_test
        #############################

        train_dataloader = DataLoader(crypto_train_data, batch_size=self.bs_train, 
                                      num_workers=self.workers, shuffle=False)
        test_dataloader = DataLoader(crypto_test_data, batch_size=self.bs_test, 
                                     num_workers=self.workers, shuffle=False)
       
        
        return train_dataloader, test_dataloader, mm


class TimeSeriesSplitDataloader(object):
    """ Initialize configurations. """
    def __init__(self, X, y, seq_len, max_batch_sz, batch_size=64, val_test_split=0.5):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_batch_sz = max_batch_sz
        self.val_test_split = val_test_split

        tscv = TimeSeriesSplit()
        self.n_splits = min(6, tscv.get_n_splits(self.X))        

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        print(f'\nTimeSeriesSplit parameters: n_splits: {self.n_splits}, seq_len: {self.seq_len}')
    
    """ Helper function. """
    def get_dataloaders(self):
        for _, (train_index, val_index) in enumerate(self.tscv.split(self.X)):
            X_train, X_val_test = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val_test = self.y.iloc[train_index], self.y.iloc[val_index]
            # further split val_test into validation set and test set
            val_size = int(len(X_val_test) * self.val_test_split)
            X_val = X_val_test.iloc[:val_size]
            y_val = y_val_test.iloc[:val_size]
            X_test = X_val_test.iloc[val_size:]
            y_test = y_val_test.iloc[val_size:]
        
        # data-preprocessing
        #########################################
        ss = StandardScaler()
        mm = MinMaxScaler()

        X_train = ss.fit_transform(X_train)
        y_train = mm.fit_transform(y_train)

        X_val = ss.transform(X_val)
        y_val = mm.transform(y_val)

        X_test = ss.transform(X_test)
        y_test = mm.transform(y_test)
        #########################################

        crypto_train_data = CryptoDataset(X_train, y_train, self.seq_len)
        crypto_val_data = CryptoDataset(X_val, y_val, self.seq_len)
        crypto_test_data = CryptoDataset(X_test, y_test, self.seq_len)

        n_items_train = len(crypto_train_data)
        print(f'\nNumber of items in training-set: {n_items_train}')

        n_items_val = len(crypto_val_data)
        print(f'Number of items in validation-set: {n_items_val}')

        n_items_test = len(crypto_test_data)
        print(f'Number of items in test-set: {n_items_test}')

        if self.max_batch_sz == True:
            self.batch_size = n_items_train

        train_dataloader = DataLoader(crypto_train_data, batch_size=self.batch_size, shuffle=False)
        val_dataloader = DataLoader(crypto_val_data, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(crypto_test_data, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader, mm

        