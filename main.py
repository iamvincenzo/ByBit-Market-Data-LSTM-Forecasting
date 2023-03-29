import os
import json
import torch    
import argparse
import numpy as np
import configparser
import pandas as pd
from HttpRequest import HTTPRequest
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from solver import Solver
from plotting_utils import Visualizer
from data_utils import CryptoDataset


class KlineRequest(HTTPRequest):
    """ Initialize configurations. """
    def __init__(self, symbol, interval, startTime, url, endpoint, method, params, api_key, api_secret, Info):
        super().__init__(url, endpoint, method, params, api_key, api_secret, Info)
        self.symbol = symbol
        self.interval = interval
        self.startTime = startTime

    """ Helper function used to save data locally. """
    def save_df_to_csv(self, path):
        self.data = self.data.rename_axis('date')
        self.data.to_csv(path, index=True)

    """ Helper function used to obtain candles data. """
    def get_bybit_bars(self, startTime, endTime):
        startTime = str(int(startTime.timestamp()))
        endTime = str(int(endTime.timestamp()))

        self.params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'from': startTime,
            'to': endTime
        }

        response = self.HTTP_Request()
        # print(response.text)
        
        df = pd.DataFrame(json.loads(response.text)['result'])

        if (len(df.index) == 0):
            print('\nNone information...\n')
            return None

        df.index = [datetime.fromtimestamp(x) for x in df.open_time]

        return df

    """ Helper function used to obtain a dataframe
        containing cndles market data. """
    def get_kline_bybit(self):
        df_list = []
        last_datetime = self.startTime
        while True:
            print(last_datetime)
            new_df = self.get_bybit_bars(last_datetime, datetime.now())
            if new_df is None:
                break
            df_list.append(new_df)
            last_datetime = max(new_df.index) + timedelta(0, 1)

        df = pd.concat(df_list)
        
        self.data = df
        self.save_df_to_csv('./data/market-data.csv')

        return df


""" Helper function used to get cmd parameters. """
def get_args():
    parser = argparse.ArgumentParser()

    # options
    ###################################################################
    parser.add_argument('--download_data', action='store_true', #default=True,
                        help='starts an ablation study')
    parser.add_argument('--plot_data', action='store_true', #default=True,
                        help='starts an ablation study')
    parser.add_argument('--train_model', action='store_true', #default=True,
                        help='starts an ablation study')
    parser.add_argument('--make_prediction', action='store_true', default=True,
                        help='starts an ablation study')
    ###################################################################

    # model/data infos
    ###################################################################
    parser.add_argument('--run_name', type=str,
                        default="run_0", help='name of current run')
    parser.add_argument('--model_name', type=str, default="lstm_bybit_analysis",
                        help='name of the model to be saved/loaded')
    parser.add_argument('--dataset_path', type=str, default='./data/market-data.csv',
                        help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./model_save', help='path were to save the trained model')
    parser.add_argument('--resume_train', action='store_true',
                        help='load the model from checkpoint before training')
    ###################################################################

    # network-training
    ###################################################################
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--bs_train', type=int, default=1407, # 16,
                        help='number of elements in training batch')
    parser.add_argument('--bs_test', type=int, default=570, #16,
                        help='number of elements in test batch')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of workers in dataloader')
    parser.add_argument('--early_stopping', type=int, default=5,
                    help='early stopping epoch treshold')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', 
                        choices=['SGD', 'Adam'], 
                        help='optimizer used for training')
    parser.add_argument('--split_perc', type=float, default=0.7,
                        help='learning rate')
    ###################################################################

    # network-architecture parameters
    ###################################################################
    parser.add_argument('--seq_len', type=int, default=60,
                        help='number of epochs')
    parser.add_argument('--hidden_size', type=int, default=2,
                        help='number of elements in training batch')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of elements in test batch')
    parser.add_argument('--output_size', type=int, default=1,
                        help='number of workers in data loader')
    ###################################################################

    # output-threshold
    ###################################################################
    parser.add_argument('--print_every', type=int, default=1, # 8,
                    help='print losses every N iteration')
    ###################################################################


    return parser.parse_args()


""" Main method used to run the simulation. """
def main(args):
    print('\nExectuion of main function...')

    if args.download_data == True:
        print('\nDownload data...')

        ########################## CONFIG-INFO ##########################
        config = configparser.ConfigParser()
        config.read('api_config.cfg')
        api_key = config['BYBIT']['api_key']
        api_secret = config['BYBIT']['api_secret']

        config = configparser.ConfigParser()
        config.read('net_config.cfg')
        url = config['URL']['url']
        endpoint = config['ENDPOINT']['endpoint']

        request_type = config['REQUEST']['request_type']
        symbol = config['PARAM']['symbol']
        interval = config['PARAM']['interval']

        method = config['METHOD']['method']
        #################################################################

        if request_type == 'kline':
            last_datetime = config['LASTDATETIME']['last_datetime']
            datetime_object = datetime.strptime(last_datetime, '%d/%m/%y %H:%M:%S')

            # define the query parameters for the API request
            params = {
                'symbol': symbol,
                'interval': interval,
                'from': None,
                'to': None
            }

            httpreq = KlineRequest(symbol, interval, datetime_object, 
                                url, endpoint, method, params, api_key, api_secret, '\nKline-demo-test')

            df = httpreq.get_kline_bybit()
            print(f'\nDataframe shape: {df.shape}')
            print(f'\n{df.head()}\n')

        elif request_type == 'classic':
            limit = config['PARAM']['limit']

            # define the query parameters for the API request
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': int(limit)
            }

            Info = '\nGeneric-Request-Demo-Test'

            # class-object creation
            httpreq = HTTPRequest(url, endpoint, method, params, api_key, api_secret, Info)

            # method invocation - makes the request for data
            response = httpreq.HTTP_Request()
            # print(response.text)

            # manipulate data to be analized
            data = json.loads(response.content)['result']['list']
            df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'], dtype=float)

            # print some info
            print(f'\nDataframe shape: {df.shape}')
            print(f'\n{df.head()}\n')

            # # convert the pandas object to a tensor
            # data = tf.convert_to_tensor(df)
            # print(data)

    elif (args.train_model == True or 
          args.make_prediction == True):
        df = pd.read_csv('./data/market-data.csv', index_col='date', parse_dates=True)

        df.drop(['symbol', 'interval', 'open_time',
                'turnover'], inplace=True, axis=1)

        columns_titles = ['open', 'high', 'low', 'volume', 'close']
        df = df.reindex(columns=columns_titles)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        train_size = int((len(df) * args.split_perc))

        # data-preprocessing
        #########################################
        X_train = X.iloc[:train_size, :]

        y_train = y.iloc[:train_size, :]

        X_test = X.iloc[train_size:, :]
        y_test = y.iloc[train_size:, :]

        mm = MinMaxScaler()
        ss = StandardScaler()

        X_train_ss = ss.fit_transform(X_train)
        y_train_mm = mm.fit_transform(y_train)

        X_test_ss = ss.transform(X_test)
        y_test_mm = mm.transform(y_test)
        ##########################################

        # parameters
        ###########################
        input_size = X_train.shape[1]
        ###########################

        crypto_train_data = CryptoDataset(X_train_ss, y_train_mm, args.seq_len)
        train_dataloader = DataLoader(crypto_train_data, batch_size=args.bs_train, 
                                      num_workers=args.workers, shuffle=False)
        
        crypto_test_data = CryptoDataset(X_test_ss, y_test_mm, args.seq_len)
        test_dataloader = DataLoader(crypto_test_data, batch_size=args.bs_test, 
                                     num_workers=args.workers, shuffle=False)
        
        n_items_train = len(crypto_train_data)
        print(f'\nNumber of items in training-set: {n_items_train}')
                
        n_items_test = len(crypto_test_data)
        print(f'Number of items in test-set: {n_items_test}')

        # hard-coded
        #############################
        args.bs_train = n_items_train
        args.bs_test = n_items_test
        args.print_every = 1
        #############################

        """
        # Debugging
        #############################################
        # accediamo ad un singolo elemento del dataset
        x, y = crypto_train_data[0]
        print(x.shape)
        print(y.shape)

        # esempio di utilizzo del dataloader
        for batch_idx, (data, target) in enumerate(train_dataloader):
            print("Batch idx {}, input shape {}, target shape {}".format(batch_idx, data.shape, target.shape))

        x, y = crypto_test_data[0]
        print(x.shape)
        print(y.shape)
        
        for batch_idx, (data, target) in enumerate(test_dataloader):
            print("Batch idx {}, input shape {}, target shape {}".format(batch_idx, data.shape, target.shape))
        #############################################
        """

        """ Specify the device type responsible to load a tensor into memory. """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'\nDevice: {device}')

        # define solver class
        solver = Solver(args=args, 
                        device=device, 
                        input_size=input_size, 
                        train_dataloader=train_dataloader, 
                        test_dataloader=test_dataloader)
        
        if args.train_model == True:
            solver.train()
        elif args.make_prediction == True:
            args.resume_train = True
            solver.make_prediction()

    elif args.plot_data == True:
        print('\nPlot data...')
        
        df = pd.read_csv('./market-data.csv', index_col='date') #, parse_dates=True)
        vz = Visualizer()
        vz.plot_training_data(df)


if __name__ == '__main__':
    args = get_args()
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    print(f'\n{args}')
    
    main(args)

