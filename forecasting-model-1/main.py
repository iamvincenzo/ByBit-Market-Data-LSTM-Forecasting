import os
import torch
import configparser
import pandas as pd

from solver import Solver
from CustomDataset import GetDataloader
from CustomDataset import TimeSeriesSplitDataloader

conf_path = './model_config_files/'


""" Main method used to run the simulation. """
def main():    
    print('\nTraining/Prediction...')

    ################### READING PARAMETERS ####################
    config = configparser.ConfigParser()
    config.read(conf_path + 'model_config.cfg')

    checkpoint_path = config['MODELINFO']['checkpoint_path']

    dataset_path = config['DATAPARAMS']['dataset_path']
    split_perc = float(config['DATAPARAMS']['split_perc'])
    seq_len = int(config['DATAPARAMS']['seq_len'])

    bs_train = int(config['TRAINPARAMS']['bs_train'])
    bs_test = int(config['TRAINPARAMS']['bs_test'])
    workers = int(config['TRAINPARAMS']['workers'])

    make_prediction = int(config['TASK']['make_prediction'])
    ##########################################################

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    df = pd.read_csv(dataset_path + 'market-data.csv', index_col='date', parse_dates=True)
    df.drop(['start'], inplace=True, axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    # parameters
    ###########################
    input_size = X.shape[1]
    ###########################

    """ # data-visualization
    ######################################################
    vz = Visualizer()
    train_size = int((len(df) * args.split_perc))
    vz.plot_data(df.iloc[:train_size, :], 'training-data')
    vz.plot_data(df.iloc[train_size:, :], 'test-data')
    vz.plot_pca(df.iloc[:train_size, :], 'training-data')
    vz.plot_pca(df.iloc[train_size:, :], 'test-data')
    ######################################################
    """

    data_timeseries = False

    if data_timeseries == False:
        getData = GetDataloader(X=X, y=y, bs_train=bs_train, bs_test=bs_test, 
                                max_batch_sz=False, workers=workers, 
                                seq_len=seq_len, split_perc=split_perc)
        # args.print_every = 1 # if max_batch_sz=True
        train_dataloader, val_dataloader, mm = getData.get_dataloaders()
        test_dataloader = None
    else:
        tspdl = TimeSeriesSplitDataloader(X=X, y=y, seq_len=seq_len, max_batch_sz=False, 
                                            batch_size=bs_train, val_test_split=0.5)
        # args.print_every = 1 # if max_batch_sz=True
        train_dataloader, val_dataloader, test_dataloader, mm = tspdl.get_dataloaders()

    """ # dataloader-debugging
    for _, (X, y) in enumerate(train_dataloader):
        print(X)
        print(y)
        a = input('...') 
    """

    """ Specify the device type responsible to load a tensor into memory. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    # define solver class
    solver = Solver(device=device, 
                    input_size=input_size, 
                    minmaxscaler=mm,
                    train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader)
    
    if make_prediction == 1:
        solver.make_prediction(X)
    else:
        solver.train()

if __name__ == '__main__':   
    main()
