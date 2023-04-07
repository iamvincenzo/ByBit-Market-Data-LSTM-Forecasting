import pandas as pd
from torch.utils.data import DataLoader

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

from solver import Solver
from model import LSTMModel
from model_cfg import config
from Normalizer import Normalizer
from CustomDataset import TimeSeriesDataset
from CustomDataset import TimeSeriesPreparation


""" Helper function used to import and plot data. """
def get_data(config):
    print('\nGetting the data...')

    df = pd.read_csv(config['data']['path'],
                     index_col='date', parse_dates=True)
    df.drop(['start'], inplace=True, axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    num_data_points = len(X)
    data_date = [d for d in X.index.strftime('%Y/%m/%d, %H:%M:%S')]
    # data_close_price = y.to_numpy()

    display_date_range = 'from ' + data_date[0] + ' to ' + data_date[num_data_points-1]
    print('\nNumber data points:', num_data_points, display_date_range)

    # if config['plots']['show_plots']:
    #     fig = figure(figsize=(25, 5), dpi=80)
    #     fig.patch.set_facecolor((1.0, 1.0, 1.0))
    #     plt.plot(data_date, data_close_price,
    #              color=config['plots']['color_actual'])
    #     xticks = [data_date[i] if ((i % config['plots']['xticks_interval'] == 0 and (num_data_points-i) > config['plots']
    #                                ['xticks_interval']) or i == num_data_points-1) else None for i in range(num_data_points)]  # make x ticks nice
    #     x = np.arange(0, len(xticks))
    #     plt.xticks(x, xticks, rotation='vertical')
    #     plt.title('Daily close price for ' +
    #               config['data']['symbol'] + ', ' + display_date_range)
    #     plt.grid(visible=None, which='major', axis='y', linestyle='--')
    #     plt.show()

    return X.to_numpy(), y.to_numpy()

""" Main function used to run the simulation. """
def main():

    # Getting parameters
    #################
    bs_size = config["training"]["batch_size"]
    in_size = config["model"]["input_size"]
    hid_layer_size = config["model"]["lstm_size"]
    n_layers = config["model"]["num_lstm_layers"]
    out_size = config["model"]["output_size"]
    dropout_rate = config["model"]["dropout"]
    device = config["training"]["device"]
    ################

    X, y = get_data(config)

    ts_prep = TimeSeriesPreparation(X, y)

    # X_final_pred: used for final prediction
    print('\nPreparing the data...')
    split_index, X_train, y_train, X_val, y_val, X_final_pred = ts_prep.prepare_data()

    # debugging
    ###############################################
    print(f'\nPrinting some info...')
    print(f'split-index: {split_index}')
    print(f'X_train-shape: {X_train.shape}')
    print(f'y_train-shape: {y_train.shape}')
    print(f'X_val-shape: {X_val.shape}')
    print(f'y_val-shape: {y_val.shape}')
    print(f'X_final_pred-shape: {X_final_pred.shape}')
    ###############################################
    
    # normalize data
    print('\nNormalizing the data...')
    scaler = Normalizer()
    X_train_norm = scaler.fit_transform(X_train)
    y_train_norm = scaler.fit_transform(y_train)
    X_val_norm = scaler.transform(X_val)
    y_val_norm = scaler.transform(y_val)

    # create Dataset 
    print('\nCreation of dataset...')
    dataset_train = TimeSeriesDataset(X_train_norm, y_train_norm)
    dataset_val = TimeSeriesDataset(X_val_norm, y_val_norm)

    # create DataLoader
    train_dataloader = DataLoader(dataset_train, batch_size=bs_size, shuffle=False) #shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=bs_size, shuffle=False) #shuffle=True)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)


    print('\nModel creation...')
    if True:
        model = LSTMModel(input_size=in_size, hidden_layer_size=hid_layer_size,
                          num_layers=n_layers, output_size=out_size, dropout=dropout_rate)
    else:
        from model import LSTMModel2
        model = LSTMModel2(device=device, input_size=in_size, 
                           hidden_size=hid_layer_size, num_layers=n_layers, output_size=out_size)    
    model = model.to(device)

    solver = Solver(model, 
                    train_dataloader, 
                    val_dataloader)
    
    print('\nStarting training...')
    solver.train_eval()

    ############################# RES EVAL #########################

    print('\nStarting test...')
    solver.make_pred()

if __name__ == '__main__':
    main()
