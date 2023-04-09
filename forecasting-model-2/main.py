import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

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

    """ if config['plots']['show_plots']:
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price,
                 color=config['plots']['color_actual'])
        xticks = [data_date[i] if ((i % config['plots']['xticks_interval'] == 0 and (num_data_points-i) > config['plots']
                                   ['xticks_interval']) or i == num_data_points-1) else None for i in range(num_data_points)]  # make x ticks nice
        x = np.arange(0, len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title('Daily close price for ' +
                  config['data']['symbol'] + ', ' + display_date_range)
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.show()
    """

    return X.to_numpy(), y.to_numpy(), data_date

""" Main function used to run the simulation. """
def main():

    # Getting parameters
    #################
    device = config["training"]["device"]
    in_size = config["model"]["input_size"]
    # win_size = config["data"]["window_size"]
    out_size = config["model"]["output_size"]
    dropout_rate = config["model"]["dropout"]
    bs_size = config["training"]["batch_size"]
    # bs_train = config["training"]["bs_train"]
    # bs_test = config["training"]["bs_test"]
    n_layers = config["model"]["num_lstm_layers"]
    hid_layer_size = config["model"]["lstm_size"]
    ################

    X, y, data_date = get_data(config)

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
    train_dataloader = DataLoader(dataset_train, batch_size=bs_size, shuffle=False) #shuffle=True
    val_dataloader = DataLoader(dataset_val, batch_size=bs_size, shuffle=False) #shuffle=True)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    print('\nModel creation...')
    if True:
        model = LSTMModel(input_size=in_size, hidden_layer_size=hid_layer_size,
                          num_layers=n_layers, output_size=out_size, dropout=dropout_rate)
    # else:
    #     from model import LSTMModel2
    #     model = LSTMModel2(device=device, input_size=in_size, 
    #                        hidden_size=hid_layer_size, num_layers=n_layers, output_size=out_size)   
 
    model = model.to(device)

    solver = Solver(model, 
                    train_dataloader, 
                    val_dataloader)
    
    print('\nStarting training...')
    solver.train_eval()

    ############################# MODEL EVALUATION #########################
    """ print('\nStarting test...')
    # solver.make_pred()

    train_dataloader = DataLoader(dataset_train, batch_size=bs_size, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=bs_size, shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out[:, -1]))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out[:, -1]))

    if config["plots"]["show_plots"]:
        # prepare data for plotting, show predicted prices
        num_data_points = len(X_train) + len(X_val) + len(X_final_pred)
        to_plot_data_y_train_pred = np.zeros(num_data_points)
        to_plot_data_y_val_pred = np.zeros(num_data_points)

        to_plot_data_y_train_pred[win_size:split_index + win_size] = scaler.inverse_transform(predicted_train)
        to_plot_data_y_val_pred[split_index + win_size:] = scaler.inverse_transform(predicted_val)

        to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

        # plots
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, y, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
        plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Compare predicted prices to actual prices")
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

        # prepare data for plotting, zoom in validation
        to_plot_data_y_val_subset = scaler.inverse_transform(y_val)
        to_plot_predicted_val = scaler.inverse_transform(predicted_val)
        to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

        # plots
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Zoom in to examine predicted price on validation data portion")
        xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
        xs = np.arange(0,len(xticks))
        plt.xticks(xs, xticks, rotation='vertical')
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()
    ########################################################################
    """
    ######################## Predicting future stock prices #############################
    model.eval() # predict on the unseen data, tomorrow's price 

    x = torch.tensor(X_final_pred).float().to(device).unsqueeze(0) # this is the data type and shape required, [batch, sequence, feature]
       
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    prediction = scaler.inverse_transform(prediction[:, -1])[0]

    """ if config["plots"]["show_plots"]:            
        # prepare plots
        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(y_val)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

        to_plot_data_y_test_pred[plot_range-1] = prediction

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        plot_date_test = data_date[-plot_range+1:]
        plot_date_test.append("next trading day")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
        plt.title("Predicted close price of the next trading day")
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()
    """

    print("\nPredicted close price of the next 30 minutes:", round(prediction[0], 2))
    #####################################################################################


if __name__ == '__main__':
    main()
