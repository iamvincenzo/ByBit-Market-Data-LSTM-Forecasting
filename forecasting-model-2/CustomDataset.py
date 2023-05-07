import numpy as np
from model_cfg import config
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    """ Initialize configurations. """
    def __init__(self, x, y):
        # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        # x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class TimeSeriesPreparation():
    """ Initialize configurations. """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w_sz = config['data']['window_size']
        self.train_split_size = config['data']['train_split_size']

    """ Function used to get the next-day-closing-price 
        used to compose the target label. """
    def prepare_y(self):
        # use the next day as label
        output = self.y[self.w_sz:]

        return output

    """ Function used to perform widnowing on input data. """
    def prepare_X(self):
        """ Calcolo del numero effettivo di righe che si ottengono facendo
            scorrere la finestra lungo il dataset. """
        n_row = self.X.shape[0] - self.w_sz + 1

        """ Spiegazione: stride_tricks
        Questa funzione crea una vista sull'array: significa 
        che genera un nuovo array con i dati di input. Ad esempio:
        array([[0,  1,  2],
            [3,  4,  5],
            [6,  7,  8],
            [9, 10, 11],
            [12, 13, 14]])
        # >>>>>>>>>>> con window_size = 2
        array([[[0,  1,  2],
                [3,  4,  5]],

            [[3,  4,  5],
                [6,  7,  8]],

            [[6,  7,  8],
                [9, 10, 11]],

            [[9, 10, 11],
                [12, 13, 14]]])
        """
        X_windowed = np.lib.stride_tricks.as_strided(self.X, shape=(n_row, self.w_sz, self.X.shape[1]),
                                                     strides=(self.X.strides[0], self.X.strides[0], self.X.strides[1]))

        return X_windowed[:-1], X_windowed[-1]

    """ Helper function used to prepare data. """
    def prepare_data(self):

        X_windowed, X_final_pred = self.prepare_X()
        y_windowed = self.prepare_y()

        # dataset-split
        split_index = int(y_windowed.shape[0] * self.train_split_size)
        X_train = X_windowed[:split_index]
        X_val = X_windowed[split_index:]
        y_train = y_windowed[:split_index]
        y_val = y_windowed[split_index:]

        num_data_points = len(X_train) + len(X_val) + len (X_final_pred)

        # if config['plots']['show_plots']:
        #     to_plot_data_y_train = np.zeros((num_data_points, 1))
        #     to_plot_data_y_val = np.zeros((num_data_points, 1))

        #     to_plot_data_y_train[config["data"]["window_size"]:split_index + config["data"]["window_size"]] = y_train
        #     to_plot_data_y_val[split_index + config["data"]["window_size"]:] = y_val

        #     to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
        #     to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)


        #     fig = plt.figure(figsize=(25, 5), dpi=80)
        #     fig.patch.set_facecolor((1.0, 1.0, 1.0))
        #     plt.plot(to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
        #     plt.plot(to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
        #     # xticks = [data_date[i] if ((i % config["plots"]["xticks_interval"] == 0 and (num_data_points-i) > config["plots"]
        #     #                         ["xticks_interval"]) or i == num_data_points-1) else None for i in range(num_data_points)]  # make x ticks nice
        #     # x = np.arange(0, len(xticks))
        #     # plt.xticks(x, xticks, rotation='vertical')
        #     plt.title("Daily close prices for " +
        #             config["data"]["symbol"] + " - showing training and validation data")
        #     plt.grid(visible=None, which='major', axis='y', linestyle='--')
        #     plt.legend()
        #     plt.show()

        return split_index, X_train, y_train, X_val, y_val, X_final_pred
