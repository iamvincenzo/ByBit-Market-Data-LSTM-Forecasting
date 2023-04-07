import os
import configparser
from datetime import timezone
from datetime import datetime

from KlineRequest import KlineRequest

conf_pth = './market_config_files/'

""" Main method used to run the simulation. """
def main():
    print('\nDownloading data...')

    ###################### CONFIG-INFO ##########################
    config = configparser.ConfigParser()
    config.read(conf_pth + 'api_config.cfg')

    api_key = config['BYBIT']['api_key']
    api_secret = config['BYBIT']['api_secret']

    config = configparser.ConfigParser()
    config.read(conf_pth + 'request_config.cfg')

    url = config['URL']['url']
    endpoint = config['ENDPOINT']['endpoint']

    request_type = config['REQUEST']['request_type']
    symbol = config['PARAM']['symbol']
    interval = config['PARAM']['interval']

    method = config['METHOD']['method']
    #############################################################

    ################### DATA SAVE PATH ##########################
    data_path = config['DATAPATH']['data_path']
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    #############################################################

    if request_type == 'kline':
        print(f'\nKline request...')

        last_datetime = config['LASTDATETIME']['last_datetime']
        datetime_object = datetime.strptime(last_datetime, '%Y/%m/%d %H:%M:%S')
        datetime_object = datetime_object.replace(tzinfo=timezone.utc)

        # define the query parameters for the API request
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': None,
            'end': None
        }

        httpreq = KlineRequest(symbol, interval, datetime_object, url, endpoint,
                                method, params, api_key, api_secret, '\nKline-demo-test')

        df = httpreq.get_kline_bybit()
        
        print(f'\nDataFrame shape: {df.shape}')
        print(f'\nDataFrame head: \n{df.head(10)}')


if __name__ == '__main__':
    main()
