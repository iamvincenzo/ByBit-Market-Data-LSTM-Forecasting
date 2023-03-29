import json
import configparser
import pandas as pd
from HttpRequest import HTTPRequest
from datetime import datetime, timedelta


class KlineRequest(HTTPRequest):
    """ Initialize configurations. """
    def __init__(self, symbol, interval, startTime, url, endpoint, method, params, api_key, api_secret, Info):
        super().__init__(url, endpoint, method, params, api_key, api_secret, Info)
        self.symbol = symbol
        self.interval = interval
        self.startTime = startTime

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

        return df


""" Main method used to run the simulation. """
def main():
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


if __name__ == '__main__':
    main()
