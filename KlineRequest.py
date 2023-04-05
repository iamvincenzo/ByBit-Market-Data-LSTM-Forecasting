import json
import pandas as pd
from datetime import datetime, timedelta

from HttpRequest import HTTPRequest

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
        startTime = int(datetime.timestamp(startTime) * 1000)
        endTime = int(datetime.timestamp(endTime) * 1000)

        self.params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'start': startTime,
            'end': endTime
        }

        response = self.HTTP_Request()
        # print(response.text)
        json_response = json.loads(response.text)['result']

        if 'list' in json_response:
            # API-doc: (An string array of individual candle) (Sort in reverse by start)
            json_response['list'].reverse()
            df = pd.DataFrame(json_response['list'], 
                              columns=['start', 'open', 'high', 'low', 'close'])

            # if (len(df.index) == 0):
            #     print('\nNone information...\n')
            #     return None
            
            df.index = [datetime.fromtimestamp(int(x) / 1000) for x in df.start]

            return df
        
        else:
            print('list not in json_response...\n')
            return None

    """ Helper function used to obtain a dataframe
        containing cndles market data. """
    def get_kline_bybit(self):
        df_list = []
        last_datetime = self.startTime

        while True:
            print(f'last-datetime: {last_datetime}')
            new_df = self.get_bybit_bars(last_datetime, datetime.now())
            if new_df is None:
                break
            df_list.append(new_df)
            last_datetime = max(new_df.index) + timedelta(0, 1)
            print(f'max(new_df.index): {max(new_df.index)}')
            print(f'updated-last-datetime: {last_datetime}')

        df = pd.concat(df_list)
        
        self.data = df
        self.save_df_to_csv('./data/market-data.csv')

        return df