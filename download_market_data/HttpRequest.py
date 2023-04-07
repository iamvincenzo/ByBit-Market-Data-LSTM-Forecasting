import time
import hmac
import hashlib
import requests


class HTTPRequest(object):
    """ Initialize configurations. """
    def __init__(self, url, endpoint, method, params, api_key, api_secret, Info):
        super(HTTPRequest, self).__init__()
        self.endpoint = endpoint
        self.method = method
        self.params = params
        self.api_key = api_key
        self.api_secret = api_secret
        self.url = url

        self.Info = Info

        self.httpClient = requests.Session()
        self.recv_window = str(5000)

    """ Methdod used to get the hmac signature used for authentication. """
    def genSignature(self):
        hash = hmac.new(bytes(self.api_secret, "utf-8"),
                        self.param_str.encode("utf-8"), hashlib.sha256)
        signature = hash.hexdigest()

        return signature

    """ Method used to make a generic request. """
    def HTTP_Request(self):
        time_stamp = str(int(time.time() * 10 ** 3))           

        # convert dict into string
        self.param_str = '&'.join([f"{k}={v}" for k, v in self.params.items()])

        signature = self.genSignature()

        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': time_stamp,
            'X-BAPI-RECV-WINDOW': self.recv_window,
            'Content-Type': 'application/json'
        }
        
        if (self.method == "POST"):
            print(f'\nMaking request to: {self.url + self.endpoint}')
            response = self.httpClient.request(self.method, self.url + self.endpoint,
                                               headers=headers, data=self.param_str)
        else:
            print(f'\nMaking request to: {self.url + self.endpoint}')
            response = self.httpClient.request(self.method, self.url + self.endpoint + '?' + self.param_str,
                                               headers=headers)

        print(self.Info + " Response Time : " + str(response.elapsed))

        return response
    
