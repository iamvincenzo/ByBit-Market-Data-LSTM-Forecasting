# ByBit-Market-Data-Download-and-Forecasting

[![Windows](https://img.shields.io/badge/Windows-11-blue?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows/) [![VS Code](https://img.shields.io/badge/VS%20Code-v1.61.0-007ACC?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/) [![Python](https://img.shields.io/badge/Python-03.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-v1.10.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.4.3-FF5733?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/) [![NumPy](https://img.shields.io/badge/NumPy-v1.21.0-4C65AF?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-v4.6.0-brightgreen?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

This project aims to use LSTM networks to predict the next closing price of orders placed on the online cryptocurrency trading platform ByBit. Additionally, I have developed a client application that allows downloading the data that needs to be analyzed by the network.

Note that the model is built for exercise purposes only, as predicting cryptocurrency prices is a much more complex task. To achieve more accurate predictions, additional models like Natural Language Processing (NLP) techniques such as sentiment analysis, text classification, and topic modeling are required to analyze news and market information.

### Forecasting-model-1 results:

![Loss functions](./imgs/f1_loss.png)

![Predictions on validation-set](./imgs/f1_pred.png)

### Forecasting-model-2 results:

![Loss functions](./imgs/f2_loss.png)

![Predictions on validation-set](./imgs/f2_pred.png)

## Acknowledgments

Part of this project (forecasting-model-1) uses code with an MIT license. The full text of the license is available in licences/LICENSE file. The MIT-licensed code is derived from Bjarte Mehus Sunde https://github.com/Bjarten/early-stopping-pytorch, and is used in accordance with the terms of the MIT license.

Part of this project (forecasting-model-2) uses code with an Apache License 2.0. The full text of the license is available in licences/LICENSE file. The Apache License 2.0 licensed code is derived from Hong Jing (Jingles) https://github.com/jinglescode/time-series-forecasting-pytorch, and is used in accordance with the terms of the Apache License 2.0.

The HttpClient that allows you to download market data was created based on the examples provided by the API of the platform at https://github.com/bybit-exchange/api-usage-examples.
