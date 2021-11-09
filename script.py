# pytorch
import datetime
from pandas.core.tools.datetimes import to_datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
# third - party
import pandas as pd
from api.dataset import DatasetConstructur
# python imports
import requests
import json
from datetime import date, timedelta
from api.rnn import RNN


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


STOCK_LIST = ['AAPL', 'AXP', 'KO', 'WMT', 'DIS', 'IBM', 'NKE', 'V']


data = DatasetConstructur(
    STOCK_LIST, 'AAPL', date(2021, 1, 1), date(2021, 1, 15))


input_size = len(data.dataset.columns)
sequence_length = len(data.dataset)

num_layers = 3
hidden_size = 2
epochs = 2
data = []
model = RNN(data.input_size, 104, 2, data.input_size, data.sequence_length)


optimizer = optim.Adam(model.parametersm, lr='0.001')
