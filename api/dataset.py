from typing import ClassVar
import pandas as pd
from torch._C import get_default_dtype
from torch.functional import split
from .stocks import Stock
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DatasetConstructur():

    def __init__(self, stock_vector, start_date, end_date, sequence_length) -> None:
        """
        Main
        """
        self.stock_vector = stock_vector
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length

        self.dataframes = self.get_dataframes()
        self.datasets = self.get_dataset_list()
        # self.tensor = self.dataset_to_tensor()
        self.input_size = len(self.datasets[0]['dataset'].columns)

    def get_stock_index(self, stock):
        stocks = {
            'AAPL': 0,
            'AXP': 1,
            'KO': 2,
            'WMT': 3,
            'DIS': 4,
            'IBM': 5,
            'NKE': 6,
            'V': 7
        }
        return stocks[stock]

    def get_dataframes(self):

        dfs = {}
        # stock_vector = self.stock_vector[:]

        # stock_vector.remove(main_stock)

        # dfs.append(Stock(main_stock, self.start_date,
        #                  self.end_date, True).get_historic())

        for stock in self.stock_vector:
            dfs[stock] = Stock(stock, self.start_date,
                               self.end_date).get_historic()

        return dfs

    def order_dataframes(self, main_stock, dataframes):

        stocks = []

        stocks.append(dataframes[main_stock])

        for stock in dataframes:
            if stock != main_stock:

                columns = dataframes[stock].columns
                for col in columns:
                    dataframes[stock] = dataframes[stock].rename(columns={
                        col: stock + '_' + col
                    })

                stocks.append(dataframes[stock])
        return stocks

    def construct_dataset(self, main_stock):

        first = True

        base_dataframes = self.dataframes.copy()
        dataframes = self.order_dataframes(main_stock, base_dataframes)

        for df in dataframes:
            if first:
                ds = df
                first = False
            else:
                ds = pd.merge(ds, df, left_index=True, right_index=True)

        cols = ds.columns

        scaler = MinMaxScaler()

        ds[cols] = scaler.fit_transform(ds[cols])

        ds['Date'] = ds.index
        ds['Month'] = pd.DatetimeIndex(ds['Date']).month
        ds['Weekday'] = pd.DatetimeIndex(ds['Date']).weekday
        ds['Stock'] = self.get_stock_index(main_stock)
        ds = ds.sort_index(axis=0)

        ds = ds.drop(columns=['Date'])

        return ds

    def get_stock_label(self, last_day, future_day):

        if last_day < future_day:
            return 1
        else:
            return 0

    def split_datasets(self, df):

        split_list = []
        size = len(df)
        current_size = 0

        while current_size + self.sequence_length+1 <= size:

            if current_size == 0:
                aux_df = df[:self.sequence_length+1]
                split_df = aux_df[:-1]
                future_value = aux_df[-1:]['Close'].tolist()[0]

                label = self.get_stock_label(
                    split_df[-1:]['Close'].tolist()[0], future_value)

                split_list.append({
                    'dataset': split_df,
                    'label': label
                })
                current_size = self.sequence_length + 1
            else:
                aux_df = df[current_size:current_size+self.sequence_length+1]
                split_df = aux_df[:-1]
                future_value = aux_df[-1:]['Close'].tolist()[0]

                label = self.get_stock_label(
                    split_df[-1:]['Close'].tolist()[0], future_value)

                split_list.append({
                    'dataset': split_df,
                    'label': label
                })
                current_size = current_size + self.sequence_length + 1
        return split_list

    def get_dataset_list(self):

        datasets = []
        for stock in self.stock_vector:
            datasets.extend(self.split_datasets(self.construct_dataset(stock)))

        return datasets

    def dataset_to_tensor(self):
        """
        Dataset -> Tensor

        In this function we turn the dataset into a 3 dimenstional tensor

        """

        t_size = len(self.dataset)
        s_lenght = self.sequence_length
        features = len(self.dataset.columns)

        if t_size % s_lenght + 1 == 0:
            n = t_size / s_lenght + 1
        else:
            n = (t_size - t_size % s_lenght) / s_lenght + 1

        tensor = torch.empty(n, s_lenght, features)

        x = 0
        z = 0
        labels = []
        for _, row in self.dataset.iterrows():
            y = 0
            for col in row:
                if z == n:
                    z = 0
                    pass
                else:
                    pass
                tensor[0][x][y] = col
                y = y + 1
            x = x + 1

        return tensor
