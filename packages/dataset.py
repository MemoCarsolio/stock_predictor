
# PyTorch imports
import torch
# third party imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# python imports
import random
# local imports
from .stocks import Stock


class DatasetConstructor():

    def __init__(
        self,
        main_stocks,
        start_date,
        end_date,
        sequence_length,
        batch_size
    ) -> None:
        """
        Dataset Constructor
        -------------------

        Parameters
        ----------
        main_stocks: list[str],
            List of all the stock NASDAQ tickers that you want the dataset to
            be built with
        start_date: datetime.date,
            initial date that you want the histroic of the stocks to start
        end_date: datetime.date,
            initial date that you want the histroic of the stocks to start
        sequence_length: int,
            size of sequence
        batch_size: int,
            size of batch

        Constant Variables:
        ---------
        stock_vector: list[str],
            the feature stocks for the rnn model

        """
        self.stock_vector = ['AAPL', 'AXP', 'KO',
                             'WMT', 'DIS', 'IBM',
                             'NKE', 'V']
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.main_stocks = main_stocks
        self.batch_size = batch_size

        self.dataframes = self.get_dataframes(self.stock_vector)
        self.main_dataframes = self.get_dataframes(self.main_stocks)
        self.datasets = self.get_dataset_list()
        self.input_size = len(self.datasets[0]['dataset'].columns)

    def get_dataframes(self, stocks):
        """
        Get Dataframes
        --------------
        Retrieves Historic dataframes from Stock Class

        Parameters
        ----------
        stocks: list[str],
            list of stocks


        Returns:
        --------
        dfs: dict{str: pd.Dataframe},
            Dictionary mapping the stock NASDAQ ticker and its historic pandas
            dataframe
        """

        dfs = {}
        for stock in stocks:
            dfs[stock] = Stock(stock, self.start_date,
                               self.end_date).get_historic()

        return dfs

    def get_stock_label(self, last_day, future_day):
        """
        Get Stock Labels
        ----------------
        Helper function to map a binary value if the stock went up or down

        Paramters
        ---------
        last_day: float,
            closing value of last day
        future_day: float,
            closing value of future day

        Returns:
        bool value,
            1 if it went up 0 if it went down

        """
        if last_day < future_day:
            return 1
        else:
            return 0

    def order_dataframes(self, main_df, dataframes):
        """
        Order Datframes
        ---------------
        Order Dataframe features in order to have the main stock 4 features
        in the front of the tensor

        Parameters
        ----------
        main_df: pd.Dataframe,
            dataframe of the main stock
        dataframes: list[pd.Dataframe],
            list of the resting feature stocks dataframes

        Returns:
        --------
        stocks: list[pd.Dataframe],
            list of ordered stocks dataframes
        """
        stocks = []

        stocks.append(main_df)

        for stock in dataframes:

            columns = dataframes[stock].columns
            for col in columns:
                dataframes[stock] = dataframes[stock].rename(columns={
                    col: stock + '_' + col
                })

            stocks.append(dataframes[stock])
        return stocks

    def construct_dataset(self, main_stock):
        """
        Construct Dataset
        -----------------
        Main function that merges all dataframes into one, scales the values
        and adds a few more columns (month, weekday[0-6])

        Parameters
        ----------
        main_stock: str,
            String of the target stock

        Returns
        -------
        ds: pd.Dataframe,
            Dataframe of all previous dataframes merged and transformed

        """

        first = True

        base_dataframes = self.dataframes.copy()

        main_df = self.main_dataframes[main_stock].copy()

        if main_df.empty:
            return pd.DataFrame()
        dataframes = self.order_dataframes(main_df, base_dataframes)

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
        ds = ds.sort_index(axis=0)

        ds = ds.drop(columns=['Date'])

        return ds

    def split_datasets(self, df):
        """
        Split Datasets
        -----------------
        Splits the dataset dataframe into sequences of the sequence_length 
        storing them in a dictionary with the label of the following day
        after the sequence

        Parameters
        ----------
        df: pd.Dataframe,
            dataset main dataframe

        Returns
        -------
        datasets: list[dict{
            'dataset': pd.Dataframe,
            'label': bit
        }]
        """
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
        """
        Get Dataset List
        ----------------
        Extends all the split datasets into one big list including all values
        of all target stocks


        Returns:
        --------
        datasets: list[dict{
            'dataset': pd.Dataframe,
            'label': bit
        }]

        """
        datasets = []
        for stock in self.main_stocks:
            df = self.construct_dataset(stock)
            if not df.empty:
                datasets.extend(self.split_datasets(df))

        return datasets

    def dataset_to_training_tensor(self):
        """
        Dataset to Tensor
        -----------------
        Transform list of dataframes into a tensor



        Returns:
        ---------
        training_batches: list[dict{
            'tensor': torch.tensor,
            'labels': torch.tensor
        }]
        """
        dataset = random.sample(self.datasets, len(self.datasets))

        size = len(dataset)

        rem = size % self.batch_size

        if rem != 0:
            size = size - rem

        batches = size / self.batch_size

        tensor = torch.empty(
            self.batch_size, self.sequence_length, self.input_size)
        training_labels = torch.empty(self.batch_size)

        training_batches = []
        total_size = 0
        for x in range(int(batches)):
            for n in range(self.batch_size):

                datagroup = dataset[total_size]
                training_labels[n] = int(datagroup['label'])

                df = datagroup['dataset']

                x = 0
                for _, row in df.iterrows():
                    y = 0
                    for col in row:

                        tensor[n][x][y] = col
                        y = y + 1
                    x = x + 1

                total_size = total_size + 1
            training_batches.append({
                'tensor': tensor.detach().clone(),
                'labels': training_labels.detach().clone()
            })
        return training_batches
