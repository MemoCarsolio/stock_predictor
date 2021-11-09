import json
import requests
import yfinance as yf
from datetime import datetime


class Stock():
    """
    Stocks Class
    ------------

    Parameters
    ----------

    stock_name: str,
        the name of the stock that you are going to retrieve information

    """

    def __init__(
        self,
        stock_name: str,
        start_date: datetime,
        end_date: datetime,
        is_target: bool = False
    ):
        if start_date == None or end_date == None:
            raise('Date values must selected')

        self.name = stock_name
        self.ticker = yf.Ticker(stock_name)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')
        self.is_target = is_target

    def get_historic(self):
        """
        Get Historic

        """
        print("Getting: ", self.name, " from ",
              self.start_date, " to ", self.end_date)
        df = self.ticker.history(start=self.start_date, end=self.end_date)
        df = df[['High', 'Low', 'Close', 'Open']]

        return df
