# python imports
from datetime import datetime
# third party imports
import yfinance as yf


class Stock():
    """
    Stocks Class
    ------------

    Parameters
    ----------

    stock_name: str,
        the name of the stock that you are going to retrieve information
    start_date: datetime.date,
            initial date that you want the histroic of the stocks to start
        end_date: datetime.date,
            initial date that you want the histroic of the stocks to start

    """

    def __init__(
        self,
        stock_name: str,
        start_date: datetime,
        end_date: datetime,
    ):
        if start_date == None or end_date == None:
            raise('Date values must selected')

        self.name = stock_name
        self.ticker = yf.Ticker(stock_name)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')

    def get_historic(self):
        """
        Get Historic
        ------------

        Returns
        -------
        df: pd.Daframe,
            historic dataframe

        """
        print("Getting: ", self.name, " from ",
              self.start_date, " to ", self.end_date)
        df = self.ticker.history(start=self.start_date, end=self.end_date)
        df = df[['High', 'Low', 'Close', 'Open']]

        return df
