import pandas as pd
import seaborn as sns
import yfinance as yf
import numpy as np
import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm
from copy import deepcopy
from datetime import datetime
# from utils import init_logger

COLOR = "#50C878"


class PortfolioAnalysis:
    """
    Class for portfolio analysis
    """
    def __init__(self, tickers: any,
                 interval: str,
                 bankroll: float = 100000.00):
        """
        Parameters
        ______________
            tickers: str or list
        a list of tickers passed as a list or a single ticker passed as a string

            interval: str
        1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo are the selctions
        m is for minutes, h hour, d day, wk week, mo months

            bankroll: float
        The total amount of money on hand the investor has to invest in stocks. This is uninvested cash
        """
        self._tickers = tickers
        self.interval = interval
        self.historical_prices = self._get_prices(tickers=self._tickers,
                                                  interval=self.interval)

        self.bankroll = bankroll
        self.invested_capital = 0.00
        self.portfolio = pd.DataFrame()
        self.portfolio_metadata = {}

    def _get_prices(self, tickers: any,
                    interval: str) -> pd.DataFrame:
        """
        Wrapper function returns the pricing data for the tickers
        passed.
        Parameters
        _________________

         tickers: str or list
        a list of tickers passed as a list or a single ticker passed as a string

         interval: str
        1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo are the selctions
        m is for minutes, h hour, d day, wk week, mo months

        Returns
        ______________

        pd.Dataframe
        """
        if isinstance(tickers, list):
            # convert to upper case incase they are not
            tickers = [x.upper() for x in tickers]
            # grab the ticker data and merge as one dataframe
            dfs = []
            for tic in tickers:
                logging.info(f"grabbing data for {tic}: interval = {interval}")
                df_data = yf.download(tic, period="max", interval=interval)
                df_data['ticker'] = tic
                # round all the data points to 4 decimals
                df_data = df_data.round(4)
                dfs.append(df_data)
            # append the data
            df_data = pd.concat(dfs)
            logging.info("Data fully loaded!")
        else:
            logging.info(f"grabbing data for {tickers}: interval = {interval}")
            df_data = yf.download(tickers, period="max", interval=interval)
            df_data['ticker'] = tickers
            logging.info("Data fully loaded!")
        # convert column names to lower case
        df_data.columns = df_data.columns.str.lower()

        return df_data

    def add_bankroll(self, cash_amount: float) -> None:
        """
        add cash to the bankroll
        """
        logging.info(f"adding {cash_amount} to bankroll")
        self.bankroll += cash_amount

    def subtract_bankroll(self, cash_amount: float) -> None:
        """
        add cash to the bankroll
        """
        # check if the cash amount is more than the bankroll
        if cash_amount > self.bankroll:
            logging.error("Cash amount is larger than the bankroll")
            raise ValueError
        logging.info(f"subtracting {cash_amount} from bankroll")
        self.bankroll -= cash_amount

    def calc_returns(self) -> None:
        """
        Function calculates both the standard and logartim returns as well as the price difference
        on the closing price at each time interval

        Returns
        ________________
        None
        """
        # check if there are more than one ticker in the dataframe
        if self.historical_prices['ticker'].nunique() > 1:
            dfs = []
            for tic, group_df in self.historical_prices.groupby(by='ticker'):
                # sort the data by the time index
                logging.info(f"Calculating returns for {tic}")
                group_df['standard_returns'] = group_df['adj close'].pct_change(1)
                group_df['log_returns'] = np.log(group_df['adj close']) - np.log(group_df['adj close'].shift(1))
                group_df['price_difference'] = group_df['adj close'].diff(1)
                dfs.append(group_df)
            # concat the dataframes together
            df_data = pd.concat(dfs)
            self.historical_prices = df_data
        else:
            logging.info(f"Calculating returns for {self.historical_prices['ticker'].unique()[0]}")
            df_data = self.historical_prices.sort_index()
            df_data['standard_returns'] = df_data['adj close'].pct_change(1)
            df_data['log_returns'] = np.log(df_data['adj close']) - np.log(df_data['adj close'].shift(1))
            df_data['price_difference'] = df_data['adj close'].diff(1)
            self.historical_prices = df_data
        logging.info(f"returns have been calculated Successfully!\n{df_data[['ticker', 'adj close', 'standard_returns', 'log_returns', 'price_difference']].head()}")

    def construct_portfolio(self, tickers: list, incep_dates: list, num_shares) -> None:
        """
        Function transforms the histroical data into a data frame where the adj price for each ticker
        passed in the parameters to rows based on the inception date of the ticker and constructs the portfolio
        metadata for calculations of returns

        Parameters
        _____________
            tickers: list
        the list of tickers in the portfolio

            incep_dates: list
        the date the shares were brought for the same indexed ticker passed in the tickers parameter.
        format- 2022-01-31 00:00:00-00:00

            num_shares: list
        the number of shares for the same indexed ticker passed in the tickers parameter
        """
        # convert to upper case incase they are not
        tickers = [x.upper() for x in tickers]
        counter = 0
        df_final = pd.DataFrame()
        if not len(tickers) == len(incep_dates) == len(num_shares):
            logging.error("Lists are not the same length")
            raise ValueError
        for tic, incep_dt, shares in zip(tickers, incep_dates, num_shares):
            # check if the ticker is in historical prices
            df_filter = deepcopy(self.historical_prices[(self.historical_prices["ticker"] == tic) & (self.historical_prices.index >= incep_dt)]["adj close"]).to_frame()
            if df_filter.empty:
                logging.error(f"{tic} is not in historical prices or there is not enough historical data, check histroical prices")
                continue
            investment_value = shares * df_filter["adj close"][0]
            try:
                # subtract the investment from the bankroll
                self.subtract_bankroll(cash_amount=investment_value)
            except Exception as e:
                logging.error(f"Could not purchase {tic} at the current share price\n{e}")
                continue
            # construct the metadata for the portfolio
            self.portfolio_metadata[tic] = {}
            self.portfolio_metadata[tic]["shares_total"] = shares
            self.portfolio_metadata[tic]["investment"] = shares * df_filter["adj close"][0]
            self.portfolio_metadata[tic]["inception"] = incep_dt
            # change the column name to indicate the ticker
            df_filter.rename(columns={"adj close": f"{tic}_adj_close"}, inplace=True)
            # merge the data
            if counter == 0:
                df_final = df_filter
            else:
                df_final = pd.merge(df_final, df_filter, right_index=True, left_index=True, how='outer')
            counter += 1
        # set the portfolio
        self.portfolio = df_final

    def calc_portfolio_returns(self) -> None:
        """
        Function constructs the portfolio holding period returns, capital gains of each asset, and the total portfolio
        capital gains based on what is constructed in the function construct portfolio

        HPR - Holding period return: https://www.investopedia.com/terms/h/holdingperiodreturn-yield.asp
        """
        # calculate the capital gains for each stock
        for tic in self.portfolio_metadata.keys():
            # create a current value column
            self.portfolio[f"{tic}_current_val"] = self.portfolio[f"{tic}_adj_close"] * self.portfolio_metadata[tic]["shares_total"]
            # subtract the investment in the asset from the current value of the asset
            self.portfolio[f"{tic}_cap_gains"] = self.portfolio[f"{tic}_current_val"] - self.portfolio_metadata[tic]["investment"]
            # set temp investment columns for each ticker based on their inception date
            indexes = self.portfolio[self.portfolio.index >= self.portfolio_metadata[tic]['inception']].index
            self.portfolio.loc[indexes, f"{tic}_invest_temp"] = self.portfolio_metadata[tic]['investment']

        # sum up the total investments at each time step
        invest_cols = [x for x in self.portfolio.columns if "invest_temp" in x]
        self.portfolio["total_investment"] = self.portfolio[invest_cols].sum(axis=1)
        # sum up the current value of the portfolio
        current_val_cols = [x for x in self.portfolio.columns if "current_val" in x]
        self.portfolio["portfolio_value"] = self.portfolio[current_val_cols].sum(axis=1)
        # Compute the log returns of portfolio
        self.portfolio['portfolio_log_returns'] = np.log(self.portfolio['portfolio_value']) - np.log(self.portfolio['portfolio_value'].shift(1))
        # calculate the portfolio cap gains at each time step
        stock_gains_columns = [x for x in self.portfolio.columns if "_cap_gains" in x]
        self.portfolio["porftfolio_cap_gains"] = self.portfolio[stock_gains_columns].sum(axis=1)
        # calculate the portfolio returns at each time step
        self.portfolio["porftfolio_hpr"] = (self.portfolio["portfolio_value"] - self.portfolio["total_investment"]) / self.portfolio["total_investment"]
        # drop the temporary investment columns for each ticker to reduce the width of the dataframe
        self.portfolio.drop(columns=invest_cols, axis=1, inplace=True)
        # drop the temporary current asset value columns for each ticker to reduce the width of the dataframe
        self.portfolio.drop(columns=current_val_cols, axis=1, inplace=True)

    def hist_portfolio_returns(self) -> None:
        """
        Function plots the distribution of the portfolio capital gains pct for the portfolio
        based on the histiorical time cadence. The qq plot is also added to gain a sense of how
        normalized the returns are.
        """
        # format the ticker list
        if len(self._tickers) == 1:
            ticker_title = self._tickers[0]
        else:
            ticker_title = self._tickers.join(", ")
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))
        sns.histplot(data=self.portfolio, x="portfolio_log_returns", color=COLOR, ax=axes[0]).set_title(f"Portfolio {self.interval} Returns - {ticker_title}")
        sm.qqplot(self.portfolio["portfolio_log_returns"], line='s', ax=axes[1])
        sns.lineplot(data=self.portfolio, x=self.portfolio.index, y="porftfolio_hpr", color=COLOR, ax=axes[2]).set_title(f"Portfolio HPR {self.interval} Returns - {ticker_title}")
        axes[1].set_title(f"{ticker_title} QQ-Plot")
        plt.show()

    def plot_historical_returns(self, height: int = 12,
                                width: int = 6) -> None:
        """
        Function plots a histogram and a qqplot for each asset in the historical
        prices and returns data.
        """
        # get the numer of assets in the historical dataset
        num_assests = self.historical_prices['ticker'].nunique()
        # create the figure for the plots
        fig, axes = plt.subplots(nrows=num_assests, ncols=2, figsize=(height, width))
        # create the plots for each asset
        counter = 0
        axes = axes.flat
        for group, group_df in self.historical_prices.groupby(by='ticker'):
            sns.histplot(data=group_df, x="log_returns", color=COLOR, ax=axes[counter]).set_title(
                f"Portfolio {self.interval} Returns - {group}")
            counter += 1
            sm.qqplot(group_df["log_returns"], line='s', ax=axes[counter])
            counter += 1

    def plot_historical_series(self, column: str='log_returns',
                               height: int = 9,
                               width: int = 1
                               ):
        """
        Function plots a time series line plot based on the pased column
        value.

        Parameters
        _____________
            column: str
        The column in the historical prices dataset

            height: int
        the height of the plotted figure size

            width: int
        The width of the plotted figure size
        """
        # get the numer of assets in the historical dataset
        num_assests = self.historical_prices['ticker'].nunique()
        counter = 0
        fig, axes = plt.subplots(nrows=num_assests, ncols=1, figsize=(width, height))
        for group, group_df in self.historical_prices.groupby(by='ticker'):
            group_df.sort_index(ascending=True, inplace=True)
            sns.lineplot(data=group_df,
                         x=group_df.index,
                         y=column,
                         ax=axes[counter]).set_title(f"{group} {self.interval} {column}")
            plt.xticks(rotation=45)
            counter += 1


if __name__ == "__main__":
    # init_logger("calc_returns")
    # load tickers
    tickers = ['msft', 'aapl', 'goog']
    inceptions = [datetime(2023, 7, 31, 0, 0, 0, 0), datetime(2023, 8, 4, 0, 0, 0, 0), datetime(2023, 8, 4, 0, 0, 0, 0)]
    shares = [10, 20, 15]
    portfolio_data = PortfolioAnalysis(tickers=tickers, interval='1d')

    print(portfolio_data.historical_prices.head())
    # calculate the returns
    portfolio_data.calc_returns()
    print(portfolio_data.historical_prices.head())
    portfolio_data.construct_portfolio(tickers=tickers,
                                       incep_dates=inceptions,
                                       num_shares=shares)
    print(portfolio_data.portfolio)
    print(portfolio_data.portfolio_metadata)


