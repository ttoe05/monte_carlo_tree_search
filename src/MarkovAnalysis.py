"""
This file is for runing markovian analysis on a set of returns for an asset

__Author = Terrill Toe
"""

import pandas as pd
import numpy as np
import seaborn as sns
import logging
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
from markov import MarkovChain
from utils import init_logger
from portfolio import PortfolioAnalysis
from datetime import datetime, timedelta, date
from pathlib import Path


class MarkovAnalysis(MarkovChain):
    """
    Class for markov analysis
    """

    def __init__(self, df: pd.DataFrame,
                 num_states: int,
                 ticker: str,
                 order: int= 1,
                 single_state_prediction: bool=True,
                 file_path: str = "./plots"):
        """
        Initialize the markov analysis class
        Parameters
        ________________
            df: pd.DataFrame
        the dataframe that will be used for analysis and constructing
        the transition matrix
            order: int
        The number of states to consider when constructing the
            single_state_prediction: bool
        in a higher order markov matrix predict only the next single state
        """
        # get the training data
        self.df = df
        self.ticker = ticker
        # self._order = order
        # self._num_states = num_states
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.mkdir(parents=True, exist_ok=True)
        super().__init__(n_states=num_states, order=order, single_state_prediction=single_state_prediction)
        self.encoded_variables = {}
        self.COLOR = "#50C878"

    def discretize(self,
                   method: str = 'categorical',
                   column_name: str = "log_returns",
                   bins: pd.IntervalIndex | int = 5,
                   labels: list = None) -> None:
        """
        Function discritizes a continuous random variable
        in the initialized dataset
        Parameters
        ______________
            method:
        The method for discretizing the data. Options are categorical or percentile
            column_name: str
        The name of the column in the dataset to be discritize
            bins: list
        a list of specified bins to discritize the data
            labels: list
        The list of labels for the discritized data. Must be the same length as bins
        """
        if method == 'categorical':
            try:
                self.df[f"{column_name}_discrete"] = pd.cut(x=self.df[column_name],
                                                            bins=bins,
                                                            labels=labels)
            except Exception as e:
                logging.error(f"issue with passed variables\n{e}")
                raise ValueError

        elif method == 'percentile':
            try:
                self.df[f"{column_name}_discrete"] = pd.qcut(x=self.df[column_name],
                                                             bins=bins,
                                                             labels=labels)

            except Exception as e:
                logging.error(f"issue with passed variables\n{e}")
                raise ValueError
        else:
            logging.error(f"Pass either categorical or percentile to method.")
            raise ValueError
        # label encoding
        # create encoded labels for the discretized methods
        label_encoder = preprocessing.LabelEncoder()
        self.df[f"{column_name}_labels"] = label_encoder.fit_transform(self.df[f"{column_name}_discrete"])
        # get the encoded variables
        logging.info("persisting the value names")
        self.encoded_variables = dict([(i, x) for i, x in enumerate(label_encoder.classes_)])

    def create_transition_mx(self,
                             states: str) -> None:
        """
        Create a transition matrix of n order
        Parameters
        ______________
            states: str
        the column that represents the current state in the transition matrix

        :return:
        """
        # fit the data to build the tranisiton matrix
        self.fit(self.df[states].tolist())

    def plot_transition_mx(self, persist: bool=False) -> None:
        """
        plot the transition matrix
        """
        df_plot = self.transition_df()
        if self.order > 1:
            annot = False
        else:
            annot = True
        sns.heatmap(data=df_plot, vmin=0.0, vmax=1.0, annot=annot).set_title(f"Transition matrix oder: {self.order}")
        if persist:
            plt.savefig(self.file_path / f"{self.ticker}transition_matrix.png", bbox_inches='tight')
        plt.show()

    def plot_states(self, col: str, persist: bool = False) -> None:
        """
        plot the states count
        """
        df_plot = self.df[col].value_counts().reset_index()
        df_plot.columns = [col, 'count']
        title_name = " ".join(col.split("_"))
        sns.barplot(df_plot, x=f"{col}", y="count", color=self.COLOR).set_title(f"{title_name.title()} Count")
        plt.xlabel("States")
        if persist:
            plt.savefig(self.file_path / f"{self.ticker}_states.png", bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    init_logger(file_name='markov_analysis.log')

    pd.options.mode.copy_on_write = True

    params = {
        'tickers': ['TSM'],
        'retention_prd': 500,
        'end_time': date(2024, 1, 6),
        'current_price': 103.2,
        'time_step': 40,
        'call_price': 104.41,
        'put_price': 101.3,
        'distribution': 'normal'
    }
    tickers = params['tickers']
    ints = PortfolioAnalysis(tickers=tickers, interval='1d')
    ints.calc_returns()
    end_date_time = datetime.combine(params['end_time'], datetime.min.time())
    retention_prd = params['retention_prd']
    start_date = params['end_time'] - timedelta(retention_prd)
    df_test = ints.historical_prices[ints.historical_prices['ticker'] == params['tickers'][0]].loc[
                           start_date.isoformat(): params['end_time'].isoformat()]
    logging.info(f"df_test: {df_test}")

    # create a list of interval index
    logging.info("Creating the bins")
    bins = pd.IntervalIndex.from_tuples([(-np.inf, -0.03), (-0.03, -0.015), (-0.015, 0.),
                                         (0.,0.015), (0.015, 0.03), (0.03, np.inf)])
    logging.info(f"bins length: {len(bins)}")
    markov_analysis = MarkovAnalysis(df=df_test,
                                     num_states=len(bins),
                                     ticker=params['tickers'],
                                     single_state_prediction=True,
                                     order=1)

    markov_analysis.discretize(bins=bins)
    # print(markov_analysis.df.isna().sum())
    # print(markov_analysis.df)
    markov_analysis.create_transition_mx(states="log_returns_labels")
    transition_matrix = markov_analysis.transition_df()
    transition_probs = transition_matrix.iloc[0, :]
    print(transition_probs)
    # print(df_test['adj close'])
    # df_test_appended = df_test['adj close']
    # current_price = 100.1
    # df_test_appended['2024-01-06'] = current_price
    # df_test_appended = df_test_appended.iloc[1:]
    # print(df_test_appended)
    # update the transition matrix
    # log_return = np.log(df_test_appended.iloc[-1]) - np.log(df_test_appended.iloc[-2])
    # markov_analysis.update
    print(markov_analysis.possible_states_lookup())
    print(markov_analysis.encoded_variables)
    markov_analysis.plot_transition_mx(persist=False)
    markov_analysis.plot_states(col="log_returns_labels", persist=False)
