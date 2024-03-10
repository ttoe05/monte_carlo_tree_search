import pandas as pd
import numpy as np
import logging
import random
from math import sqrt
from scipy.stats import t as t_dist
from datetime import date, timedelta
from MarkovAnalysis import MarkovAnalysis
from scipy.stats import norm, ecdf
from copy import deepcopy
from portfolio import PortfolioAnalysis
from utils import init_logger_json
from datetime import datetime
from random import uniform


"""
This class method creates the root node of the monte carlo tree search algorithim
Based on the run over time, the node will expand its branches to other child nodes based on the decisition
the parent node came to
"""


class MCTSNode:
    """
    Class initializes a parent node and the parent node will create
    subsequent child nodes to make informed decisions
    """
    def __init__(self,
                 call_price: float,
                 put_price: float,
                 call_strike: float,
                 put_strike: float,
                 current_price: float,
                 time_to_expiry: int,
                 # eval_date: date,
                 prices: any,
                 states: pd.IntervalIndex,
                 max_tree: int = 3,
                 state: any = None,
                 order: int = 1,
                 action: int = None,
                 # initial_price: float = None,
                 node_num: int = 0,
                 num_simulations: int = 10000,
                 risk_ratio: float = 0.0,
                 retention_prd: int = 400,
                 time_decay_alpha: float = 0.01,
                 root: bool = True) -> None:
        """
        Initialize the root parent node
        """
        self._root = root
        self.node_num = node_num
        self.time_to_expiry = time_to_expiry
        self.has_children = False
        self.call_price = call_price
        self.put_price = put_price
        self.call_strike = call_strike
        self.put_strike = put_strike
        # self.initial_price = current_price if not initial_price else initial_price
        self.current_price = current_price
        self.action = action
        self.states = states
        self.order = order
        self.max_tree = max_tree
        self.time_decay_alpha = time_decay_alpha
        # self.eval_date = eval_date
        self.retention_prd = retention_prd
        self.risk_ratio = risk_ratio
        self.position_returns = [0,]
        self.prices = prices.to_frame(name='price') if type(prices) == pd.Series else prices
        self.num_simulations = num_simulations
        self.df_monte = pd.DataFrame()
        self.df_probs = pd.DataFrame()
        self.ticker = "None"
        self.children = {}
        self.logging_extra = {"time_to_expiry": self.time_to_expiry,
                              "action": self.action,
                              "current_price": self.current_price,
                              "call_price_thresh": self.call_price,
                              "put_price_thresh": self.put_price,
                              # "state": self.state,
                              "risk_ratio": self.risk_ratio,
                              "root_node": self._root,
                              "max_tree": self.max_tree,
                              "node_num": self.node_num}
        self.log_returns = self._calc_returns()
        print(f"Log returns calculated {self.log_returns}")
        self.markov = self._build_transition_matrix() if self._root else None
        self.transition_matrix = self.markov.transition_df() if self._root else pd.DataFrame()
        self.state = self._get_root_node_state() if self._root and not state else state


        logging.info(f"Node initialized: {self.node_num}", extra=self.logging_extra)

    def _get_root_node_state(self):
        """
        gets the state for the root node. upon initilization
        """
        print("initializing root node state")
        return_val = np.log(self.current_price) - np.log(self.prices['price'].iloc[-1])
        for state, value in self.markov.encoded_variables.items():
            if return_val in value:
                return state


    def _calc_returns(self):
        """
        Calculate the log returns on the prices
        :return:
        """
        log_returns = np.log(self.prices['price']) - np.log(self.prices['price'].shift(1))
        # convert the series to dataframe
        log_returns_df = pd.DataFrame({'Date': log_returns.index, 'log_returns': log_returns.values})
        logging.info(f"Log returns created for node {self.node_num}", extra=self.logging_extra)
        return log_returns_df.dropna()

    def student_t_fit(self):
        """
        Function fits the returns to a student t distirbuiton to get the mean,
        std, and degrees of freedom
        :return:
            degree_freedoms, mean, standard deviation
        """
        degree_freedoms, m, v = t_dist.fit(self.log_returns['log_returns'])
        logging.info(f"Student t ran for node {self.node_num}", extra=self.logging_extra)
        # run the ktest to see if the distribution fits
        # TODO: add functionality to test the fit. If it does not return false
        return degree_freedoms, True

    def _monte_carlo_process(self, time_step: int,
                            current_price: float,
                            num_simulation: int,
                            degree_freedoms: float = None,
                            student_t: bool = False ) -> None:
        """
        Monte carlo simulation of log returns for a given stock over n time cadence
        :param current_price: the current price of the stock
        :param time_step: time steps into the future
        :param num_simulation: number of simulations
        :param student_t: if true use a student t distribution else normal
        :return: pd.DataFrame
        """
        # get the moments from the log returns
        mean = self.log_returns['log_returns'].mean()
        variance = self.log_returns['log_returns'].var()
        standard_dev = self.log_returns['log_returns'].std()
        # generate a list of random log returns
        drift = mean - (0.5 * variance)
        if student_t:
            Z = t_dist.ppf(np.random.rand(time_step, num_simulation), degree_freedoms)
        else:
            Z = norm.ppf(np.random.rand(time_step, num_simulation))
        log_returns = np.exp(drift + standard_dev * Z)
        price_paths = np.zeros_like(log_returns)
        price_paths[0] = current_price
        for t in range(1, time_step):
            price_paths[t] = price_paths[t - 1] * log_returns[t]

        predictions = price_paths.T
        columns = []
        for i in range(time_step):
            columns.append(f'day_{i + 1}')

        df_predictions = pd.DataFrame(predictions, columns=columns)
        self.df_monte = df_predictions
        logging.info(f"Ran simulation using monte-carlo for node {self.node_num}", extra=self.logging_extra)

    def get_ecdf(self, col_idx: int = -1):
        """
        Get the empirical cumulative distribution of the simulated results
        """
        col = self.df_monte.columns[col_idx]
        values = self.df_monte[col].to_numpy()
        # print(f'calculating the ecdf for {col}:\t{values[:10]} ')
        predictions_ecdf = ecdf(values)
        self.df_probs = pd.DataFrame({'quantiles': predictions_ecdf.sf.quantiles,
                                      'probs': predictions_ecdf.sf.probabilities})

    def calc_trade_prob(self) -> tuple:
        """
        Calculate the win probability
        :return: float
            prob win, prob loss
        """
        # filter the probs dataframe
        try:
            call_prob = self.df_probs[self.df_probs['quantiles'] >= self.call_price]['probs'].values[0]
        except:
            call_prob = 0.0001
        try:
            put_prob = 1 - (self.df_probs[self.df_probs['quantiles'] <= self.put_price]['probs'].values[-1])
        except:
            put_prob = 0.0001
        logging.info(f"Call probability calculated\ncall_prob: {call_prob}, put_prob: {put_prob}, trade_prob: {call_prob + put_prob}",
                     extra=self.logging_extra)
        return (call_prob + put_prob), call_prob, put_prob

    def _get_absolute_function_params(self):
        """
        :return:
        """
        # get the units between the two points
        units = self.call_price - self.put_price
        # divide units by 2
        half_units = units / 2
        # get the middle point
        middle_point = self.put_price + half_units
        # get the slope
        slope = (0 - (-1)) / (self.call_price - middle_point)
        shift_right = middle_point
        return slope, shift_right

    def calc_risk_return(self):
        """
        Function uses the absolute function to calculate the current risk or return of the current
        straddle positon
        :return:
        """
        # get the absolute value function parameters
        v_slope, v_shift_right = self._get_absolute_function_params()
        # calculate the theoritical log returns value
        log_return_theoritical = v_slope * abs(self.current_price - v_shift_right) - 1
        return log_return_theoritical


    def expansion_policy(self) -> int:
        """
        The expansion policy is defined as to how to decide whether or not to select a specific node. We define the UCT
        as the following:
            1. simulate the projected prices up to the expiration date based on the current price the stock
            2. Get the probability of profit. If the probability of profit is greater than 50% accept the node
            3. get the expected risk to reward ratio
            4. If the risk -reward ratio is 1 to 1 and the probability of profit is greater than 50% accept the node
            5. If either condition is not met, evaluate selling the
        Returns int
            0 - sell, 1-unsure so expand, 2 - hold
        """
        # fit a student t test to the log returns to get the df and see if it fits
        degree_freedom, fit = self.student_t_fit()
        # run the simulation up to expiry to get the distribution of future prices
        self._monte_carlo_process(time_step=self.time_to_expiry,
                                  current_price=self.current_price,
                                  num_simulation=self.num_simulations,
                                  degree_freedoms=degree_freedom,
                                  student_t=fit)
        self.get_ecdf()
        # get the probability of profit
        trade_prob, call_prob, put_prob = self.calc_trade_prob()
        # get the risk to reward ratio
        # risk_ratio = self.calc_risk_ratio(call_prob=call_prob, put_prob=put_prob)
        uct_score = 0
        # make a decision
        if trade_prob >= 0.5 and trade_prob < 0.6:
            uct_score = 1
        elif trade_prob >= 0.6:
            uct_score = 2
        else:
            uct_score = 0
        # if risk_ratio > 0:
        #     uct_score += 1

        logging.info(f"trade_prob: {trade_prob}, call_prob: {call_prob}, put_prob: {put_prob}, UCT_score: {uct_score}",
                     extra=self.logging_extra)
        return uct_score

    def _build_transition_matrix(self) -> MarkovAnalysis:
        """
        Builds the transition matrix based on the states
        :return:
        """
        num_states = len(self.states)
        markov_analysis = MarkovAnalysis(df=self.log_returns,
                                         num_states=num_states,
                                         ticker=self.ticker,
                                         order=self.order,
                                         single_state_prediction=True)
        print(f"created the markov object {markov_analysis}")
        print(f"markov object {markov_analysis.df.head()}")
        markov_analysis.discretize(bins=self.states)
        print(f"discretizing")
        markov_analysis.create_transition_mx(states="log_returns_labels")
        # return the object to keep functions like updating the transition mx
        logging.info(f"Markov object created for node {self.node_num}", extra=self.logging_extra)
        return markov_analysis

    def get_next_state(self) -> int:
        """
        Gett the next state based on the transition matrix
        :return:
        """
        # get the conditional probabilities
        print(f"Getting the transition matrix to select the next state: \n{self.transition_matrix}")
        print(f"current state: {self.state}")
        print(f"transition probs for state {self.state}: {self.transition_matrix.iloc[self.state, :]}")
        transition_probs = self.transition_matrix.iloc[self.state, :]
        # check if the probs equals to 0
        if transition_probs.sum() == 0.:
            # randomly select a state from the transition probs
            state = random.choice(transition_probs.tolist())
        else:
            state = transition_probs.idxmax()
        # check the type of the index, can be tuple or integer depending on single_state parameter
        if type(state) is tuple:
            state = state[0]

        logging.info(f"current state: {self.state}, predicted next state: {state}", extra=self.logging_extra)
        return state

    def select_random_value_state(self, state: int) -> float:
        """
        Select a random value from the state's bin
        """
        # get the left and right bounds
        print(f"printing encoded variables for the transition matrix: {self.markov.encoded_variables}\nstate: {state} - {type(state)}")
        right_bound = self.markov.encoded_variables[state].right
        left_bound = self.markov.encoded_variables[state].left
        # select a random value from the past log returns
        return_choices = self.log_returns[
            (self.log_returns["log_returns"] >= left_bound) & (self.log_returns["log_returns"] < right_bound)
            ]["log_returns"].tolist()
        return random.choice(return_choices)

    def run_uct(self) -> int:
        """
        Function runs the uct policy that is defined as calculating the next future states potential return of the position
            1. get the next state based on the current state
            2. select a random log return from the historical returns that falls within the states bin
            3. calculate the future price using the log return
            4. get the future position return value using the absolute function
            5. compare to the current position return value to see if it improved or got worse
            6. select the node based on the improvement or not
        Return:
            int
                0 - hold, 1 -sell

        """
        # calculate the current position return
        current_position_return = self.calc_risk_return()
        # append the positon return to the list
        self.position_returns.append(current_position_return)
        # check if the current position return is less than the past position return
        if self.position_returns[-1] < self.position_returns[-2]:
            return 1
        else:
            return 0

    def select_node(self):
        """
        Based on the UCT of the parent node select the best node
        1. select the node based on the UCT policy
        2. set the action of the child node based on the UCT policy
        3. predict the next state and select a random value from the state
        4. calculate the next days price based on the random return value
        3. create the child node
        """
        # run the UCT policy
        uct_policy = self.run_uct()
        next_state = self.get_next_state()
        # randomly select a value between the IntervalIndex of the states
        random_value = self.select_random_value_state(state=next_state)
        # calculate the future price using the log return
        future_price = self.current_price * (np.exp(random_value) + 1)
        # create a child node if it does not exist
        if not self.has_children:
            # create the child
            self.children[self.action] = MCTSNode(call_price=self.call_price,
                                                  put_price=self.put_price,
                                                  call_strike=self.call_strike,
                                                  put_strike=self.put_strike,
                                                  current_price=future_price,
                                                  time_to_expiry=self.time_to_expiry,
                                                  # eval_date=self.eval_date,
                                                  prices=self.prices,
                                                  states=None,
                                                  state=next_state,
                                                  order=self.order,
                                                  action=self.action,
                                                  node_num=self.node_num + 1,
                                                  # initial_price=self.initial_price,
                                                  num_simulations=self.num_simulations,
                                                  risk_ratio=self.risk_ratio,
                                                  retention_prd=self.retention_prd,
                                                  root=False)

            return_childe_node = self.children[self.action]
            self.has_children = True
        elif self.action in self.children.keys():
            return_childe_node = self.children[self.action]
        else:
            # create the child if it has not been created
            self.children[self.action] = MCTSNode(call_price=self.call_price,
                                                  put_price=self.put_price,
                                                  call_strike=self.call_strike,
                                                  put_strike=self.put_strike,
                                                  current_price=future_price,
                                                  time_to_expiry=self.time_to_expiry,
                                                  # eval_date=self.eval_date,
                                                  prices=self.prices,
                                                  states=None,
                                                  max_tree=self.max_tree,
                                                  state=next_state,
                                                  order=self.order,
                                                  action=self.action,
                                                  node_num=self.node_num + 1,
                                                  # initial_price=self.initial_price,
                                                  num_simulations=self.num_simulations,
                                                  risk_ratio=self.risk_ratio,
                                                  retention_prd=self.retention_prd,
                                                  root=False)
            return_childe_node =  self.children[self.action]

        logging.info(f"New child node created by node: {self.node_num} child_node: {return_childe_node.node_num}",
                     extra=self.logging_extra)
        return return_childe_node

    def _calc_time_decay(self):
        """
        Function calculates the time decay effect on the call_price and put_price thresholds
        to break even as time decreases

        resource: https://www.wallstreetmojo.com/time-decay/
        """
        # calculate the time value of the call option
        time_value_call = (self.call_strike - self.current_price) / self.time_to_expiry
        time_value_put = (self.current_price - self.put_strike) / self.time_to_expiry
        # check if the time value is negative or positive
        if time_value_call < 0:
            # convert the time value to positive and add alpha
            time_value_call = sqrt(time_value_call ** 2) * self.time_decay_alpha
        if time_value_put < 0:
            # convert the time value to positive and add alpha
            time_value_put = sqrt(time_value_put ** 2) * self.time_decay_alpha
        # return the time value
        logging.info(f"applying time decay value to the thresholds: {time_value_call}, {time_value_put}",
                     extra=self.logging_extra)
        return time_value_call, time_value_put

    def update_price_returns(self, date: str|pd.DatetimeIndex, price: float) -> None:
        """
        update the prices array with the updated price
        :param price:
        :param date: the date of the updated price
        :return:
        """
        # set the current price
        self.current_price = price
        # add the new price to the price series
        # self.prices[date] = price
        data = {'price': price}
        self.prices = pd.concat([self.prices, pd.DataFrame(data, index=[date])])
        # remove the top row to keep the array size at the specified retention period length
        self.prices = self.prices.iloc[1:]
        # update the log returns
        log_return = np.log(self.current_price) - np.log(self.prices.iloc[-2])

        row = {'Date': date, 'log_returns': log_return}
        self.log_returns[len(self.log_returns)] = row
        self.logging_extra['current_price'] = self.current_price
        logging.info(f"Node {self.node_num} updated prices, updated_current_price: {self.current_price},calculated_retrun: {log_return}",
                     extra=self.logging_extra)

    def update_transition_matrix(self) -> None:
        """
        update the transition matrix with the updated new information.
        :return:
        """
        self.markov.df = self.log_returns
        self.markov.discretize(bins=self.states)
        self.markov.create_transition_mx(states="log_returns_labels")
        self.transition_matrix = self.markov.transition_df()
        logging.info(f"Node {self.node_num} updated the transition matrix {self.transition_matrix}",
                     extra=self.logging_extra)

    def update_time_expiry(self) -> None:
        """
        Update the time to expiry of the contracts
        :return:
        """
        self.time_to_expiry -= 1
        self.logging_extra['time_to_expiry'] = self.time_to_expiry
        logging.info(f"Node {self.node_num} updated the time to expiry", extra=self.logging_extra)

    def update_thresholds(self) -> None:
        """
        update the thresholds based on the time decay
        """
        # get the time decay values
        time_decay_call, time_decay_put = self._calc_time_decay()
        self.call_price += time_decay_call
        self.put_price -= time_decay_put
        self.logging_extra['call_price_thresh'] = self.call_price
        self.logging_extra['put_price_thresh'] = self.put_price

    def pass_node_updates(self, node_obj) -> None:
        """
        Function passes the updates from the parent down to the
        :return:
        """
        # update attributes the node should have
        node_obj.time_expiry = self.time_to_expiry
        node_obj.markov = self.markov
        # node_obj.current_price = self.current_price
        node_obj.position_returns = self.position_returns
        node_obj.states = self.states
        logging.info(msg=(
            f"passing updates from Node: {self.node_num} to Node: {node_obj.node_num}"
            f"child_time_expiry: {node_obj.time_expiry}, child_current_price: {node_obj.current_price},"
            f"child_risk_ratio: {node_obj.risk_ratio}"
        ), extra=self.logging_extra)

    def update_tree_length(self) -> None:
        """
        Function updates the tree depth according to what is remaining for the time to
        expiry
        """
        if self.time_to_expiry <= self.max_tree:
            self.max_tree -= 1
            self.logging_extra['max_tree'] = self.max_tree

    def tree_traverse(self, node, tree_depth=0):
        """
        traverse the tree
        """
        tree_depth_val = tree_depth + 1
        # update the node with the next days information
        updated_price = self.prices.loc[index_val.isoformat(), ['price']].values[0]
        node.update_price_returns(price=updated_price, date=index_val)
        # node.update_price_returns(price=out_sample_prices.iloc[index_val].values[0], date=index_val)
        node.update_time_expiry()
        node.update_thresholds()
        node.update_transition_matrix()
        node.update_tree_length()
        expand_policy = node.expansion_policy()
        if expand_policy == 2:
            node.action = 0
            logging.info(f"expansion policy is 2, action is holding", extra=self.logging_extra)
            return 0
        elif expand_policy == 0:
            node.action = 1
            logging.info(f"expansion policy is 0, action is sell", extra=self.logging_extra)
            return 1
        else:
            child_node = node.select_node()
            self.pass_node_updates(node_obj=child_node)
            logging.info(f"expansion policy is 1, selected a child node", extra=self.logging_extra)
            return node.tree_traverse(node=child_node, tree_depth=tree_depth_val)



if __name__ == "__main__":
    #initialize logging
    init_logger_json(file_name="mcts_testing")
    # get the price of the ticker
    params = {
        'tickers': ['TSM'],
        'retention_prd': 400,
        'end_time': date(2024, 1, 16),
        'current_price': 103.2,
        'time_step': 16,
        'call_price': 104.41,
        'put_price': 101.3,
        'call_strike': 104.0,
        'put_strike': 103.0,
        'distribution': 't_dist'
    }
    bins = pd.IntervalIndex.from_tuples([(-np.inf, -0.03), (-0.03, -0.015), (-0.015, 0.),
                                         (0., 0.015), (0.015, 0.03), (0.03, np.inf)])
    tickers = params['tickers']
    tsm = PortfolioAnalysis(tickers=tickers, interval='1d')
    end_date_time = datetime.combine(params['end_time'], datetime.min.time())
    retention_prd = params['retention_prd']
    start_date = params['end_time'] - timedelta(retention_prd)
    print(f"Date range: {start_date.isoformat()} - {params['end_time'].isoformat()}")

    prices = tsm.historical_prices[tsm.historical_prices['ticker'] == params['tickers'][0]]['adj close'].loc[
               start_date.isoformat(): params['end_time'].isoformat()]
    # get the next 10 day of prices
    start_time = params['end_time'] + timedelta(days=2)
    end_time = start_time + timedelta(days=params['time_step'])
    # out of sample
    out_sample_prices = tsm.historical_prices[tsm.historical_prices['ticker'] == params['tickers'][0]]['adj close'].loc[
                        start_time.isoformat(): date(2024, 2, 9).isoformat()]
    print(f"prices being passed to the root node {prices}")
    print(f"Creating the root node")
    mcts_root_node = MCTSNode(call_price=params['call_price'],
                              put_price=params['put_price'],
                              call_strike=params['call_strike'],
                              put_strike=params['put_strike'],
                              current_price=params['current_price'],
                              time_to_expiry=len(out_sample_prices.index),
                              prices=prices,
                              states=bins,
                              order=2)
    print(f"root node created")
    print(f"Model evaluating time period {start_time.isoformat()} to {end_time.isoformat()}")
    # convert to a datafrane
    out_sample_prices = out_sample_prices.to_frame(name='price')
    out_sample_prices["mcts_decision"] = np.nan
    series_indx = 0
    # iterate over the prices for evaluation
    for index_val in out_sample_prices.index:
        print(f"indexes of prices: {mcts_root_node.prices.index}")
        print(f"root node prices type: {type(mcts_root_node.prices)}")
        # update the node with the next days information
        print(f"Index value: {index_val}")
        updated_price = out_sample_prices.loc[index_val.isoformat(), ['price']].values[0]
        mcts_root_node.update_price_returns(price=updated_price, date=index_val)
        if series_indx > 0:
            # update the time to expiry and the thresholds
            print("updating the root node time to expiry and thresholds")
            mcts_root_node.update_time_expiry()
            mcts_root_node.update_thresholds()
        mcts_root_node.update_transition_matrix()
        mcts_root_node.update_tree_length()
        expansion_policy = mcts_root_node.expansion_policy()
        # if policy is 2 hold else expand node
        if expansion_policy == 2:
            print(f"Root node selected action 0 because expansion policy is {expansion_policy}")
            mcts_root_node.action = 0
        elif expansion_policy == 0:
            print(f"Root node selected action 1 because expansion policy is {expansion_policy}")
            mcts_root_node.action = 1
        else:
            child_node = mcts_root_node.select_node()
            mcts_root_node.pass_node_updates(node_obj=child_node)
            tree_value = mcts_root_node.tree_traverse(node=child_node)
            mcts_root_node.action = tree_value
            if tree_value == 1:
                print(f"Chile node selected action 1")
            else:
                print(f"Chile node selected action 1 because UCT policy is {tree_value}")
        # populate the price data
        out_sample_prices.loc[index_val, "mcts_decision"] = mcts_root_node.action
        series_indx += 1

    print(out_sample_prices["mcts_decision"].tolist())
    print(out_sample_prices.index.tolist())