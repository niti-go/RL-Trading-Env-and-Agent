import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
#import pandas_datareader.data as web
import random
import matplotlib.pyplot as plt
import yfinance


class TradingEnv(gym.Env):
    def __init__(self, ticker='AAPL', start_date='2019-01-01', end_date='2022-01-01', transaction_cost=0.001, window_size=50):
        super(TradingEnv, self).__init__()

        # Download historical stock data
        #this is a dataframe containing a few columns: dj Close, Return, SMA50
        self.data = self.load_data(ticker, start_date, end_date)
        
        # Parameters
        self.ticker = ticker
        self.transaction_cost = transaction_cost #this is a percentage of a transaction
        self.window_size = window_size  # Number of past days to use as features

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell (can only buy/sell 1 share at a timestep)
        self.action_space = spaces.Discrete(3)

        # Observation space: a state contains 5 features for each day in the window
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, 5), dtype=np.float32)

        self.metadata={"render_modes":["human"]}
        self.bought_prices = [] #store the prices we bought shares at
        self.current_step = 0 # the current day that the agent can buy/sell/hold
        self.initial_balance = 10000 # Starting cash, can make this a parameter
        self.balance = 10000  
        self.shares_held = 0  # Number of shares owned
        
        self.done = False

    def load_data(self, ticker, start_date, end_date):
        ticker = yfinance.Ticker(ticker)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        df['Return'] = df['Close'].pct_change()  # Daily returns
        df['SMA50'] = df["Close"].rolling(window=50).mean() # 50-day simple moving average
        #df['SMA200'] = df["Close"].rolling(window=200).mean()
        df.dropna(inplace=True)
        #return df[['Close', 'Return', 'SMA50', 'SMA200']]
        return df[['Close', 'Return', 'SMA50']]

    def reset(self):
        '''
        Resets the balance etc. and current time step to window_size
        Returns the first observation (state)
        '''
        self.current_step = self.window_size  # Start after the window size. 0 is the start date.
        self.balance = self.initial_balance
        self.shares_held = 0
        self.bought_prices = []
        self.done = False
        return self.get_observation()

    def get_observation(self):
        '''
        Returns the current state (window_size, 5) array of data for the current timestep
        '''
        # A (window_size, 5) array of data, which is the state
        # Use the last 'window_size' days of data for the current observation
        current_frame = self.data.iloc[self.current_step - self.window_size:self.current_step]
        return current_frame.to_numpy()

    def step(self, action):
        # Get current price and features
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0
        done = False

        # Calculate transaction cost of trading 1 share
        transaction_cost = self.transaction_cost * current_price
        
        reward = 0
        if action == 1:  # Buy
            if self.balance >= current_price + transaction_cost:
                self.shares_held += 1
                self.balance -= current_price + transaction_cost
                self.bought_prices.append(current_price + transaction_cost)
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price - transaction_cost
                #reward is the immediate profit when selling a share - the cost it was bought at
                reward = (current_price - transaction_cost) - self.bought_prices.pop(0)
                #this above is realized gains only i think. but unrealized gains
                #are incorporated because at the end you can sell all your shares??
                #but actually you can't because you can only buy/sell one share at each time step

        ## Update balance and portfolio value
        ##Reward is the immediate profit/loss from this individual trade  
        #portfolio_value = self.balance + self.shares_held * current_price
        #reward = portfolio_value - self.initial_balance  # Reward is the change in portfolio value

        # Move to the next step
        self.current_step += 1

        # Done if we are at the end of the dataset
        if self.current_step >= len(self.data) - 1:
            done = True

        return self.get_observation(), reward, done, {"portfolio value": self.balance + self.shares_held * current_price}

    def render(self):
        # Simple visualization of the balance and portfolio value
        portfolio_value = self.balance + self.shares_held * self.data.iloc[self.current_step]['Close']
        print(f"Day: {self.current_step}, Balance: {self.balance:.2f}, Shares Held: {self.shares_held}, Portfolio Value: {portfolio_value:.2f}")

if __name__ == "__main__":
    env = TradingEnv(ticker="AAPL", start_date="2024-01-01", end_date="2025-01-01")
    #print (env.data)

    state = env.reset()
    done = False
    total_reward = 0

    # Run a simple random agent for demonstration
    while not done:
        action = random.choice([0, 1, 2])  # Random action: 0 = Hold, 1 = Buy, 2 = Sell
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total Reward: {total_reward}")
