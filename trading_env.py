# -------------------------------------------------------------
# Gymnasium environment for stock trading.
# State: last N daily returns + current position (0 = cash, 1 = long).
# Actions: 0 = hold, 1 = buy (go long), 2 = sell (go to cash).
# Reward: change in portfolio value (PnL) at each step.
# -------------------------------------------------------------
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TradingEnv class
# df: DataFrame with at least 'Return' and 'Close' columns.
# window_size: number of past returns used as state.
# initial_balance: starting cash.
# random_start: if True, start at a random valid index (for training); else start at window_size.
# seed: random seed.
# ---------------------------------------------------------------------------
class TradingEnv(gym.Env):

    # Initialize the TradingEnv class
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_balance: float = 100_000.0,
        random_start: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.random_start = random_start

        # Indices we can step from: need at least window_size history
        self.valid_start = window_size
        self.valid_end = len(self.df) - 1  # we need at least one step ahead

        # State: [last `window_size` returns, current_position (0 or 1)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size + 1,),
            dtype=np.float32,
        )
        # Actions: 0 hold, 1 buy, 2 sell
        self.action_space = gym.spaces.Discrete(3)

        self._seed = seed
        self._current_idx = None
        self._position = 0  # 0 cash, 1 long
        self._cash = initial_balance
        self._shares = 0.0
        self._last_value = initial_balance

    # ---------------------------------------------------------------------------
    # Reset the environment
    # seed: random seed.
    # options: options. Unused here; kept for Gymnasium’s API
    # Returns tuple of observation and info.
    # ---------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set the random seed if provided
        if self._seed is not None:
            np.random.seed(self._seed)
        # choose start index for the episode
        # if random_start, choose a random index between valid_start and valid_end
        # otherwise, start at valid_start
        if self.random_start and self.valid_end > self.valid_start:
            self._current_idx = int(self.np_random.integers(self.valid_start, self.valid_end + 1))
        else:
            self._current_idx = self.valid_start
        # reset portfolio state, so every episode starts with a clean slate
        self._position = 0
        self._cash = self.initial_balance
        self._shares = 0.0
        self._last_value = self.initial_balance
        # build and return the initial observation and info
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ---------------------------------------------------------------------------
    # Get the observation (state)
    # builds and returns the observation vector for the current step and position
    # ---------------------------------------------------------------------------
    def _get_obs(self):
        # index of the first return
        start = self._current_idx - self.window_size
        # returns for the last window_size days
        returns = self.df["Return"].iloc[start : self._current_idx].values.astype(np.float32)
        # current position as single float - [0.0] for cash, [1.0] for long
        pos = np.array([float(self._position)], dtype=np.float32)
        # concatenate "returns" and "position" into a single observation vector
        return np.concatenate([returns, pos])

    # ---------------------------------------------------------------------------
    # Get the info
    # builds and returns the info dict - value, position, step
    # ---------------------------------------------------------------------------
    def _get_info(self):
        # get the current price
        price = self.df["Close"].iloc[self._current_idx]
        # get the current value of the portfolio
        value = self._cash + self._shares * price
        return {"value": value, "position": self._position, "step": self._current_idx}

    # ---------------------------------------------------------------------------
    # Get the portfolio value
    # calculates and returns the current value of the portfolio
    # ---------------------------------------------------------------------------
    def _portfolio_value(self):
        current_price = self.df["Close"].iloc[self._current_idx]
        return self._cash + self._shares * current_price

    # ---------------------------------------------------------------------------
    # Step the environment:
    # 1. Execute action at the current day (update portfolio: buy/sell/hold).
    # 2. Move to the next day.
    # 3. Reward = fractional PnL (change in portfolio value over the step).
    # 4. Return Gymnasium convention: (observation, reward, terminated, truncated, info)
    #    observation: the state *after* the step (for the next decision)
    #    reward: fractional PnL from this step
    #    terminated: True when we reach the end of the data
    #    truncated: not used (always False)
    #    info: dict with value, position, step, reward
    # ---------------------------------------------------------------------------
    def step(self, action: int):
        # Close price at the current day (the day we're about to act on)
        price = self.df["Close"].iloc[self._current_idx]
        # Portfolio value before applying the action
        prev_value = self._portfolio_value()
        
        # Update portfolio based on the action. 0 = hold, 1 = buy, 2 = sell
        # We keep it simple: no short-selling, no partial positions.
        if action == 1 and self._position == 0:
            # Buy: spend all cash on shares
            self._shares = self._cash / price
            self._cash = 0.0
            self._position = 1
        elif action == 2 and self._position == 1:
            # Sell: liquidate
            self._cash = self._shares * price
            self._shares = 0.0
            self._position = 0

        # Move to the next day
        self._current_idx += 1
        # Portfolio value after the action
        now_value = self._portfolio_value() 
        # Calculate the reward: fractional PnL (change in portfolio value)
        # the PnL signal the agent learns from.
        # Positive = portfolio gained, negative = portfolio lost. 
        # +1e-8 to avoid division by zero
        reward = (now_value - prev_value) / (prev_value + 1e-8)  

        # Check if the episode is terminated
        # Episode is terminated when we reach the end of the data
        terminated = self._current_idx >= self.valid_end
        truncated = False # not used, always False; kept for Gymnasium’s API
        obs = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        return obs, float(reward), terminated, truncated, info
