# ---------------------------------------------------------------------------
# Run the trained DQN agent on the test set (no exploration).
# Fetches data (symbol, period), splits by train_ratio; runs agent on the last
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from data_loader import fetch_stock_data, train_test_split
from dqn_agent import DQNAgent
from trading_env import TradingEnv

# ---------------------------------------------------------------------------
# Run the trained DQN agent on the test set (no exploration).
# Fetches data (symbol, period), splits by train_ratio; runs agent on the last
# (1 - train_ratio) of the data and plots equity curve vs buy-and-hold.
#   model_path: path to the saved .pt model (from train.py)
#   symbol: stock ticker for fetching data (e.g. AAPL)
#   period: how much history to fetch from Yahoo Finance (e.g. "2y")
#   train_ratio: same as in training; first train_ratio of data is "train", rest is test
#   window_size: must match the env used during training
#   seed: for env.reset (deterministic start at beginning of test)
#   save_plot_path: if set, save the plot to this path and do not show interactively
# Returns: (values, actions_taken) — portfolio value at each step and list of actions.
# ---------------------------------------------------------------------------
def run_agent(
    model_path: Union[str, Path],
    symbol: str = "AAPL",
    period: str = "2y",
    train_ratio: float = 0.7,
    window_size: int = 10,
    seed: int = 42,
    save_plot_path: Optional[Union[str, Path]] = None,
):
    # check if the model path is a file
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 1) Load the test data
    df = fetch_stock_data(symbol=symbol, period=period)
    _, test_df = train_test_split(df, train_ratio=train_ratio)
    # check if the test set is too small to run the agent
    if len(test_df) < window_size + 5:
        raise ValueError("Test set too small")

    # 2) Create the environment and agent
    env = TradingEnv(test_df, window_size=window_size, random_start=False, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(str(model_path))

    # 3) Testing loop (no random start: start at beginning of test)
    obs, info = env.reset(seed=seed)
    values = [info["value"]]
    actions_taken = []
    # run the episode until it terminates
    while True:
        # select an action based on the current state
        action = agent.select_action(obs, training=False)
        # take the action and observe the next state, reward, and termination
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        # update the total reward
        values.append(info["value"])
        # if the episode is terminated, break the loop
        if terminated or truncated:
            break

    # 4) Calculate the returns and the buy-and-hold returns
    values = np.array(values)
    returns_agent = np.diff(values) / (values[:-1] + 1e-8)
    n = len(values)
    
    # Buy-and-hold over the same test window (same timesteps as agent)
    test_prices = test_df["Close"].values
    start_idx = window_size - 1  # first price we "start" at (aligned with env)
    bh_value_start = 100_000.0
    bh_prices = test_prices[start_idx : start_idx + n]
    bh_values = bh_value_start * (bh_prices / bh_prices[0])

    # 5) Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    steps = np.arange(len(values))
    ax1.plot(steps, values, label="DQN Agent", color="C0")
    ax1.plot(steps, bh_values, label="Buy & Hold", color="C1", alpha=0.8)
    ax1.set_ylabel("Portfolio value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Test set: DQN Trading Agent vs Buy & Hold")

    ax2.plot(steps[:-1], returns_agent, alpha=0.7, color="C0", label="Agent step returns")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("Step return")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_plot_path}")
        plt.close()
    else:
        plt.show()

    # 6) Print the total return and the final value
    total_return_agent = (values[-1] - values[0]) / (values[0] + 1e-8)
    total_return_bh = (bh_values[-1] - bh_values[0]) / (bh_values[0] + 1e-8)
    print(f"Agent total return: {total_return_agent:.4f} | Buy & Hold: {total_return_bh:.4f}")
    print(f"Agent final value: ${values[-1]:,.2f} | Initial: ${values[0]:,.2f}")
    return values, actions_taken


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/dqn_trading.pt", help="Path to saved model")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--plot", default="outputs/equity_curve.png", help="Where to save plot")
    args = parser.parse_args()

    run_agent(
        model_path=args.model,
        symbol=args.symbol,
        save_plot_path=args.plot,
    )
