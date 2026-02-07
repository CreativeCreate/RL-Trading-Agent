# Reinforcement Learning Trading Agent

A DQN-based agent that learns to make **buy / sell / hold** decisions in a simulated stock market environment.

## Quick start

**Installation**

```bash
pip install -r requirements.txt
```

**Train** the agent on historical data:

```bash
python train.py --symbol AAPL --period 2y --episodes 200
```

**Run** the trained agent on the test set and plot results:

```bash
python run_agent.py --model outputs/dqn_trading.pt --plot outputs/equity_curve.png
```

Use the same `--symbol` (and default `period`/`train_ratio`) as in training so the test split matches.

## Project structure

| File             | Purpose                                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| `data_loader.py` | Fetch stock data (Yahoo Finance), compute returns and features, train/test split by time                    |
| `trading_env.py` | Gymnasium environment: state (last 10 returns + position), actions (hold/buy/sell), reward (fractional PnL) |
| `dqn_agent.py`   | DQN with replay buffer, target network, epsilon-greedy exploration                                          |
| `train.py`       | Training loop; saves model and episode rewards to `outputs/`                                                |
| `run_agent.py`   | Load model, run on test set (no exploration), plot equity curve vs buy-and-hold                             |

## Run options

- **train.py**: `--symbol` (default AAPL), `--period` (default 2y), `--episodes` (default 200), `--seed` (default 42).
- **run_agent.py**: `--model` (path to saved `.pt`), `--symbol`, `--plot` (path to save the figure; if set, plot is saved and not shown interactively).

Other parameters (e.g. `train_ratio`, `window_size`) use the same defaults in both train and run; change them in code if needed.

## Outputs

- **outputs/** — created when you run `train.py`; contains `dqn_trading.pt` and `episode_rewards.npy`.
- **outputs/equity_curve.png** — created when you run `run_agent.py` with `--plot outputs/equity_curve.png`.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies (gymnasium, torch, pandas, yfinance, matplotlib, numpy).
