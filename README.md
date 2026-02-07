# Assignment 3: Reinforcement Learning Trading Agent

A DQN-based agent that learns to make **buy / sell / hold** decisions in a simulated stock market environment.

## Quick start

```bash
cd assignment_03
pip install -r requirements.txt
```

**Train** (requires network for Yahoo Finance data):

```bash
python train.py --symbol AAPL --period 2y --episodes 200
```

**Run** the trained agent and plot results:

```bash
python run_agent.py --model outputs/dqn_trading.pt --plot outputs/equity_curve.png
```

## Project structure

| File | Purpose |
|------|--------|
| `data_loader.py` | Fetch Yahoo Finance data; train/test split |
| `trading_env.py` | Gymnasium environment: state, actions, rewards |
| `dqn_agent.py` | DQN with replay buffer and target network |
| `train.py` | Training loop; saves model and rewards to `outputs/` |
| `run_agent.py` | Load model, run on test set, plot equity curve vs buy-and-hold |

## Run options

- **train.py**: `--symbol`, `--period`, `--episodes`, `--seed`
- **run_agent.py**: `--model`, `--symbol`, `--plot`

## Learning guide

See **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** for a step-by-step explanation of what we built and how it works.
