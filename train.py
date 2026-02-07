# ---------------------------------------------------------------------------
# Training script for the DQN trading agent.
# Loads data, creates environment and agent, runs episodes, saves model and training stats.
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Optional

import numpy as np

from data_loader import fetch_stock_data, train_test_split
from dqn_agent import DQNAgent
from trading_env import TradingEnv

# ---------------------------------------------------------------------------
# Train the DQN trading agent
# loads the data, creates the environment and agent, runs episodes, and saves the model and the episode rewards.
# symbol: the stock symbol to train on
# period: the period of time to train on
# train_ratio: the ratio of the data to use for training
# window_size: the size of the window to use for the state
# episodes: the number of episodes to train for
# target_update_every: the number of episodes to update the target network
# save_dir: the directory to save the model and the episode rewards
# seed: the seed to use for the random number generator
# returns the agent, environment, episode rewards, and save directory
# ---------------------------------------------------------------------------
def train(
    symbol: str = "AAPL",
    period: str = "2y",
    train_ratio: float = 0.7,
    window_size: int = 10,
    episodes: int = 200,
    target_update_every: int = 10,
    save_dir: Optional[Path] = None,
    seed: int = 42,
):
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "outputs"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load the training data
    df = fetch_stock_data(symbol=symbol, period=period)
    train_df, _ = train_test_split(df, train_ratio=train_ratio)
    print(f"Training on {len(train_df)} days ({symbol})")

    # 2) Create the environment and agent
    env = TradingEnv(train_df, window_size=window_size, random_start=True, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        buffer_size=20_000,
        batch_size=64,
        hidden_dim=64,
    )

    # 3) Training loop
    episode_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        # run the episode until it terminates
        while True:
            # select an action based on the current state
            action = agent.select_action(obs, training=True)
            # take the action and observe the next state, reward, and termination
            next_obs, reward, terminated, truncated, info = env.step(action)
            # store the transition in the replay buffer
            agent.store(obs, action, reward, next_obs, terminated or truncated)
            # update the Q-network
            loss = agent.update()
            # update the current state
            obs = next_obs
            # update the total reward
            total_reward += reward
            # update the number of steps
            steps += 1
            # if the episode is terminated, break the loop
            if terminated or truncated:
                break
        # decay the exploration rate
        agent.decay_epsilon()
        # update the target network
        if (ep + 1) % target_update_every == 0:
            agent.sync_target()
        # append the total reward to the list of episode rewards
        episode_rewards.append(total_reward)
        # print the average reward of the last 20 episodes
        if (ep + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {ep + 1}/{episodes} | Steps: {steps} | Avg reward (last 20): {avg:.6f} | ε: {agent.epsilon:.3f}")

    # 4) Save the model and the episode rewards
    model_path = save_dir / "dqn_trading.pt"
    agent.save(str(model_path))
    np.save(save_dir / "episode_rewards.npy", np.array(episode_rewards))
    print(f"Model saved to {model_path}")
    return agent, env, episode_rewards, save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol (e.g. AAPL)")
    parser.add_argument("--period", default="2y", help="Data period (e.g. 2y)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        symbol=args.symbol,
        period=args.period,
        episodes=args.episodes,
        seed=args.seed,
    )
