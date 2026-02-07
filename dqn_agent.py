# ---------------------------------------------------------------------------
# Custom DQN agent for the trading environment.
# ReplayBuffer: Stores (state, action, reward, next_state, done) for experience replay.
# QNetwork: Small MLP: state -> Q-values for each action.
# DQNAgent: DQN with experience replay and target network.
# ---------------------------------------------------------------------------
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# ReplayBuffer: Stores (state, action, reward, next_state, done) for experience replay.
# capacity: maximum number of transitions to store.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    # Push a new transition into the buffer
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    # Sample a batch of transitions from the buffer
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    # number of transitions in the buffer
    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------
# QNetwork: Small MLP: state -> Q-values for each action.
# state_dim: number of features in the state
# action_dim: number of actions
# hidden_dim: number of neurons in the hidden layer
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # define the Q-network layers: 2 hidden layers with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(), # keep the output non-negative
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    # forward pass: compute the Q-values for the state
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# DQNAgent: DQN with experience replay and target network.
# state_dim: number of features in the state
# action_dim: number of actions
# lr: learning rate
# gamma: discount factor
# epsilon_start: initial exploration rate
# epsilon_end: final exploration rate
# epsilon_decay: exploration rate decay rate
# buffer_size: maximum number of transitions to store
# batch_size: number of transitions to sample for training
# hidden_dim: number of neurons in the hidden layer
# device: device to use for training
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        hidden_dim: int = 64,
        device: Optional[str] = None,
    ):
        # configure settings for the agent
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # initialize the Q-network, target network, optimizer, and replay buffer
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
    
    # ---------------------------------------------------------------------------
    # Select an action based on the state and training mode
    # state: the current state
    # training: whether the agent is training
    # returns the action to take
    # ---------------------------------------------------------------------------
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        # with probability epsilon, take a random action
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # otherwise, take the action with the highest Q-value
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.q_net(x)
            return int(q.argmax(dim=1).item())

    # ---------------------------------------------------------------------------
    # Store a transition in the replay buffer
    # state: the current state
    # action: the action taken
    # reward: the reward received
    # next_state: the next state
    # done: whether the episode is done
    # ---------------------------------------------------------------------------
    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # push the transition into the replay buffer
        self.buffer.push(state, action, reward, next_state, done)

    # ---------------------------------------------------------------------------
    # Decay the exploration rate
    # makes the agent explore less over time by decreasing the exploration rate
    # ---------------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------------------------
    # Update the Q-network
    # performs one gradient step with target Q-values
    # returns the loss if a batch was trained, else None
    # ---------------------------------------------------------------------------
    def update(self) -> Optional[float]:
        # if the buffer is too small, return None
        if len(self.buffer) < self.batch_size:
            return None
        # sample a batch of transitions from the replay buffer
        batch = self.buffer.sample(self.batch_size)
        # convert the batch to tensors
        states = torch.from_numpy(np.array([b[0] for b in batch])).float().to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float).to(self.device)
        next_states = torch.from_numpy(np.array([b[3] for b in batch])).float().to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float).to(self.device)
        # Q-values for the current state
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Q-values and targets for the next state
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)
        # compute the loss
        loss = nn.functional.mse_loss(q_values, targets)
        # zero the gradients before backpropagation
        self.optimizer.zero_grad()
        # backpropagate the loss
        loss.backward()
        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        # update the Q-network weights
        self.optimizer.step()
        return loss.item()

    # ---------------------------------------------------------------------------
    # Sync the target network
    # copies the Q-network weights to the target network
    # ---------------------------------------------------------------------------
    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ---------------------------------------------------------------------------
    # Save the model
    # saves the Q-network and target network weights to a file
    # this helps us load the model later for evaluation
    # path: path to save the model
    # ---------------------------------------------------------------------------
    def save(self, path: str):
        torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, path)

    # ---------------------------------------------------------------------------
    # Load the model
    # loads the Q-network and target network weights from a file
    # use for loading saved models for evaluation
    # path: path to load the model
    # ---------------------------------------------------------------------------
    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data["q_net"])
        self.target_net.load_state_dict(data["target_net"])
