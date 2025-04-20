import numpy as np
import torch

class SimpleReplayBuffer:
    def __init__(self, capacity, input_shape, n_actions):
        self.capacity = capacity
        self.states = np.zeros((capacity, *input_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        idx = self.position % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[idxs], dtype=torch.float32),
            torch.tensor(self.actions[idxs], dtype=torch.long),
            torch.tensor(self.rewards[idxs], dtype=torch.float32),
            torch.tensor(self.next_states[idxs], dtype=torch.float32),
            torch.tensor(self.dones[idxs], dtype=torch.bool)
        )

    def __len__(self):
        return self.size
