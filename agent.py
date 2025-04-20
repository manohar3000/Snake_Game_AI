import torch
import numpy as np
from models import AgentNN
from replay_buffer import SimpleReplayBuffer

class Agent:
    def __init__(self,
                 input_dims,
                 num_actions,
                 lr=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 eps_decay=0.99,
                 eps_min=0.1,
                 replay_buffer_capacity=25000,
                 batch_size=64,
                 sync_network_rate=500):

        self.num_actions = num_actions
        self.learn_step_counter = 0
        self.losses = []
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Create online and target networks.
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

        self.replay_buffer = SimpleReplayBuffer(replay_buffer_capacity, input_dims, num_actions)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        device = self.online_network.device
        states = states.to(device)
        next_states = next_states.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        predicted_q = self.online_network(states)
        predicted_q = predicted_q[range(self.batch_size), actions]

        # Calculate the target Q value.
        target_q = self.target_network(next_states).max(dim=1)[0]
        target_q = rewards + self.gamma * target_q * (1 - dones.float())

        loss = self.loss_fn(predicted_q, target_q)
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)
