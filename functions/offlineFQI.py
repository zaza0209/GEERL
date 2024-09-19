# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:54:52 2024

@author: Lenovo
"""

import numpy as np
import torch
import sys
import torch.optim as optim
from torch import nn
#%%
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes=[64]):
        super(QNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_nodes[0]), nn.ReLU()]
        for i in range(len(hidden_nodes) - 1):
            layers.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes[-1], action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class OfflineFQI:
    def __init__(self, replay_buffer=None, 
                 batch_size=32, update_target_every=10, min_epsilon=0.01, 
                 gamma=0.9,
                 decay_rate=0.001, early_stopping_patience=10, action_space=[0,1],
                 hidden_nodes=[64], learning_rate=0.001,
                 convergence_threshold=1e-5,
                 early_stopping_threshold=0.01, verbose=1):
        
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.epsilon = 1.0
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.hidden_nodes = hidden_nodes
        self.learning_rate =learning_rate
        self.action_space=action_space
        self.gamma=gamma
        self.q_network=None
        self.verbose=verbose
        self.convergence_threshold=convergence_threshold
    
    def load_data(self, rollouts):
        # Flatten the rollouts
        self.states = torch.tensor([s for item in rollouts for s in item['states'].reshape(-1, item['states'].shape[-1])], dtype=torch.float32)
        self.actions = [a for item in rollouts for a in item['actions'].reshape(-1)]
        self.rewards = torch.tensor([r for item in rollouts for r in item['rewards'].reshape(-1)], dtype=torch.float32)
        self.next_states = torch.tensor([ns for item in rollouts for ns in item['next_states'].reshape(-1, item['next_states'].shape[-1])], dtype=torch.float32)
        self.state_dim = self.states.shape[1]
        self.action_dim = len(np.unique(self.actions))

    def init_network(self):
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_nodes)
        self.target_network = QNetwork(self.state_dim, self.action_dim, self.hidden_nodes)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def update_q_network_batch(self, states, actions, rewards, next_states):
        # Convert actions to indices
        action_indices = torch.tensor([self.action_space.index(a) for a in actions], dtype=torch.int64)

        # Compute Q-values for current states/actions
        q_values = self.q_network(states).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values for next states
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values

        # Calculate loss
        self.loss = nn.MSELoss()(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()
    

    def fit(self, num_epochs=200, batch_size=32):
        if not self.q_network:
            self.init_network()
        num_data = self.states.shape[0]
        
        for epoch in range(num_epochs):
            if self.verbose:
                print('epoch', epoch)
                sys.stdout.flush()
            permutation = torch.randperm(num_data) if batch_size != num_data else np.arange(num_data)

            for i in range(0, num_data, batch_size):
                indices = permutation[i:i+batch_size]
                batch_states, batch_actions, batch_next_states, batch_rewards = self.states[indices], [self.actions[i] for i in indices], self.next_states[indices], self.rewards[indices]
                
                if self.replay_buffer:
                    for i in range(len(batch_states)):
                        self.replay_buffer.push(batch_states[i], batch_actions[i], batch_rewards[i], batch_next_states[i])
                    # Update Q-network using replay buffer
                    loss = self.update_q_network_replay()
                else:
                    # Update Q-network directly with the batch
                    loss = self.update_q_network_batch(batch_states, batch_actions, batch_rewards, batch_next_states)
                if self.verbose:
                    print('loss', loss)
                    sys.stdout.flush()
                if self.is_converge():
                    print(f"converge after {epoch} epochs.")
                    sys.stdout.flush()
                    return
                # if loss < self.best_loss - self.early_stopping_threshold:
                #     self.best_loss = loss
                #     self.patience_counter = 0
                # else:
                #     self.patience_counter += 1
                #     if self.patience_counter >= self.early_stopping_patience:
                #         print(f"Early stopping triggered after {epoch} epochs.")
                #         return

            if epoch % self.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
    def is_converge(self):
        converged = all(
            torch.abs(p1 - p2).sum().item() <= self.convergence_threshold for p1, p2 in zip(self.q_network.parameters(), self.target_network.parameters())
        )
        return converged
    
    def predict_action(self, states):
        states= states.reshape((-1, self.state_dim))
        states_tensor = torch.FloatTensor(states)
        q_values = self.q_network(states_tensor)
        max_action_indices = torch.argmax(q_values, dim=1)
        actions = [self.action_space[idx.item()] for idx in max_action_indices]
        return np.array(actions)
    def update_q_network_replay(self):
        # Implement the logic for updating the Q-network using the replay buffer
        # This function should sample a batch from the replay buffer and update the network
        pass
