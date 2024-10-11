import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import copy

from base_models import LinearRegressor, NeuralNet
from my_utils import glogger

glogger.setLevel("INFO")


class OfflineFQI:
    def __init__(self, model_type, action_size) -> None:
        self.model_type = model_type
        self.action_size = action_size

        # sanity check
        self._sanity_check()

    def _sanity_check(self):
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def fit(self, states, actions, rewards, max_iter):
        torch.set_num_threads(1)
        # convenience variables
        N, T = actions.shape
        sdim = states.shape[-1]

        # reshape data
        next_states = states[:, 1:, :].reshape(-1, sdim)
        states = states[:, :-1, :].reshape(-1, sdim)
        actions = actions[:, :].flatten()
        rewards = rewards[:, :].flatten()

        # init model
        if self.model_type == "lm":
            current_model = [None for i in range(self.action_size)]
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    tmp = np.zeros([N * T, self.action_size])
                    for a in range(self.action_size):
                        tmp[:, a] = current_model[a].predict(next_states).flatten()
                    Y = (rewards + 0.9 * np.max(tmp, axis=1)).reshape(-1, 1)

                # generate input
                new_model = [
                    LinearRegressor(
                        featurize_method="polynomial",
                        degree=2,
                        interaction_only=False,
                        is_standarized=False,
                    )
                    for i in range(self.action_size)
                ]
                for a in range(self.action_size):
                    idx = actions == a
                    X_a = states[idx]
                    Y_a = Y[idx]
                    new_model[a].train(X_a, Y_a)
                if i % 10 == 0:
                    glogger.info(
                        "{}, fqi_lm mse:{}, mean_target:{}".format(
                            i, np.mean([m.mse for m in new_model]), np.mean(Y)
                        )
                    )
                current_model = copy.deepcopy(new_model)
        elif self.model_type == "nn":
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=[64]
            )

            # training loop
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    current_model.eval()
                    tmp = (
                        current_model.forward(
                            torch.tensor(next_states, dtype=torch.float32)
                        )
                        .detach()
                        .numpy()
                    )
                    Y = (rewards + 0.9 * np.max(tmp, axis=1)).reshape(-1, 1)
                # generate input
                X = states

                # train model
                new_model = NeuralNet(
                    in_dim=sdim, out_dim=self.action_size, hidden_dims=[64]
                )
                # new_model = copy.deepcopy(current_model)
                Y = torch.tensor(Y, dtype=torch.float32)

                optimizer = torch.optim.Adam(new_model.parameters(), lr=0.05)
                for _ in range(500):
                    new_model.train()
                    Y_pred = new_model.forward(
                        torch.tensor(X, dtype=torch.float32)
                    ).gather(1, torch.tensor(actions.reshape(-1, 1), dtype=torch.int64))
                    optimizer.zero_grad()
                    loss = torch.nn.MSELoss()(Y, Y_pred)
                    loss.backward()
                    optimizer.step()
                if i % 10 == 0:
                    glogger.info(
                        "{}, fqi_nn mse:{}, mean_target:{}".format(
                            i, loss.item(), np.mean(Y.detach().numpy())
                        )
                    )
                current_model = copy.deepcopy(new_model)

        self.model = copy.deepcopy(new_model)

    def act(self, states):
        if self.model_type == "lm":
            tmp = np.zeros([states.shape[0], self.action_size])
            for a in range(self.action_size):
                tmp[:, a] = (
                    self.model[a].predict(states.reshape(states.shape[0], -1)).flatten()
                )
        if self.model_type == "nn":
            self.model.eval()
            states = states.reshape(states.shape[0], -1)
            tmp = (
                self.model.forward(torch.tensor(states, dtype=torch.float32))
                .detach()
                .numpy()
            )

        return np.argmax(tmp, axis=1)


class FQE:
    def __init__(self, model_type, action_size, policy) -> None:
        self.model_type = model_type
        self.action_size = action_size
        self.policy = policy

        # sanity check
        self._sanity_check()

    def _sanity_check(self):
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def fit(self, states, actions, rewards, max_iter):
        torch.set_num_threads(1)
        # convenience variables
        N, T = actions.shape
        sdim = states.shape[-1]

        # reshape data
        next_states = states[:, 1:, :].reshape(-1, sdim)
        states = states[:, :-1, :].reshape(-1, sdim)
        actions = actions[:, :].flatten()
        rewards = rewards[:, :].flatten()

        # init model
        if self.model_type == "lm":
            current_model = [None for _ in range(self.action_size)]
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    tmp = np.zeros([N * T, self.action_size])
                    for a in range(self.action_size):
                        tmp[:, a] = current_model[a].predict(next_states).flatten()

                    selected_actions = (
                        self.policy.act(next_states).flatten().reshape(-1, 1)
                    )
                    print(selected_actions.shape, tmp.shape, rewards.shape)
                    Y = (
                        rewards.reshape(-1, 1)
                        + 0.9 * np.take_along_axis(tmp, selected_actions, axis=1)
                    ).reshape(-1, 1)

                # generate input
                new_model = [
                    LinearRegressor(
                        featurize_method="polynomial",
                        degree=2,
                        interaction_only=False,
                        is_standarized=False,
                    )
                    for i in range(self.action_size)
                ]
                for a in range(self.action_size):
                    idx = actions == a
                    X_a = states[idx]
                    Y_a = Y[idx]
                    new_model[a].train(X_a, Y_a)
                glogger.info(
                    "{}, fqe_lm mse:{}, mean_target:{}".format(
                        i, np.mean([m.mse for m in new_model]), np.mean(Y)
                    )
                )
                current_model = copy.deepcopy(new_model)

        elif self.model_type == "nn":
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=[64]
            )

            # training loop
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    current_model.eval()
                    tmp = (
                        current_model.forward(
                            torch.tensor(next_states, dtype=torch.float32)
                        )
                        .detach()
                        .numpy()
                    )
                    selected_actions = self.policy.act(next_states).reshape(-1, 1)

                    Y = (
                        rewards.reshape(-1, 1)
                        + 0.9 * np.take_along_axis(tmp, selected_actions, axis=1)
                    ).reshape(-1, 1)

                # generate input
                X = states
                Y = torch.tensor(Y, dtype=torch.float32)

                # train model
                new_model = NeuralNet(
                    in_dim=sdim,
                    out_dim=self.action_size,
                    hidden_dims=[64],
                )
                optimizer = torch.optim.Adam(new_model.parameters(), lr=0.05)
                for _ in range(500):
                    new_model.train()
                    Y_pred = new_model.forward(
                        torch.tensor(X, dtype=torch.float32)
                    ).gather(1, torch.tensor(actions.reshape(-1, 1), dtype=torch.int64))
                    optimizer.zero_grad()
                    loss = torch.nn.MSELoss()(Y, Y_pred)
                    loss.backward()
                    optimizer.step()

                glogger.info(
                    "{}, fqe_nn mse:{}, mean_target:{}".format(
                        i, loss.item(), np.mean(Y.detach().numpy())
                    )
                )
                current_model = copy.deepcopy(new_model)

        self.model = copy.deepcopy(new_model)

    def evaluate(self, states):
        actions_taken = self.policy.act(states)
        if self.model_type == "lm":
            tmp = np.zeros([states.shape[0], self.action_size])
            for a in range(self.action_size):
                tmp[:, a] = (
                    self.model[a].predict(states.reshape(states.shape[0], -1)).flatten()
                )
            return np.take_along_axis(
                tmp, actions_taken.reshape(-1, 1), axis=1
            ).flatten()
        elif self.model_type == "nn":
            self.model.eval()
            X = torch.tensor(states, dtype=torch.float32)
            actions_taken = torch.tensor(
                actions_taken.reshape(-1, 1), dtype=torch.int64
            )
            Y = self.model.forward(X).gather(
                1, torch.tensor(actions_taken, dtype=torch.int64)
            )
            return Y.detach().numpy().flatten()


class DQN:
    def __init__(
        self,
        state_dim,
        action_size,
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=100,
    ):
        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Q-Network and target network
        self.q_network = self._build_network(state_dim, action_size, hidden_dims)
        self.target_network = self._build_network(state_dim, action_size, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_network(self, input_dim, output_dim, hidden_dims):
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update_Q(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
