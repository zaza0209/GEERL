import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from utils.base_models import NeuralNet
from utils.utils import glogger


class OfflineDoubleDQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        gamma=0.9,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = NeuralNet(
            in_dim=state_dim, out_dim=action_dim, hidden_dims=hidden_dims
        )
        self.target_net = NeuralNet(
            in_dim=state_dim, out_dim=action_dim, hidden_dims=hidden_dims
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = state.reshape(-1, self.state_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.policy_net(state)
            return torch.argmax(q_values, 1)

    def update(self, states, actions, rewards, next_states):

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q_values = self.policy_net(states).gather(1, actions)

        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards + self.gamma * next_q_values
        # print(current_q_values.shape, target_q_values.shape)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(
        self,
        states,
        actions,
        rewards,
        next_states,
        num_iterations,
        update_frequency=100,
    ):
        torch.set_num_threads(1)
        states = states.reshape(-1, self.state_dim)
        next_states = next_states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        for iteration in range(num_iterations):
            batch_idxs = random.sample(range(states.shape[0]), self.batch_size)
            loss = self.update(
                states[batch_idxs],
                actions[batch_idxs],
                rewards[batch_idxs],
                next_states[batch_idxs],
            )

            if iteration % update_frequency == 0:
                self.update_target_network()

            if iteration % 100 == 0:
                glogger.info(
                    f"Iteration {iteration}/{num_iterations}, Loss: {loss:.4f}"
                )
        return loss

    def act(self, state):
        return self.select_action(state)


class OfflineDoubleDQNWrapped:

    def __init__(self, preprocessor, state_size, action_size, name, **kwargs):
        self.preprocessor = preprocessor
        self.state_size = state_size
        self.action_size = action_size
        self.name = "ddqn_" + name

        self.agent = OfflineDoubleDQN(
            action_dim=action_size,
            state_dim=state_size,
            hidden_dims=[64, 64],
            learning_rate=0.001,
            gamma=0.9,
            batch_size=256,
        )

    def train(self, xs, z, actions, rewards, max_iter):
        N, T = xs.shape[:2]

        states_p = np.zeros((N, T, self.state_size))
        rewards_p = np.zeros((N, T - 1))
        if hasattr(self.preprocessor, "reset_buffer"):
            self.preprocessor.reset_buffer(N)

        if type(self.preprocessor).__name__ in [
            "ModelFreePreprocess",
            "ModelFreePreprocessOracle",
        ]:
            for t in range(T):
                xt = xs[:, t]
                if t == 0:
                    states_p[:, t] = self.preprocessor.preprocess(
                        xt=xt, xtm1=None, z=z, atm1=None, rtm1=None
                    )
                else:
                    xtm1 = xs[:, t - 1]
                    atm1 = actions[:, t - 1]
                    rtm1 = rewards[:, t - 1]
                    states_p[:, t], rewards_p[:, t - 1] = self.preprocessor.preprocess(
                        xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=rtm1
                    )

        else:
            for t in range(T):
                xt = xs[:, t]
                if t == 0:
                    states_p[:, t] = self.preprocessor.preprocess(
                        xt=xt, xtm1=None, z=z, atm1=None
                    )
                else:
                    xtm1 = xs[:, t - 1]
                    atm1 = actions[:, t - 1]
                    states_p[:, t] = self.preprocessor.preprocess(
                        xt=xt, xtm1=xtm1, z=z, atm1=atm1
                    )
            rewards_p = copy.deepcopy(rewards)

        next_states_p = states_p[:, 1:].copy()
        states_p = states_p[:, :-1].copy()

        self.agent.train(
            states_p, actions, rewards_p, next_states_p, max_iter, update_frequency=100
        )

    def act(self, xt, z, xtm1, atm1, uat=None, is_return_prob=False, **kwargs):
        if type(self.preprocessor).__name__ in [
            "ModelFreePreprocess",
            "ModelFreePreprocessOracle",
        ]:
            states = self.preprocessor.preprocess(
                xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
            )
        else:
            states = self.preprocessor.preprocess(xt=xt, xtm1=xtm1, z=z, atm1=atm1)
        probs = np.zeros((xt.shape[0], 2))
        actions = self.agent.act(states)
        if is_return_prob:
            return actions, probs
        else:
            return actions
