import torch
import torch.nn as nn
from DoubleDQN import OfflineDoubleDQN, OfflineDoubleDQNWrapped


class OfflineCQL(OfflineDoubleDQN):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        gamma=0.9,
        batch_size=64,
        alpha=1.0,
    ):
        super().__init__(
            state_dim, action_dim, hidden_dims, learning_rate, gamma, batch_size
        )
        self.alpha = alpha

    def update(self, states, actions, rewards, next_states):

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values

        td_loss = nn.MSELoss()(current_q_values, target_q_values)
        cql_loss = self.alpha * (
            self.policy_net(states).logsumexp(dim=1).mean() - current_q_values.mean()
        )
        loss = td_loss + cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class OfflineCQLWrapped(OfflineDoubleDQNWrapped):
    def __init__(self, preprocessor, state_size, action_size, name, **kwargs):
        super().__init__(preprocessor, state_size, action_size, name)
        self.name = "cql_" + name
        self.agent = OfflineCQL(
            state_dim=state_size,
            action_dim=action_size,
            hidden_dims=[64, 64],
            learning_rate=0.001,
            gamma=0.9,
            batch_size=256,
            alpha=1.0,
        )
