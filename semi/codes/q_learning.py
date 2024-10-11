import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import logging
import copy
from my_utils import glogger
import collections
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# mylogger = logging.getLogger("testSA2-fqe")


# dataloader
class RLDataset(Dataset):
    def __init__(self, S, A, R):
        self.s_dim = S.shape[2]
        self.N, self.T = A.shape
        self.S = S[:, : self.T].reshape([self.N * self.T, self.s_dim])
        self.Sp = S[:, 1 : (self.T + 1)].reshape([self.N * self.T, self.s_dim])
        # print(self.S[:3])
        # print(self.Sp[:3])
        self.A = A.reshape([self.N * self.T, 1])
        self.R = R.reshape([self.N * self.T, 1])

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        S = torch.tensor(self.S[idx], dtype=torch.float32)
        Sp = torch.tensor(self.Sp[idx], dtype=torch.float32)
        A = torch.tensor(self.A[idx], dtype=torch.int64)
        R = torch.tensor(self.R[idx], dtype=torch.float32)
        return S, A, R, Sp


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, states, actions, rewards, next_states) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        try:
            experiences = zip(states, actions, rewards, next_states)
            self.buffer.extend(experiences)
        except:
            self.buffer.append((states, actions, rewards, next_states))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return (
            np.array(states).reshape(batch_size, -1),
            np.array(actions).reshape(batch_size, -1),
            np.array(rewards, dtype=np.float32).reshape(batch_size, -1),
            np.array(next_states).reshape(batch_size, -1),
        )


# implementation of offline DQN
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None):
        super(QNet, self).__init__()
        if hidden_dims == None:
            hidden_dims = [32, 64, 32]
        nn_dims = [in_dim] + hidden_dims + [out_dim]
        modules = []
        for i in range(len(nn_dims) - 1):
            if i == len(nn_dims) - 2:
                modules.append(nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1])))
            else:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1]), nn.ReLU())
                )
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class dqn_model(nn.Module):
    def __init__(self, model_parameters) -> None:
        super().__init__()
        self.Q = QNet(
            in_dim=model_parameters["in_dim"],
            out_dim=model_parameters["out_dim"],
            hidden_dims=model_parameters["hidden_dims"],
        )
        self.target_Q = QNet(
            in_dim=model_parameters["in_dim"],
            out_dim=model_parameters["out_dim"],
            hidden_dims=model_parameters["hidden_dims"],
        )
        self.update_target_net()
        # self.target_Q.eval()
        self.gamma = model_parameters["gamma"]

    def loss_function(self, s, a, r, sp):
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.int64)
            r = torch.tensor(r, dtype=torch.float32)
            sp = torch.tensor(sp, dtype=torch.float32)
        # print(self.Q(s))
        # print(a)
        # print(self.Q(s).gather(1, a))
        state_action_values = self.Q(s).gather(1, a).flatten()
        # vanilla dqn
        # next_state_values = self.target_Q(sp).max(1)[0]
        # target_values = r.flatten() + self.gamma * next_state_values.detach()
        # print(state_action_values[:2])
        # print(next_state_values[:2])
        # print(target_values[:2])
        # exit()

        # double dqn
        _, a_selected = self.Q(sp).detach().max(1)
        next_state_values = (
            self.target_Q(sp).gather(1, a_selected.reshape([-1, 1])).detach().flatten()
        )
        target_values = r.flatten() + self.gamma * next_state_values
        td_error = nn.MSELoss()(target_values, state_action_values)
        return td_error

    def update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def act(self, s):
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float32)
        else:
            s = s.to(torch.float32)
        with torch.no_grad():
            self.Q.eval()
            # print(self.Q(s)[:2])
            action = torch.argmax(self.Q(s), dim=1)
        return action.cpu().detach().numpy()

    def __call__(self, s):
        return self.act(s)

    def evaluate_Q(self):
        s = np.array([[15], [20.0], [25]])
        s = torch.tensor(s, dtype=torch.float32)
        return self.Q(s).detach().numpy()


def offline_dqn_learning(data, model_parameters, train_parameters, ts_writer=None):
    # torch.set_num_threads(1)
    ## extract parameters
    S, A, R = data["S"], data["A"], data["R"]
    ## normalizatin
    # S_n = state_normalizer.normalize(S)
    # print(np.mean(S_n, axis=(0, 1)))
    # R_n = reward_noemalizer.normalize(R)
    # print(np.mean(R_n, axis=(0, 1)))

    ## set parameter
    in_dim = model_parameters["in_dim"]
    out_dim = model_parameters["out_dim"]
    hidden_dims = model_parameters["hidden_dims"]
    batch_size, epochs, target_update_freq, learning_rate, grad_norm = (
        train_parameters["batch_size"],
        train_parameters["epochs"],
        train_parameters["target_update_freq"],
        train_parameters["learning_rate"],
        train_parameters["grad_norm"],
    )
    ## make dataloader
    dataloader_train = torch.utils.data.DataLoader(
        RLDataset(S, A, R),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    ## pre training
    device = torch.device("cpu")
    model = dqn_model(model_parameters=model_parameters).to(device)
    optimizer = optim.Adam(model.Q.parameters(), lr=learning_rate)  # 0.001
    # optimizer = optim.RMSprop(model.Q.parameters(),
    #                           lr=learning_rate,
    #                           alpha=0.95,
    #                           eps=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000000, gamma=0.5)

    model.Q.train()
    model.target_Q.eval()
    iters = 1
    num_updates = 0
    for ep in range(epochs):
        # running_loss = []
        for batch_idx, batch in enumerate(dataloader_train):
            s, a, r, sp = batch[0], batch[1], batch[2], batch[3]
            optimizer.zero_grad()
            loss = model.loss_function(s, a, r, sp)  # problem is here
            loss.backward()
            nn.utils.clip_grad_norm_(model.Q.parameters(), grad_norm)
            # nn.utils.clip_grad_value_(model.Q.parameters(), 1)
            optimizer.step()
            scheduler.step()
            if iters % target_update_freq == 0:
                model.update_target_net()
                num_updates += 1
                if ts_writer:
                    ts_writer.add_scalar(
                        "DQN update_{}".format(S.shape[1]), num_updates, iters
                    )
                # soft_update(model.Q, model.target_Q, tau=1)
            if iters % 100 == 0:
                glogger.debug(
                    "Iter: {}, DQN training loss: {}".format(iters, loss.item())
                )
                if ts_writer:
                    ts_writer.add_scalar(
                        "DQN training loss_{}".format(S.shape[1]), loss.item(), iters
                    )
                    grad_norm_now = nn.utils.clip_grad_norm_(
                        model.Q.parameters(), 10000000
                    )
                    ts_writer.add_scalar(
                        "DQN grad norm_{}".format(S.shape[1]), grad_norm_now, iters
                    )
            iters += 1
    return model.to("cpu")


class dqn_model_linear:
    def __init__(self, model_parameters) -> None:
        super().__init__()
        self.state_dim = model_parameters["in_dim"]
        self.n_actions = model_parameters["out_dim"]
        self.Q = [LinearRegression(fit_intercept=True) for _ in range(self.n_actions)]
        # random init Q
        for i in range(self.n_actions):
            self.Q[i].fit(np.random.rand(100, self.state_dim), np.random.rand(100, 1))
        self.update_target_net()
        self.gamma = model_parameters["gamma"]

    def update_Q(self, s, a, r, sp):
        # state_action_values = (
        #     np.array([self.Q[i].predict(s) for i in range(self.n_actions)])
        #     .reshape(-1, self.n_actions)
        #     .max(1)
        # )
        # vanilla dqn
        next_state_values = (
            np.array([self.target_Q[i].predict(sp) for i in range(self.n_actions)])
            .reshape(-1, self.n_actions)
            .max(1)
        )
        target_values = r.flatten() + self.gamma * next_state_values

        # double dqn
        # _, a_selected = self.Q(sp).detach().max(1)
        # next_state_values = (
        #     self.target_Q(sp).gather(1, a_selected.reshape([-1, 1])).detach().flatten()
        # )
        # target_values = r.flatten() + self.gamma * next_state_values

        for i in range(self.n_actions):
            idx = a.flatten() == i
            self.Q[i].fit(s[idx], target_values[idx].reshape(-1, 1))

    def update_target_net(self):
        self.target_Q = copy.deepcopy(self.Q)
        print(self.target_Q[0].coef_)

    def act(self, s):
        s = np.array(s).reshape([-1, self.state_dim])
        state_action_values = np.array(
            [self.Q[i].predict(s) for i in range(self.n_actions)]
        ).reshape(-1, self.n_actions)
        return np.argmax(state_action_values, axis=1)

    def __call__(self, s):
        return self.act(s)


def calculate_expected_discounted_reward_MC(reward, gamma):
    N, T = reward.shape
    s = np.zeros([N])
    for t in range(T - 1, -1, -1):
        s = gamma * s + reward[:, t]
    return np.mean(s)


if __name__ == "__main__":
    pass
