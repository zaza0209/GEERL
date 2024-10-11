import numpy as np
import torch


class TrainedD3rlpyPolicy:
    def __init__(self, agent) -> None:
        self.agent = agent

    def act(self, states):
        return self.agent.predict(states)  # d3rlpy


class RandomMatchPolicy:
    def __init__(self, p_match=0.5) -> None:
        self.p_match = p_match

    def act(self, states):
        num_team = len(states)
        competition = np.random.binomial(n=1, p=self.p_match, size=[num_team])

        matching_opponent = (np.ones([num_team]) * range(num_team)).astype(int)
        matching_pool = matching_opponent[competition == 1].flatten().tolist()
        # print(matching_pool)

        while len(matching_pool) > 1:
            i, j = np.random.choice(matching_pool, size=[2], replace=False)
            matching_pool.remove(i)
            matching_pool.remove(j)
            matching_opponent[i] = j
            matching_opponent[j] = i

        if len(matching_pool) == 1:
            matching_opponent[matching_pool[0]] = matching_pool[0]
            competition[matching_pool[0]] = 0

        return competition, matching_opponent


class RandomIndivPolicy:
    def __init__(self, p) -> None:
        self.p = p
        self.name = "random_{}".format(p)

    def act(self, states):
        num_indiv = states.shape[0]
        return np.random.binomial(n=1, p=self.p, size=[num_indiv])


class TrainedIndivPolicy:
    def __init__(
        self, agent, name="", epsilon=0.0, state_mean=None, state_std=None
    ) -> None:
        self.agent = agent
        self.name = "trained_{}".format(name)
        self.epsilon = epsilon
        self.state_mean = state_mean
        self.state_std = state_std

    def act(self, states):
        # print(states.shape)
        if self.state_mean is not None:
            states = (states - self.state_mean) / self.state_std
        # print(states.shape)
        try:
            states = states.reshape(states.shape[0], 1, states.shape[1])
            actions_greedy = self.agent(states).flatten()
        except:
            states = states.reshape(states.shape[0], states.shape[-1])
            actions_greedy = self.agent(states).flatten()
        if isinstance(actions_greedy, torch.Tensor):
            actions_greedy = actions_greedy.detach().numpy()

        actions_random = np.random.binomial(n=1, p=0.5, size=[states.shape[0]])
        I_random = np.random.binomial(n=1, p=self.epsilon, size=[states.shape[0]])
        return (1 - I_random) * actions_greedy + I_random * actions_random


# class TrainedMatchPolicy:
#     def __init__(self, Qmodel_type, Qmodel_file, state_space=None) -> None:
#         if Qmodel_type in ["rbf", "polynomial"]:
#             Qmodel = BasisFunctionApproximator(basis_type=Qmodel_type, state_dim=1)
#             raise NotImplementedError
#         elif Qmodel_type == "nn":
#             # Qmodel = NNApproximator(state_dim=1, hidden_dims=[32, 32])
#             Qmodel = NNApproximator(
#                 state_dim=1,
#                 hidden_dims=[64, 64],
#                 state_converting_basis="polynomial",
#                 degree=1,
#                 ctn2dis=None,
#                 state_space=None,
#                 lr=0.001,
#                 bias=0,
#             )
#             self.approximator = Qmodel.load(
#                 Qmodel_file,
#                 state_dim=1,
#                 hidden_dims=[64, 64],
#                 state_converting_basis="polynomial",
#                 degree=1,
#                 ctn2dis=None,
#                 state_space=None,
#                 lr=0.001,
#                 bias=0,
#             )
#             self.trainer = QTrainer(
#                 self.approximator,
#                 state_dim=1,
#                 is_learning=True,
#                 is_offline=True,
#                 reward_function=None,
#                 state_space=None,
#                 discretized_action_type="encountered",
#                 transition_function=None,
#             )
#         elif Qmodel_type == "tab":
#             Qmodel = TabularApproximator(state_space=state_space)
#             self.approximator = Qmodel.load(Qmodel_file)
#             self.trainer = QTrainer(
#                 self.approximator,
#                 state_dim=(
#                     len(state_space[0]) if type(state_space[0]) != np.float64 else 1
#                 ),
#                 is_learning=True,
#                 is_offline=True,
#                 reward_function=None,
#                 state_space=None,
#                 discretized_action_type="encountered",
#                 transition_function=None,
#             )
#         self.teams = None

#     def act(self, states, no_selfmatch=False):
#         if self.teams:
#             for i in range(len(self.teams)):
#                 self.teams[i].state = states[i]
#         else:
#             self.teams = [
#                 Team(
#                     team_id=i,
#                     reward_function=None,
#                     transition_function=None,
#                     s0=states[i],
#                 )
#                 for i in range(states.shape[0])
#             ]

#         matched_teams = [
#             x.team_id
#             for x in self.trainer.match_teams_optimally(
#                 self.teams, self.approximator.predict_q, no_selfmatch
#             )
#         ]  # [a0,a1,b0,b1,...]
#         competitions = np.ones([len(self.teams)], dtype=int)
#         matching_opponent = np.zeros_like(competitions, dtype=int)
#         for i in range(0, len(self.teams), 2):
#             # print(i, matched_teams[i])
#             matching_opponent[matched_teams[i]] = matched_teams[i + 1]
#             matching_opponent[matched_teams[i + 1]] = matched_teams[i]

#         return competitions, matching_opponent
