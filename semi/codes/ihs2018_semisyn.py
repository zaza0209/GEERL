import pandas as pd
import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.preprocessing import PolynomialFeatures
import copy
import pickle
import os

from agent_wrappers import RandomIndivPolicy, RandomMatchPolicy
from base_models import NeuralNetRegressor, LinearRegressor


IS_DATA_AVAILABLE = False


class IHS2018Data:
    def __init__(self) -> None:
        filename = "../data/2018_imputed_data_daily_hr_extended.csv"
        if IS_DATA_AVAILABLE:
            self.raw_data = pd.read_csv(filename)  # unscaled data
        else:
            self.raw_data = None

        # only use 10 weeks of data
        self.raw_data = self.raw_data[self.raw_data["week"] < 2]

        self.dict_transform_fns = {
            "step": np.cbrt,
            "step_count": np.cbrt,
            "sleep": np.sqrt,
            "sleep_count": np.sqrt,
            "mood": lambda x: x,
            "dow": lambda x: x,
            "week": lambda x: x,
        }
        self.dict_detransform_fns = {
            "step": lambda x: np.power(x, 3),
            "step_count": lambda x: np.power(x, 3),
            "sleep": lambda x: np.power(x, 2),
            "sleep_count": lambda x: np.power(x, 2),
            "mood": lambda x: x,
            "dow": lambda x: x,
            "week": lambda x: x,
        }

    def rename_vars(self, state_vars, reward_var):
        dict_rename = {
            "step": "step_count",
            "sleep": "sleep_count",
            "mood": "mood",
            "dow": "dow",
            "week": "week",
        }

        next_state_vars = [
            dict_rename[s] for s in state_vars if s != "week" and s != "dow"
        ]

        for i, s in enumerate(state_vars):
            if s != "week" and s != "dow":
                state_vars[i] = dict_rename[s] + "_prev"
            else:
                state_vars[i] = dict_rename[s]
        reward_var = dict_rename[reward_var]

        return state_vars, reward_var, next_state_vars

    def load_data(self, state_vars, reward_var, is_transformed=False):
        # prepare transform functions
        if is_transformed:
            self.state_transform_fns = [self.dict_transform_fns[s] for s in state_vars]
            self.state_detransform_fns = [
                self.dict_detransform_fns[s] for s in state_vars
            ]
            self.reward_transform_fn = self.dict_transform_fns[reward_var]
            self.reward_detransform_fn = self.dict_detransform_fns[reward_var]

        # rename variables
        state_vars, reward_var, next_state_vars = self.rename_vars(
            state_vars, reward_var
        )

        # copy data
        data = self.raw_data.copy()
        data["date"] = pd.to_datetime(data["date_day"], format="%Y-%m-%d")
        data["dow"] = data["date"].dt.dayofweek + 1

        # binarize action and competition
        data["action"] = (data["NOTIFICATION_TYPE"] != "no_message").astype(int)

        # get data
        states = data.loc[:, state_vars]
        next_states = data.loc[:, next_state_vars]
        rewards = data.loc[:, reward_var]
        actions = data.loc[:, ["action"]]

        if is_transformed:
            for i, s in enumerate(state_vars):
                states[s] = self.state_transform_fns[i](states[s])
            for i, s in enumerate(next_state_vars):
                next_states[s] = self.state_transform_fns[i](next_states[s])
            rewards = self.reward_transform_fn(rewards)

        team_states_dim = len([s for s in state_vars if s != "week" and s != "dow"])
        opponent_states = np.zeros([states.shape[0], team_states_dim])
        team_states = np.zeros([states.shape[0], team_states_dim])
        competitions = np.zeros([states.shape[0]])

        return {
            "states": states.to_numpy().astype(np.float32),
            "actions": actions.to_numpy().astype(np.float32),
            "next_states": next_states.to_numpy().astype(np.float32),
            "rewards": rewards.to_numpy().flatten().astype(np.float32),
            "opponent_states": opponent_states.astype(np.float32),
            "team_states": team_states.astype(np.float32),
            "competitions": competitions.astype(np.float32),
        }


class IHS2018SemiSynData:
    def __init__(
        self,
        state_vars=["step"],
        reward_var="step",
        is_include_team_effect=False,
        is_transformed=False,
        is_clip=False,
        is_standarized=True,
        degree=1,
        cor_boost_coef=1,
    ) -> None:
        torch.set_num_threads(1)
        # copy arguments
        self.state_vars = state_vars
        self.reward_var = reward_var
        self.is_include_team_effect = is_include_team_effect
        self.is_transformed = is_transformed
        self.is_clip = is_clip
        self.is_standarized = is_standarized
        self.degree = degree
        self.cor_boost_coef = cor_boost_coef
        self.is_include_week = True if "week" in state_vars else False
        self.is_include_dow = True if "dow" in state_vars else False

        # sanity check
        self.sanity_check()

        # check if reward is in state_vars
        self.reward_idx = (
            self.state_vars.index(self.reward_var)
            if self.reward_var in self.state_vars
            else None
        )

        # load data
        if IS_DATA_AVAILABLE:
            self.ihs2018_data = IHS2018Data()
            self.data = self.ihs2018_data.load_data(
                state_vars=copy.deepcopy(state_vars),
                reward_var=copy.deepcopy(reward_var),
                is_transformed=self.is_transformed,
            )
            self.next_states_max = np.percentile(
                self.data["next_states"], axis=0, q=97.5
            )
            self.next_states_min = np.percentile(
                self.data["next_states"], axis=0, q=2.5
            )
            self.rewards_max = np.percentile(self.data["rewards"], axis=0, q=97.5)
            self.rewards_min = np.percentile(self.data["rewards"], axis=0, q=2.5)
            self.fit_transition_models()
        else:
            # raise NotImplementedError
            self.data = None
            d_min_max = {
                "step": [7.80, 25.98],
                "sleep": [11.61, 27.17],
                "mood": [4, 10],
            }
            self.next_states_max = np.array(
                [d_min_max[s][1] for s in self.state_vars if s != "week" and s != "dow"]
            )
            self.next_states_min = np.array(
                [d_min_max[s][0] for s in self.state_vars if s != "week" and s != "dow"]
            )
            self.rewards_max = np.array([d_min_max[self.reward_var][1]])
            self.rewards_min = np.array([d_min_max[self.reward_var][0]])

        # flag for trained
        self.is_model_trained = False

    def sanity_check(self):
        assert set(self.state_vars).issubset(
            set(["step", "sleep", "mood", "week", "dow"])
        ), "Invalid state variables"
        assert self.reward_var in ["step", "sleep", "mood"], "Invalid reward variable"
        assert (
            not self.is_transformed
        ), "Data should  not be transformed, since the raw data is already transformed"

    def featurize(self, states):
        new_states = PolynomialFeatures(
            degree=self.degree, include_bias=False, interaction_only=False
        ).fit_transform(states)
        return new_states

    def load_trained_model(self, dirname):
        self.transition_models = []
        for a in range(2):
            with open(
                os.path.join(
                    dirname,
                    "18_transition_model_d{}_a{}__{}__{}.pkl".format(
                        self.degree, a, "_".join(self.state_vars), self.reward_var
                    ),
                ),
                "rb",
            ) as f:
                self.transition_models.append(pickle.load(f))

    def save_trained_model(self, dirname):
        if self.is_model_trained:
            for a in range(2):
                with open(
                    os.path.join(
                        dirname,
                        "18_transition_model_d{}_a{}__{}__{}.pkl".format(
                            self.degree, a, "_".join(self.state_vars), self.reward_var
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.transition_models[a], f)
        else:
            raise ValueError("Model is not trained yet")

    def sample_init_state_competition_matching_state(self, n_team, n_indiv, p_match):

        # option1: bootstraping from original data
        init_indiv_states = []
        for i in range(n_team):
            if IS_DATA_AVAILABLE:
                states = self.data["states"]  # state
                idx = np.random.choice(states.shape[0], size=n_indiv, replace=True)
                tmp = states[idx, :].copy()
                # set dow and week
                if self.is_include_week:
                    week_idx = self.state_vars.index("week")
                    tmp[:, week_idx] = 0
                if self.is_include_dow:
                    dow_idx = self.state_vars.index("dow")
                    tmp[:, dow_idx] = 1
            else:
                dict_states_mean_std = {
                    "step": [14.4, 4.2],
                    "sleep": [7.5, 2.1],
                    "mood": [1.6, 0.4],
                }
                tmp = np.zeros([n_indiv, len(self.state_vars)])
                for j, s in enumerate(self.state_vars):
                    tmp[:, j] = np.random.normal(
                        dict_states_mean_std[s][0], dict_states_mean_std[s][1], n_indiv
                    )

            init_indiv_states.append(tmp)  # [n_indiv, state_dim]

        # no previous week information, besides there are warmup weeks
        new_competition, matching_opponent = RandomMatchPolicy(p_match=p_match).act(
            [0] * n_team
        )

        # calculaet team states
        init_team_states = []
        init_opponent_states = []

        # team states (remove week and dow)
        team_state_vars_idx = [
            i for i, e in enumerate(self.state_vars) if e not in ("week", "dow")
        ]
        for i in range(n_team):
            mat_indiv = np.array(init_indiv_states[i][team_state_vars_idx])
            mat_team = np.zeros([len(team_state_vars_idx)])
            for j in range(len(team_state_vars_idx)):
                mat_team[j] = np.mean(mat_indiv[:, j])
            init_team_states.append(mat_team)
        for i in range(n_team):
            init_opponent_states.append(init_team_states[matching_opponent[i]])

        return (
            init_indiv_states,
            init_team_states,
            init_opponent_states,
            new_competition,
        )

    def generate_state_random_error(self, n_team, n_indiv, T):
        # S_{i,j,t} = \E\left[S_{i,j,t} \mid S_{i,j,t-1}, A_{i,j,t-1},D_{t-1}\right] + \epsilon_{i,j,t} + \varepsilon_{i,t}

        dict_sigmasq = {
            "step": [14.4, 4.2],
            "sleep": [7.5, 2.1],
            "mood": [1.6, 0.4],
        }
        for key, value in dict_sigmasq.items():
            original_correlation = dict_sigmasq[key][1] / (
                dict_sigmasq[key][0] + dict_sigmasq[key][1]
            )
            new_correlation = min(original_correlation * self.cor_boost_coef, 0.95)
            variance_all = dict_sigmasq[key][0] + dict_sigmasq[key][1]
            dict_sigmasq[key][1] = variance_all * new_correlation
            dict_sigmasq[key][0] = variance_all * (1 - new_correlation)

        # remove week and dow (no error for week and dow)
        state_vars = [s for s in self.state_vars if s != "week" and s != "dow"]

        transformed_errors = []
        for k in range(n_team):
            tmp = np.zeros([n_indiv, T, len(state_vars)])
            for i in range(len(state_vars)):
                sigma1sq, sigma2sq = dict_sigmasq[state_vars[i]]

                # \varepsilon_{i,t}
                time_errors = multivariate_normal.rvs(
                    mean=0,
                    cov=sigma2sq,
                    size=[T],
                ).reshape(1, T, 1)
                time_errors = np.repeat(time_errors, n_indiv, axis=0)

                # \epsilon_{i,j,t}
                indiv_errors = multivariate_normal.rvs(
                    mean=np.zeros([T]),
                    cov=np.eye(T) * sigma1sq,
                    size=[n_indiv],
                ).reshape(n_indiv, T, 1)
                tmp[:, :, [i]] = time_errors + indiv_errors
                # tmp[:, :, [i]] = time_errors
            transformed_errors.append(tmp)
        return transformed_errors

    def generate_reward_random_error(
        self, n_team, n_indiv, T, phi, cor_error_type="it", eval_mode=False
    ):

        dict_sigmasq = {
            "step": [14.4, 4.2],
            "sleep": [7.5, 2.1],
            "mood": [1.6, 0.4],
        }
        for key, value in dict_sigmasq.items():
            original_correlation = dict_sigmasq[key][1] / (
                dict_sigmasq[key][0] + dict_sigmasq[key][1]
            )
            new_correlation = min(original_correlation * self.cor_boost_coef, 0.95)
            variance_all = dict_sigmasq[key][0] + dict_sigmasq[key][1]
            dict_sigmasq[key][1] = variance_all * new_correlation
            dict_sigmasq[key][0] = variance_all * (1 - new_correlation)
            # dict_sigmasq[key][0] = variance_constant / (
            #     1 + phi**2 + new_correlation / (1 - new_correlation)
            # )
            # dict_sigmasq[key][1] = (
            #     new_correlation / (1 - new_correlation) * dict_sigmasq[key][0]
            # )
            # dict_sigmasq[key][1] = 0.0001

        assert cor_error_type in ["i", "it"], "Invalid error type for reward"
        transformed_errors = []
        for k in range(n_team):
            sigma1sq, sigma2sq = dict_sigmasq[self.reward_var]

            # alpha_{it}
            if cor_error_type == "it":
                team_errors = multivariate_normal.rvs(
                    mean=0,
                    cov=sigma2sq,
                    size=[T],
                ).reshape(1, T)
                team_errors = np.repeat(team_errors, n_indiv, axis=0)
            # alpha_{i}
            elif cor_error_type == "i":
                team_errors = multivariate_normal.rvs(
                    mean=0,
                    cov=sigma2sq,
                    size=[1],
                ).reshape(1, 1)
                raise AttributeError
                team_errors = np.repeat(
                    np.repeat(team_errors, n_indiv, axis=0), T, axis=1
                )

            # \xi_{i,j,t}
            indiv_errors = np.zeros([n_indiv, T])
            indiv_errors[:, 0] = multivariate_normal.rvs(
                mean=0,
                cov=sigma1sq,
                size=[n_indiv],
            )
            for t in range(1, T):
                indiv_errors[:, t] = phi * indiv_errors[
                    :, t - 1
                ] + multivariate_normal.rvs(mean=0, cov=sigma1sq, size=[n_indiv])
            if eval_mode:
                transformed_errors.append(
                    (team_errors + indiv_errors.reshape(n_indiv, T)) * 0.001
                )
            else:
                transformed_errors.append(
                    team_errors + indiv_errors.reshape(n_indiv, T)
                )
        return transformed_errors

    def fit_transition_models(self):
        states = self.data["states"]  # state, np.array([N*T,sdim])
        next_states = self.data["next_states"]
        opponent_states = self.data["opponent_states"]
        team_states = self.data["team_states"]
        rewards = self.data["rewards"]
        actions = self.data["actions"]
        competitions = self.data["competitions"]

        # combine individual and team states
        if self.is_include_team_effect:
            raise NotImplementedError
        else:
            states_combined = states

        # check if reward is in state_vars
        if self.reward_idx is not None:
            next_states_combined = next_states
        else:
            next_states_combined = np.concatenate(
                [next_states, rewards.reshape(-1, 1)],
                axis=1,
                dtype=np.float32,
            )  ## last column is reward

        self.transition_models = [
            LinearRegressor(
                featurize_method="polynomial",
                degree=2,
                interaction_only=False,
                is_standarized=False,
            )
            for _ in range(2)
        ]
        for a in [0, 1]:
            states_combined_a = states_combined[actions.flatten() == a]
            next_states_combined_a = next_states_combined[actions.flatten() == a]
            self.transition_models[a].train(states_combined_a, next_states_combined_a)
        x = np.arange(15, 25, 0.1)
        import matplotlib.pyplot as plt

        for a in [0, 1]:
            y = self.transition_models[a].predict(x.reshape(-1, 1))[:, 1]
            plt.plot(x, y, label="action={}".format(a))
        # # save plot
        plt.legend()
        plt.savefig("../figures/transition_model.png")

        self.is_model_trained = True

    def next_time_point(
        self,
        state,
        action,
        opponent_state,
        team_state,
        competition,
        state_errors,
        reward_errors,
    ):

        states = np.array(state)
        actions = np.array(action)
        # opponent_states = np.array(opponent_state)
        # team_states = np.array(team_state)
        n = states.shape[0]

        if self.is_include_team_effect:
            raise NotImplementedError
        else:
            states_combined = states

        # init output
        out_samples = np.zeros(
            [
                n,
                (
                    len(self.next_states_min) + 1
                    if self.reward_idx is None
                    else len(self.next_states_min)
                ),
            ]
        )

        # sample next state and reward
        for a in [0, 1]:
            idx_a = actions.flatten() == a
            if sum(idx_a) == 0:
                continue
            states_combined_a = states_combined[idx_a]
            out_samples[idx_a] = self.transition_models[a].predict(states_combined_a)

            # if a == 1:
            #     idx = (states_combined[:, 0] > 20) & idx_a
            #     if sum(idx) > 0:
            #         out_samples[idx, 1] += 5
            # if a == 0:
            #     idx = (states_combined[:, 0] < 20) & idx_a
            #     if sum(idx) > 0:
            #         out_samples[idx, 1] += 5
            # states_combined_a = np.concatenate(
            #     [states_combined[idx_a], np.repeat([[a]], repeats=sum(idx_a), axis=0)],
            #     axis=1,
            # )
            # out_samples[idx_a] = self.transition_models[a].predict(states_combined_a)

        # extract next state and reward
        if self.reward_idx is not None:
            next_states_samples = out_samples + state_errors
            rewards_samples = next_states_samples[:, self.reward_idx]
        else:
            next_states_samples = out_samples[:, :-1] + state_errors
            rewards_samples = out_samples[:, -1] + reward_errors

        # clip next state and reward
        # if self.is_clip:
        #     next_states_samples = np.clip(
        #         next_states_samples, self.next_states_min, self.next_states_max
        #     )
        #     rewards_samples = np.clip(
        #         rewards_samples,
        #         self.rewards_min,
        #         self.rewards_max,
        #     )
        return next_states_samples, rewards_samples

    def sim_traj(
        self,
        n_indiv,
        n_teams,
        n_weeks,
        indiv_policy,
        match_policy,
        phi,
        seed,
        eval_mode=False,
    ):
        assert n_weeks < 12, "n_weeks should be less than 12 in the training data"
        # np.random.seed(seed)
        warmup_weeks = 1
        states_dim = len(self.state_vars)
        team_states_dim = len(
            [s for s in self.state_vars if s != "week" and s != "dow"]
        )

        # claim memory
        states = [
            np.zeros(
                [
                    n_indiv,
                    (n_weeks + warmup_weeks) * 7 + 1,
                    states_dim,
                ]
            )
            for _ in range(n_teams)
        ]  # first day is Sunday
        actions = [
            np.zeros([n_indiv, (n_weeks + warmup_weeks) * 7]) for _ in range(n_teams)
        ]
        rewards = [
            np.zeros([n_indiv, (n_weeks + warmup_weeks) * 7]) for _ in range(n_teams)
        ]
        team_states = [
            np.zeros([(n_weeks + warmup_weeks), team_states_dim])
            for _ in range(n_teams)
        ]
        opponent_states = [
            np.zeros([(n_weeks + warmup_weeks), team_states_dim])
            for _ in range(n_teams)
        ]
        team_rewards = [np.zeros([(n_weeks + warmup_weeks)]) for _ in range(n_teams)]
        competitions = [np.zeros([n_weeks + warmup_weeks]) for _ in range(n_teams)]

        # generate (correlated) random errors for states and rewards
        state_errors = self.generate_state_random_error(
            n_teams, n_indiv, (warmup_weeks + n_weeks) * 7
        )
        reward_errors = self.generate_reward_random_error(
            n_teams,
            n_indiv,
            (warmup_weeks + n_weeks) * 7,
            phi=phi,
            eval_mode=eval_mode,
        )

        # sample initial states
        (
            init_indiv_states,
            init_team_states,
            init_opponent_states,
            init_competition,
        ) = self.sample_init_state_competition_matching_state(
            n_team=n_teams, n_indiv=n_indiv, p_match=0.0
        )

        # simulate the trajectory from 1 to n_weeks
        current_indiv_states = copy.deepcopy(
            init_indiv_states
        )  # [np.array([n_indiv, sdim])] * n_teams
        current_team_states = copy.deepcopy(
            init_team_states
        )  # [np.array([sdim])] * n_teams
        current_opponent_states = copy.deepcopy(init_opponent_states)
        current_competitions = copy.deepcopy(init_competition)
        for team_id in range(n_teams):
            states[team_id][:, 0, :] = current_indiv_states[team_id].copy()

        for week in range(n_weeks + warmup_weeks):
            for team_id in range(n_teams):
                for day in range(7):
                    # individual level policy
                    current_actions = indiv_policy.act(
                        states=current_indiv_states[team_id]
                    )
                    next_indiv_states, inidv_rewards = self.next_time_point(
                        state=current_indiv_states[team_id],
                        action=current_actions,
                        opponent_state=current_opponent_states[team_id],
                        team_state=current_team_states[team_id],
                        competition=current_competitions[team_id],
                        state_errors=state_errors[team_id][:, week * 7 + day, :],
                        reward_errors=reward_errors[team_id][:, week * 7 + day],
                    )
                    # check whether include week and dow
                    next_indiv_states = np.hstack(
                        [
                            next_indiv_states,
                            np.zeros(
                                [n_indiv, self.is_include_dow + self.is_include_week]
                            ),
                        ]
                    )
                    if self.is_include_week:
                        week_idx = self.state_vars.index("week")
                        next_indiv_states[:, week_idx] = np.ones([n_indiv]) * week
                    if self.is_include_dow:
                        dow_index = self.state_vars.index("dow")
                        next_indiv_states[:, dow_index] = np.ones([n_indiv]) * (day + 1)
                    states[team_id][:, week * 7 + day + 1, :] = next_indiv_states.copy()
                    actions[team_id][:, week * 7 + day] = current_actions.copy()
                    rewards[team_id][:, week * 7 + day] = inidv_rewards.copy()

                    current_indiv_states[team_id] = next_indiv_states.copy()

            # team level policy
            ## TODO not fully implemented yet
            ##########################################
            team_state_to_act = np.zeros([n_teams, team_states_dim])
            # for team_id in range(n_teams):
            #     tmp = np.mean(
            #         states[team_id][:, ((week * 7) + 1) : ((week * 7 + 7) + 1)], axis=(0, 1)
            #     )
            #     if include_dow and include_week:
            #         team_state_to_act[team_id] = copy.deepcopy(
            #             tmp[:-2]
            #         )  # remove dow for matching
            #     elif include_week or include_dow:
            #         team_state_to_act[team_id] = copy.deepcopy(tmp[:-1])
            #     else:
            #         team_state_to_act[team_id] = copy.deepcopy(tmp)

            (
                next_competitions,  # competition[:, t],
                next_matching_opponent,
            ) = match_policy.act(states=team_state_to_act)

            ##########################################

            # for team_id in range(n_teams):
            #     # store team level states
            #     team_states[team_id][week, :] = current_team_states[team_id].copy()
            #     opponent_states[team_id][week, :] = current_opponent_states[team_id].copy()
            #     team_rewards[team_id][week] = np.mean(
            #         rewards[team_id][:, (week * 7) : (week * 7 + 7)]
            #     )
            #     competitions[team_id][week] = current_competitions[team_id]
            #     # update team level states
            #     tmp = np.mean(
            #         states[team_id][:, ((week * 7) + 1) : ((week * 7 + 7) + 1)], axis=(0, 1)
            #     )
            #     if include_dow and include_week:
            #         current_team_states[team_id] = copy.deepcopy(tmp[:-2])
            #     elif include_week or include_dow:
            #         current_team_states[team_id] = copy.deepcopy(tmp[:-1])
            #     else:
            #         current_team_states[team_id] = copy.deepcopy(tmp)
            #     current_competitions = next_competitions.copy()
            # for team_id in range(n_teams):
            #     current_opponent_states[team_id] = current_team_states[
            #         next_matching_opponent[team_id]
            #     ].copy()  # * wait to update until all current_team_states are updated

        # return trajectories
        ## remove warmup weeks
        return (
            [
                states[team_id][:, warmup_weeks * 7 :] for team_id in range(n_teams)
            ],  # [np.array([n_indiv, (n_weeks) * 7 + 1, sdim])] * n_teams
            [
                actions[team_id][:, warmup_weeks * 7 :] for team_id in range(n_teams)
            ],  # [np.array([n_indiv, (n_weeks) * 7])] * n_teams
            [
                rewards[team_id][:, warmup_weeks * 7 :] for team_id in range(n_teams)
            ],  # [np.array([n_indiv, (n_weeks) * 7])] * n_teams
            [
                team_states[team_id][warmup_weeks:, :] for team_id in range(n_teams)
            ],  # [np.array([(n_weeks), sdim])] * n_teams
            [
                opponent_states[team_id][warmup_weeks:, :] for team_id in range(n_teams)
            ],  # [np.array([(n_weeks), sdim])] * n_teams
            [
                team_rewards[team_id][warmup_weeks:] for team_id in range(n_teams)
            ],  # [np.array([(n_weeks)])] * n_teams
            [
                competitions[team_id][warmup_weeks:] for team_id in range(n_teams)
            ],  # [np.array([(n_weeks)])] * n_teams
        )


def plot_trajectory(states_sim, state_vars):
    import matplotlib.pyplot as plt

    ihs2018_data = IHS2018Data()
    _, _, state_vars = ihs2018_data.rename_vars(state_vars, "step")
    state_var = state_vars[0]

    if IS_DATA_AVAILABLE:
        raw_data = ihs2018_data.raw_data.sort_values(by=["USERID", "day"])
        raw_data = raw_data[["USERID", "day", state_var]]
        wide_data = pd.pivot_table(
            index="USERID", columns="day", values=state_var, data=raw_data
        )
        dat = wide_data.to_numpy()
        states_real = dat

        idxs = np.random.choice(min(states_real.shape[0], states_sim.shape[0]), size=20)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 4])
        for i in idxs:
            ax.plot(
                np.arange(states_real.shape[1]),
                states_real[i, :],
                color="red",
                alpha=0.5,
            )
            ax.plot(
                np.arange(states_sim.shape[1]),
                states_sim[i, :, 0],
                color="blue",
                alpha=0.5,
            )
        plt.legend(["real", "sim"])
        plt.savefig("../figures/{}_real_sim.png".format(state_var))
    else:
        idxs = np.random.choice(states_sim.shape[0], size=20)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 4])
        for i in idxs:
            ax.plot(
                np.arange(states_sim.shape[1]),
                states_sim[i, :, 0],
                color="blue",
                alpha=0.5,
            )
        plt.legend(["sim"])
        plt.savefig("../figures/{}_sim.png".format(state_var))


def test():
    # ihs2020_data = IHS2020Data()
    # data = ihs2020_data.load_data(["step", "week"], "step", is_transformed=True)
    state_vars = ["sleep"]
    ihs2018_semisyn_data = IHS2018SemiSynData(
        state_vars=state_vars,
        reward_var="mood",
        is_include_team_effect=False,
        is_transformed=False,
        is_clip=False,
        is_standarized=True,
    )
    # fit transition models
    ihs2018_semisyn_data.fit_transition_models()

    # save trained models
    ihs2018_semisyn_data.save_trained_model("../models")

    # load trained models
    ihs2018_semisyn_data.load_trained_model("../models")

    # sample trajectory
    (
        states,
        actions,
        rewards,
        team_states,
        opponent_states,
        team_rewards,
        competitions,
    ) = ihs2018_semisyn_data.sim_traj(
        n_indiv=30,
        n_teams=5,
        n_weeks=10,
        seed=0,
        indiv_policy=RandomIndivPolicy(p=0.5),
        match_policy=RandomMatchPolicy(),
        phi=0.0,
    )
    print(np.mean(rewards))
    # plot trajectory
    plot_trajectory(states[0], state_vars)


if __name__ == "__main__":
    test()
