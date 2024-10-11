import numpy as np
import sys, pickle, os
import pandas as pd

sys.path.append("../../")
import functions.GEE_Q as GEE_Q

# from functions.simulate_data_1d_flexible import *
import functions.utilities as uti
import functions.cov_struct as cov_structs
import ihs2018_semisyn
from agent_wrappers import RandomIndivPolicy, TrainedIndivPolicy, RandomMatchPolicy
from my_utils import glogger
from p_tqdm import p_map
from collections import namedtuple
import json, time
from DoubleDQN import OfflineDoubleDQN
from CQL import OfflineCQL

# from IPython.core import ultratb
from policy_learning import OfflineFQI

# sys.excepthook = ultratb.FormattedTB(
#     mode="Verbose", color_scheme="Linux", call_pdb=False
# )

CV_CRITERION = "min"  # "min"  # "min" "2" "3" "4" "5"
BASIS = "polynomial"  # "polynomial" "rbf"
GAMMA = 0.9
glogger.setLevel("WARNING")


def standardize(x, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / std, mean, std
    else:
        return (x - mean) / std


def destandardize(x, mean, std):
    return x * std + mean


def run_exp_one(
    seed,
    methods,
    other_methods,
    state_vars,
    reward_var,
    N,
    m,
    T,
    path_name,
    is_clip,
    is_normalized,
    phi,
    year,
    degree,
    cor_boost_coef,
    selected_num_basis,
):

    # parameters for evaluating the "optimal" policy
    gamma = GAMMA
    basis = BASIS  # "polynomial" "rbf"
    cv_criterion = CV_CRITERION  # "min" "2" "3" "4" "5"

    action_space = [0, 1]

    # collect data
    rollouts = []
    np.random.seed(
        int(
            42 * seed
            + 32429 * phi
            + 23585 * cor_boost_coef
            + 32134 * N
            + 14343 * m
            + 74245 * T
        )
    )
    if year == 2018:
        IS_DATA_AVAILABLE = ihs2018_semisyn.IS_DATA_AVAILABLE
        ihs_semisyn_data = ihs2018_semisyn.IHS2018SemiSynData(
            state_vars=state_vars,
            reward_var=reward_var,
            is_include_team_effect=False,
            is_transformed=False,
            is_clip=is_clip,
            is_standarized=True,
            degree=degree,
            cor_boost_coef=cor_boost_coef,
        )
    elif year == 2020:
        IS_DATA_AVAILABLE = ihs2020_semisyn.IS_DATA_AVAILABLE
        ihs_semisyn_data = ihs2020_semisyn.IHS2020SemiSynData(
            state_vars=state_vars,
            reward_var=reward_var,
            is_include_team_effect=False,
            is_transformed=True,
            is_clip=is_clip,
            is_standarized=True,
            degree=degree,
            cor_boost_coef=cor_boost_coef,
        )

    # train or load transition models
    if IS_DATA_AVAILABLE:
        ihs_semisyn_data.fit_transition_models()
        ihs_semisyn_data.save_trained_model("../models/")
    else:
        ihs_semisyn_data.load_trained_model("../models/")

    # sim trajectory
    (
        states0,
        actions0,
        rewards0,
        team_states0,
        opponent_states0,
        team_rewards0,
        competitions0,
    ) = ihs_semisyn_data.sim_traj(
        n_indiv=N,
        n_teams=m,
        n_weeks=T,
        indiv_policy=RandomIndivPolicy(p=0.5),
        match_policy=RandomMatchPolicy(),
        phi=phi,
        seed=seed,
    )

    if is_normalized:
        states_mean = np.mean(
            np.concatenate(states0, axis=0).reshape(-1, states0[0].shape[-1]), axis=0
        )
        states_std = np.std(
            np.concatenate(states0, axis=0).reshape(-1, states0[0].shape[-1]), axis=0
        )
        rewards_mean = np.mean(np.concatenate(rewards0, axis=0).flatten())
        rewards_std = np.std(np.concatenate(rewards0, axis=0).flatten())

    for i in range(m):
        states1 = states0[i]
        actions1 = actions0[i]
        rewards1 = rewards0[i]

        if is_normalized:
            states1 = standardize(
                states1.reshape(-1, states1.shape[-1]), states_mean, states_std
            ).reshape(states1.shape)
            rewards1 = standardize(rewards1, rewards_mean, rewards_std)

        res = {
            "states": states1[:, :-1, :],
            "actions": actions1,
            "next_states": states1[:, 1:, :],
            "rewards": rewards1,
        }
        rollouts.append(res)

    agents = [RandomIndivPolicy(p=0.5), RandomIndivPolicy(p=1)]
    for method in methods:
        if method == "exchangeable":
            cov_struct = cov_structs.Exchangeable()
            statsmodels_cov = cov_structs.Exchangeable_sm()
            separate_corr = 0
            rho = None
        elif method == "independent":
            cov_struct = cov_structs.Independence()
            statsmodels_cov = cov_structs.Independence_sm()
            separate_corr = 0
            rho = None
        elif method == "ordinary":
            cov_struct = cov_structs.Independence()
            statsmodels_cov = None
            separate_corr = 0
            rho = None
        elif method == "autoex":
            cov_struct = cov_structs.Autoex(set_weight=None, var_td=1)
            statsmodels_cov = cov_structs.Autoex_sm()
            separate_corr = 0
            rho = None
        elif method == "exchangeable_subjects":
            cov_struct = cov_structs.Exchangeable_subjects()
            statsmodels_cov = cov_structs.Exchangeable_subjects_sm()
            separate_corr = 0
            rho = None
        elif method == "ordinary_exchangeable_subjects":
            cov_struct = cov_structs.Exchangeable_subjects()
            statsmodels_cov = cov_structs.Exchangeable_subjects_sm()
            separate_corr = 0
            rho = None
        elif method == "autoex_exsubject":
            cov_struct = cov_structs.Autoex_exsubject(set_weight=None, var_td=1)
            statsmodels_cov = cov_structs.Autoex_exsubject_sm()
            separate_corr = 0
            rho = None
        elif method == "ordinary_autoex_exsubject":
            cov_struct = cov_structs.Autoex_exsubject(set_weight=None, var_td=1)
            statsmodels_cov = cov_structs.Autoex_exsubject_sm()
            separate_corr = 0
            rho = None
        else:
            raise ValueError("Invalid method")

        # CV select number of basis
        cv_loss = "tdssq"
        if cv_criterion == "min":
            if os.path.exists(os.path.join(path_name, "CV_basis_res.json")):
                Basis_res = namedtuple(
                    "Basis_res",
                    [
                        "num_basis_min",
                        "num_basis_1se",
                        "cv_loss",
                        "cv_se",
                        "basis",
                        "num_basis_list",
                    ],
                )
                with open(os.path.join(path_name, "CV_basis_res.json"), "r") as f:
                    data = json.loads(f.read())
                CV_basis_res = Basis_res(**data)
            else:
                if basis == "polynomial":
                    num_basis_list = [1, 2, 3]
                elif basis == "rbf":
                    num_basis_list = [2, 3, 4, 5]
                cv_seed = 42
                CV_basis_res = uti.select_num_basis_cv(
                    rollouts,
                    cov_struct,
                    statsmodels_cov,
                    action_space,
                    basis=basis,
                    num_threads=10,
                    seed=cv_seed,
                    gamma=gamma,
                    num_basis_list=num_basis_list,
                    num_batches=1,
                    new_GEE=1,
                    file_path=path_name,
                    optimal_GEE=(
                        0
                        if method == "ordinary" or method == "ordinary_autoex_exsubject"
                        else 1
                    ),
                    combine_actions=1,
                    refit=0,
                )
                with open(os.path.join(path_name, "CV_basis_res.json"), "w") as f:
                    f.write(json.dumps(CV_basis_res._asdict()))
            if cv_criterion == "min":
                selected_num_basis = CV_basis_res.num_basis_min[cv_loss]
            elif cv_criterion == "1se":
                selected_num_basis = CV_basis_res.num_basis_1se[cv_loss]
            cluster = [
                GEE_Q.Cluster(**rollout, basis=basis, num_basis=selected_num_basis)
                for rollout in rollouts
            ]
        else:
            cluster = [
                GEE_Q.Cluster(**rollout, basis=basis, num_basis=int(cv_criterion))
                for rollout in rollouts
            ]
        if (
            method == "ordinary"
            or method == "ordinary_autoex_exsubject"
            or method == "ordinary_exchangeable_subjects"
        ):
            GEE = GEE_Q.GEE_fittedQ(
                cov_struct,
                statsmodels_cov=statsmodels_cov,
                rho=rho,
                gamma=gamma,
                action_space=action_space,
                separate_corr=separate_corr,
                statsmodel_maxiter=1,
                combine_actions=1,
                optimal_GEE=0,
            )
        else:
            GEE = GEE_Q.GEE_fittedQ(
                cov_struct,
                statsmodels_cov=statsmodels_cov,
                rho=rho,
                gamma=gamma,
                action_space=action_space,
                separate_corr=separate_corr,
                statsmodel_maxiter=1,
                combine_actions=1,
                optimal_GEE=1,
            )

        # %%
        if os.path.exists(os.path.join(path_name, "seed_" + str(seed) + ".dat")):
            raise NotImplementedError
            # with open(os.path.join(path_name, "seed_" + str(seed) + ".dat"), "rb") as f:
            #     qmodels = pickle.load(f)["q_function_list"]
            # GEE.q_function_list = qmodels
        else:
            GEE.fit(
                cluster,
                num_batches=1,
                max_iter=50,
                batch_iter=50,
                verbose=1,
                accelerate_method="batch_processing",
                global_TD=True,
            )
            if method == "autoex_exsubject":
                with open(os.path.join(path_name, "autocoef.csv"), "a") as f:
                    f.write(
                        str(GEE.cov_struct.dep_params["autoregressive_coef"]) + "\n"
                    )
            elif method == "exchangeable_subjects":
                with open(os.path.join(path_name, "corrcoef.csv"), "a") as f:
                    f.write(str(GEE.cov_struct.dep_params) + "\n")
        if is_normalized:
            agents.append(
                TrainedIndivPolicy(
                    agent=GEE.predict_action,
                    name=method,
                    state_mean=states_mean,
                    state_std=states_std,
                )
            )
        else:
            agents.append(
                TrainedIndivPolicy(
                    agent=GEE.predict_action,
                    name=method,
                    state_mean=None,
                    state_std=None,
                )
            )

    for method in other_methods:
        states = np.concatenate(states0, axis=0)
        actions = np.concatenate(actions0, axis=0)
        rewards = np.concatenate(rewards0, axis=0)
        if method == "fqi_nn":
            agent = OfflineFQI(model_type="nn", action_size=2)
            agent.fit(states, actions, rewards, max_iter=100)
        elif method == "fqi_lm":
            agent = OfflineFQI(model_type="lm", action_size=2)
            agent.fit(states, actions, rewards, max_iter=100)
        elif method == "ddqn":
            agent = OfflineDoubleDQN(
                state_dim=1,
                action_dim=2,
                hidden_dims=[16],
                learning_rate=1e-3,
                gamma=0.9,
                batch_size=256,
            )
            states1 = states[:, :-1, :]
            next_states1 = states[:, 1:, :]
            ddqn_loss = agent.train(
                states1,
                actions,
                rewards,
                next_states1,
                num_iterations=20000,
                update_frequency=100,
            )
        elif method == "cql":
            agent = OfflineCQL(
                state_dim=1,
                action_dim=2,
                hidden_dims=[16],
                learning_rate=1e-3,
                gamma=0.9,
                batch_size=256,
            )
            states1 = states[:, :-1, :]
            next_states1 = states[:, 1:, :]
            cql_loss = agent.train(
                states1,
                actions,
                rewards,
                next_states1,
                num_iterations=20000,
                update_frequency=100,
            )
        else:
            raise ValueError("Invalid method")

        agents.append(
            TrainedIndivPolicy(
                agent=agent.act, name=method, state_mean=None, state_std=None
            )
        )
        # print(agent.model[0].model.intercept_)
        # print(agent.model[0].model.coef_)
        # print(agent.model[1].model.intercept_)
        # print(agent.model[1].model.coef_)

    eval_seed = seed * 10
    N_eva = 100  # 0
    T_eva = 5

    def evaluate_policy(ihs_semisyn_data, indiv_policy, match_policy, seed):
        IS_DATA_AVAILABLE = ihs2018_semisyn.IS_DATA_AVAILABLE
        ihs_semisyn_data = ihs2018_semisyn.IHS2018SemiSynData(
            state_vars=state_vars,
            reward_var=reward_var,
            is_include_team_effect=False,
            is_transformed=False,
            is_clip=is_clip,
            is_standarized=True,
            degree=degree,
            cor_boost_coef=1,
        )

        # train or load transition models
        if IS_DATA_AVAILABLE:
            ihs_semisyn_data.fit_transition_models()
            ihs_semisyn_data.save_trained_model("../models/")
        else:
            ihs_semisyn_data.load_trained_model("../models/")
        (
            states0,
            actions0,
            rewards0,
            team_states0,
            opponent_states0,
            team_rewards0,
            competitions0,
        ) = ihs_semisyn_data.sim_traj(
            n_indiv=N_eva,
            n_teams=10,
            n_weeks=T_eva,
            indiv_policy=indiv_policy,
            match_policy=match_policy,
            seed=seed,
            phi=0,
            eval_mode=True,
        )
        states = np.concatenate(states0, axis=0)
        actions = np.concatenate(actions0, axis=0)
        rewards = np.concatenate(rewards0, axis=0)

        av_r = np.mean(rewards)
        dis_r = 0
        for t in range(rewards.shape[1]):
            dis_r += gamma**t * np.mean(rewards[:, t])
        return av_r, dis_r, np.unique(actions, return_counts=1)

    res = []
    for agent in agents:
        av_r, dis_r, unique_action = evaluate_policy(
            ihs_semisyn_data=ihs_semisyn_data,
            match_policy=RandomMatchPolicy(),
            indiv_policy=agent,
            seed=eval_seed,
        )
        res.append((av_r, dis_r, unique_action, agent.name))

    res = pd.DataFrame(res, columns=["av_r", "dis_r", "unique_action", "method"])

    return res


def run_exp_one_star(args):
    return run_exp_one(*args)


def run_exp(cbc):
    methods = [
        "exchangeable_subjects",
        "independent",
        "ordinary",
        "ordinary_exchangeable_subjects",
    ]

    other_methods = ["fqi_lm", "ddqn", "cql"]
    num_repeats = 50
    # num_indivs = [10, 20, 30, 40, 50]
    num_indivs = [10]
    # num_teams = [5, 10, 15, 20, 25]
    num_teams = [5]
    num_weeks = [1, 2, 3, 4, 5]
    # num_weeks = [2]
    SELECTED_NUM_BASISs = [2]  # use cv, not used

    job_list = [
        # (["sleep"], "mood"),
        (["step"], "sleep"),
        # (["mood"], "step"),
        # (["sleep"], "step"),
    ]

    is_clip = False
    is_normalized = False  # normalized before Q learning
    phis = [0]
    year = 2018
    degree = 1
    cor_boost_coef = cbc  # 1: original data, 2: boost correlation by 2 times
    # normalized = True
    CORES = 15

    note = ""
    for phi in phis:
        for state_vars, reward_var in job_list:
            for num_indiv in num_indivs:
                for num_team in num_teams:
                    for num_week in num_weeks:
                        for selected_num_basis in SELECTED_NUM_BASISs:

                            def setpath():
                                if not os.path.exists("../results"):
                                    os.makedirs("../results", exist_ok=True)
                                path_name = (
                                    "../results/ihs{}_individual_rl_results/lm_{}_{}_".format(
                                        year, "-".join(state_vars), reward_var
                                    )
                                    + "_cv_criterion"
                                    + str(CV_CRITERION)
                                    + "_"
                                    + BASIS
                                    + "_m_"
                                    + str(num_team)
                                    + "_N_"
                                    + str(num_indiv)
                                    + "_T_"
                                    + str(num_week)
                                    + "_clip_"
                                    + str(is_clip)
                                    + "_degree_"
                                    + str(degree)
                                    + "_phi_"
                                    + str(phi)
                                    + "_cbc_"
                                    + str(cor_boost_coef)
                                    + "_"
                                    + "numbasis_"
                                    + str(selected_num_basis)
                                    + note
                                )
                                if not os.path.exists(path_name):
                                    os.makedirs(path_name, exist_ok=True)
                                return path_name

                            path_name = setpath()

                            # select number of basis
                            if CV_CRITERION == "min":
                                glogger.warning("select_basis")
                                out0 = p_map(
                                    run_exp_one,
                                    [1000],
                                    [methods],
                                    [other_methods],
                                    [state_vars],
                                    [reward_var],
                                    [num_indiv],
                                    [num_team],
                                    [num_week],
                                    [path_name],
                                    [is_clip],
                                    [is_normalized],
                                    [phi],
                                    [year],
                                    [degree],
                                    [cor_boost_coef],
                                    [selected_num_basis],
                                    num_cpus=CORES,
                                    desc="select_basis",
                                )
                                outs = p_map(
                                    run_exp_one,
                                    [i + 20 for i in range(num_repeats - 1)],
                                    [methods] * (num_repeats - 1),
                                    [other_methods] * (num_repeats - 1),
                                    [state_vars] * (num_repeats - 1),
                                    [reward_var] * (num_repeats - 1),
                                    [num_indiv] * (num_repeats - 1),
                                    [num_team] * (num_repeats - 1),
                                    [num_week] * (num_repeats - 1),
                                    [path_name] * (num_repeats - 1),
                                    [is_clip] * (num_repeats - 1),
                                    [is_normalized] * (num_repeats - 1),
                                    [phi] * (num_repeats - 1),
                                    [year] * (num_repeats - 1),
                                    [degree] * (num_repeats - 1),
                                    [cor_boost_coef] * (num_repeats - 1),
                                    [selected_num_basis] * (num_repeats - 1),
                                    num_cpus=CORES,
                                )
                            else:
                                outs = p_map(
                                    run_exp_one,
                                    [i + 20 for i in range(num_repeats)],
                                    [methods] * (num_repeats),
                                    [other_methods] * (num_repeats),
                                    [state_vars] * (num_repeats),
                                    [reward_var] * (num_repeats),
                                    [num_indiv] * (num_repeats),
                                    [num_team] * (num_repeats),
                                    [num_week] * (num_repeats),
                                    [path_name] * (num_repeats),
                                    [is_clip] * (num_repeats),
                                    [is_normalized] * (num_repeats),
                                    [phi] * (num_repeats),
                                    [year] * (num_repeats),
                                    [degree] * (num_repeats),
                                    [cor_boost_coef] * (num_repeats),
                                    [selected_num_basis] * (num_repeats),
                                    num_cpus=CORES,
                                )
                            try:
                                outs = pd.concat(out0 + outs)
                            except:
                                outs = pd.concat(outs)
                            outs.to_csv(os.path.join(path_name, "results.csv"))


def run_exp_test(cbc):
    methods = [
        "exchangeable_subjects",
        "independent",
        "ordinary",
        "ordinary_exchangeable_subjects",
    ]
    other_methods = ["fqi_lm", "ddqn", "cql"]
    num_repeats = 4
    num_indivs = [20]
    num_teams = [5]
    num_weeks = [2]
    SELECTED_NUM_BASISs = [2]

    job_list = [
        (["step"], "sleep"),
    ]

    is_clip = False
    is_normalized = False  # normalized before Q learning
    phis = [0.0]
    year = 2018
    degree = 1
    cor_boost_coef = cbc  # 1: original data, 2: boost correlation by 2 times
    # normalized = True
    CORES = 4

    note = ""
    for phi in phis:
        for state_vars, reward_var in job_list:
            for num_indiv in num_indivs:
                for num_team in num_teams:
                    for num_week in num_weeks:
                        for selected_num_basis in SELECTED_NUM_BASISs:

                            def setpath():
                                if not os.path.exists("../results"):
                                    os.makedirs("../results", exist_ok=True)
                                path_name = (
                                    "../results/ihs{}_individual_rl_resultstest/lm_{}_{}_".format(
                                        year, "-".join(state_vars), reward_var
                                    )
                                    + "_cv_criterion"
                                    + str(CV_CRITERION)
                                    + "_"
                                    + BASIS
                                    + "_m_"
                                    + str(num_team)
                                    + "_N_"
                                    + str(num_indiv)
                                    + "_T_"
                                    + str(num_week)
                                    + "_clip_"
                                    + str(is_clip)
                                    + "_degree_"
                                    + str(degree)
                                    + "_phi_"
                                    + str(phi)
                                    + "_cbc_"
                                    + str(cor_boost_coef)
                                    + "_"
                                    + "numbasis_"
                                    + str(selected_num_basis)
                                    + note
                                )
                                if not os.path.exists(path_name):
                                    os.makedirs(path_name, exist_ok=True)
                                return path_name

                            path_name = setpath()

                            # select number of basis
                            if CV_CRITERION == "min":
                                glogger.warning("select_basis")
                                out0 = run_exp_one(
                                    100,
                                    methods,
                                    other_methods,
                                    state_vars,
                                    reward_var,
                                    num_indiv,
                                    num_team,
                                    num_week,
                                    path_name,
                                    is_clip,
                                    is_normalized,
                                    phi,
                                    year,
                                    degree,
                                    cor_boost_coef,
                                    selected_num_basis,
                                )

                                outs = p_map(
                                    run_exp_one,
                                    [i + 20 for i in range(num_repeats - 1)],
                                    [methods] * (num_repeats - 1),
                                    [other_methods] * (num_repeats - 1),
                                    [state_vars] * (num_repeats - 1),
                                    [reward_var] * (num_repeats - 1),
                                    [num_indiv] * (num_repeats - 1),
                                    [num_team] * (num_repeats - 1),
                                    [num_week] * (num_repeats - 1),
                                    [path_name] * (num_repeats - 1),
                                    [is_clip] * (num_repeats - 1),
                                    [is_normalized] * (num_repeats - 1),
                                    [phi] * (num_repeats - 1),
                                    [year] * (num_repeats - 1),
                                    [degree] * (num_repeats - 1),
                                    [cor_boost_coef] * (num_repeats - 1),
                                    [selected_num_basis] * (num_repeats - 1),
                                    num_cpus=CORES,
                                    desc="run",
                                )
                            else:
                                outs = p_map(
                                    run_exp_one,
                                    [i + 20 for i in range(num_repeats)],
                                    [methods] * (num_repeats),
                                    [other_methods] * (num_repeats),
                                    [state_vars] * (num_repeats),
                                    [reward_var] * (num_repeats),
                                    [num_indiv] * (num_repeats),
                                    [num_team] * (num_repeats),
                                    [num_week] * (num_repeats),
                                    [path_name] * (num_repeats),
                                    [is_clip] * (num_repeats),
                                    [is_normalized] * (num_repeats),
                                    [phi] * (num_repeats),
                                    [year] * (num_repeats),
                                    [degree] * (num_repeats),
                                    [cor_boost_coef] * (num_repeats),
                                    [selected_num_basis] * (num_repeats),
                                    num_cpus=CORES,
                                    desc="run",
                                )
                            try:
                                outs = pd.concat([out0] + outs)
                            except:
                                outs = pd.concat(outs)
                            outs.to_csv(os.path.join(path_name, "results.csv"))


if __name__ == "__main__":
    # run_exp1() # only for state_var == reward_var
    run_exp(1)
