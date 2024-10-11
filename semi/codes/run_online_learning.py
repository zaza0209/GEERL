import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from q_learning import ReplayBuffer, dqn_model, dqn_model_linear
from my_utils import glogger
import ihs2018_semisyn
from agent_wrappers import RandomIndivPolicy, RandomMatchPolicy
from p_tqdm import p_map
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "1"


from torch.utils.tensorboard import SummaryWriter
import os
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

# ts_writer = SummaryWriter(os.path.join("./ts_outs/"))
ts_writer = None


class TrainedIndivPolicy:
    def __init__(self, agent, name="", epsilon=0.0) -> None:
        self.agent = agent
        self.name = "trained_{}".format(name)
        self.epsilon = epsilon

    def act(self, states):
        # print(states.shape)
        states = states.reshape(states.shape[0], states.shape[1])
        actions_greedy = self.agent(states)
        # print(actions_greedy)
        actions_random = np.random.binomial(n=1, p=0.5, size=[states.shape[0]])
        I_random = np.random.binomial(n=1, p=self.epsilon, size=[states.shape[0]])
        return (1 - I_random) * actions_greedy + I_random * actions_random


def run_online_dqn_learning(
    seed,
    state_vars,
    reward_var,
    phi,
    cbc,
    model_parameters,
    train_parameters,
    ts_writer=ts_writer,
):
    print(state_vars, reward_var, cbc, phi)
    # simulation environment
    ihs_semisyn_data = ihs2018_semisyn.IHS2018SemiSynData(
        state_vars=state_vars,
        reward_var=reward_var,
        is_include_team_effect=False,
        is_transformed=False,
        is_clip=False,
        is_standarized=True,
        degree=1,
        cor_boost_coef=cbc,
    )
    IS_DATA_AVAILABLE = ihs2018_semisyn.IS_DATA_AVAILABLE
    if IS_DATA_AVAILABLE:
        ihs_semisyn_data.fit_transition_models()
        ihs_semisyn_data.save_trained_model("../models/")
    else:
        ihs_semisyn_data.load_trained_model("../models/")

    # learning

    torch.set_num_threads(1)
    T_eval = 10

    ## set parameter
    gamma = model_parameters["gamma"]
    batch_size, iters, target_update_freq, learning_rate, grad_norm = (
        train_parameters["batch_size"],
        train_parameters["iters"],
        train_parameters["target_update_freq"],
        train_parameters["learning_rate"],
        train_parameters["grad_norm"],
    )

    ## pre training
    device = torch.device("cpu")
    model = dqn_model(model_parameters=model_parameters).to(device)
    optimizer = optim.Adam(model.Q.parameters(), lr=learning_rate)  # 0.001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000000, gamma=0.5)

    model.Q.train()
    model.target_Q.eval()
    num_updates = 0
    epsilon = 0.99
    T = 10

    replay_buffer = ReplayBuffer(100000)

    for i in range(iters):

        # initialize state
        states, _, _, _ = ihs_semisyn_data.sample_init_state_competition_matching_state(
            n_team=1, n_indiv=1, p_match=0
        )
        state = states[0]
        episode_reward = 0

        # generate random errors
        state_errors = ihs_semisyn_data.generate_state_random_error(
            n_team=1, n_indiv=1, T=T * 7
        )
        state_error = state_errors[0]
        reward_errors = ihs_semisyn_data.generate_reward_random_error(
            n_team=1,
            n_indiv=1,
            T=T * 7,
            phi=0,
            eval_mode=True,
        )
        reward_error = reward_errors[0]

        for t in range(70):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice([0, 1])
            else:
                with torch.no_grad():
                    action = (
                        model.Q(torch.FloatTensor(state.reshape(1, -1))).argmax().item()
                    )

            next_state, reward = ihs_semisyn_data.next_time_point(
                state=state,
                action=action,
                opponent_state=None,
                team_state=None,
                competition=None,
                state_errors=state_error[:, t],
                reward_errors=reward_error[:, t],
            )
            episode_reward += reward.flatten()

            # store in replay buffer
            replay_buffer.append(state, action, reward, next_state)

            if len(replay_buffer) > batch_size:
                states0, actions0, rewards0, next_states0 = replay_buffer.sample(
                    batch_size
                )
                # with torch.no_grad():
                #     next_q_values = model.target_Q(torch.FloatTensor(next_states0))

                # target_q_value = (
                #     torch.FloatTensor(rewards0) + gamma * next_q_values.max(1)[0]
                # ).detach()

                # q_value = (
                #     model.Q(torch.FloatTensor(states0))
                #     .gather(1, torch.tensor(actions0, dtype=torch.int64).reshape(-1, 1))
                #     .squeeze(-1)
                # )
                # loss = (q_value - target_q_value).pow(2).mean()

                loss = model.loss_function(states0, actions0, rewards0, next_states0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        epsilon = max(0.01, epsilon * 0.995)

        if i % target_update_freq == 0:
            model.update_target_net()
            num_updates += 1
            if ts_writer:
                ts_writer.add_scalar("DQN update", num_updates, i)
        if i % 10 == 0:
            with torch.no_grad():
                (
                    states0,
                    actions0,
                    rewards0,
                    team_states0,
                    opponent_states0,
                    team_rewards0,
                    competitions0,
                ) = ihs_semisyn_data.sim_traj(
                    n_indiv=10,
                    n_teams=10,
                    n_weeks=2,
                    indiv_policy=TrainedIndivPolicy(model, epsilon=0.0),
                    match_policy=RandomMatchPolicy(),
                    phi=phi,
                    seed=seed,
                )
                rewards = np.concatenate(rewards0, axis=0)
                actions = np.concatenate(actions0, axis=0)
                props = np.mean(actions.flatten())
                av_r = np.mean(rewards)
                dis_r = 0
                for t in range(rewards.shape[1]):
                    dis_r += gamma**t * np.mean(rewards[:, t])

            glogger.info(
                "Iter: {},epsilon:{}, DQN training loss: {}, dis r:{}, av r:{},props:{}, qvalues:{}".format(
                    i, epsilon, loss.item(), dis_r, av_r, props, model.evaluate_Q()
                )
            )
            if ts_writer:
                ts_writer.add_scalar("DQN training loss", loss.item(), i)
                grad_norm_now = nn.utils.clip_grad_norm_(model.Q.parameters(), 10000000)
                ts_writer.add_scalar("DQN grad norm", grad_norm_now, i)
                ts_writer.add_scalar("DQN dis r", dis_r, i)
                ts_writer.add_scalar("DQN av r", av_r, i)
                ts_writer.add_scalar("DQN prop 1", props, i)

    # evaluation
    dis_rs = []
    av_rs = []
    props = []
    with torch.no_grad():
        for i in range(10):
            N_eval = 100
            T_eval = 5
            (
                states0,
                actions0,
                rewards0,
                team_states0,
                opponent_states0,
                team_rewards0,
                competitions0,
            ) = ihs_semisyn_data.sim_traj(
                n_indiv=N_eval,
                n_teams=10,
                n_weeks=T_eval,
                indiv_policy=TrainedIndivPolicy(model, epsilon=0.0),
                match_policy=RandomMatchPolicy(),
                phi=phi,
                seed=seed,
                eval_mode=True,
            )
            rewards = np.concatenate(rewards0, axis=0)
            actions = np.concatenate(actions0, axis=0)
            prop = np.mean(actions.flatten())
            av_r = np.mean(rewards)
            dis_r = 0
            for t in range(rewards.shape[1]):
                dis_r += gamma**t * np.mean(rewards[:, t])
            dis_rs.append(dis_r)
            av_rs.append(av_r)
            props.append(prop)

        return model, dis_rs, av_rs, props


def run_online_dqn_learning__(
    seed,
    state_vars,
    reward_var,
    phi,
    cbc,
    model_parameters,
    train_parameters,
    ts_writer=ts_writer,
):
    print(state_vars, reward_var, cbc, phi)
    # simulation environment
    ihs_semisyn_data = ihs2018_semisyn.IHS2018SemiSynData(
        state_vars=state_vars,
        reward_var=reward_var,
        is_include_team_effect=False,
        is_transformed=False,
        is_clip=False,
        is_standarized=True,
        degree=1,
        cor_boost_coef=cbc,
    )
    IS_DATA_AVAILABLE = ihs2018_semisyn.IS_DATA_AVAILABLE
    if IS_DATA_AVAILABLE:
        ihs_semisyn_data.fit_transition_models()
        ihs_semisyn_data.save_trained_model("../models/")
    else:
        ihs_semisyn_data.load_trained_model("../models/")

    # learning

    torch.set_num_threads(1)
    T_eval = 10

    ## set parameter
    gamma = model_parameters["gamma"]
    batch_size, iters, target_update_freq, learning_rate, grad_norm = (
        train_parameters["batch_size"],
        train_parameters["iters"],
        train_parameters["target_update_freq"],
        train_parameters["learning_rate"],
        train_parameters["grad_norm"],
    )

    ## create replay buffer
    buffer = ReplayBuffer(100000)

    ## pre training
    device = torch.device("cpu")
    model = dqn_model(model_parameters=model_parameters).to(device)
    optimizer = optim.Adam(model.Q.parameters(), lr=learning_rate)  # 0.001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000000, gamma=0.5)

    model.Q.train()
    model.target_Q.eval()
    num_updates = 0

    # init env and pre collection
    N = 10
    T = 10
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
        n_teams=2,
        n_weeks=T,
        indiv_policy=RandomIndivPolicy(p=0.5),
        match_policy=RandomMatchPolicy(),
        phi=phi,
        seed=seed,
    )
    # print(states0, actions0, rewards0, np.mean(rewards0))
    # exit()

    states1 = np.concatenate(states0, axis=0)[:, :-1].reshape(-1, states0[0].shape[-1])
    actions1 = np.concatenate(actions0, axis=0).reshape(-1, 1)
    rewards1 = np.concatenate(rewards0, axis=0).reshape(-1, 1)
    next_states1 = np.concatenate(states0, axis=0)[:, 1:].reshape(
        -1, states0[0].shape[-1]
    )

    buffer.append(states1, actions1, rewards1, next_states1)

    for i in range(iters):
        # sample from buffer
        s, a, r, sp = buffer.sample(batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.int64, device=device)
        r = torch.tensor(r, dtype=torch.float32, device=device)
        sp = torch.tensor(sp, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        loss = model.loss_function(s, a, r, sp)  # problem is here
        loss.backward()
        # nn.utils.clip_grad_norm_(model.Q.parameters(), grad_norm)
        # nn.utils.clip_grad_value_(model.Q.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if i % target_update_freq == 0:
            model.update_target_net()
            num_updates += 1
            if ts_writer:
                ts_writer.add_scalar("DQN update", num_updates, i)
        if i % 10 == 0:
            with torch.no_grad():
                (
                    states0,
                    actions0,
                    rewards0,
                    team_states0,
                    opponent_states0,
                    team_rewards0,
                    competitions0,
                ) = ihs_semisyn_data.sim_traj(
                    n_indiv=10,
                    n_teams=10,
                    n_weeks=2,
                    indiv_policy=TrainedIndivPolicy(model, epsilon=0.0),
                    match_policy=RandomMatchPolicy(),
                    phi=phi,
                    seed=seed,
                )
                rewards = np.concatenate(rewards0, axis=0)
                actions = np.concatenate(actions0, axis=0)
                props = np.mean(actions.flatten())
                av_r = np.mean(rewards)
                dis_r = 0
                for t in range(rewards.shape[1]):
                    dis_r += gamma**t * np.mean(rewards[:, t])

            glogger.info(
                "Iter: {}, DQN training loss: {}, dis r:{}, av r:{},props:{}, qvalues:{}".format(
                    i, loss.item(), dis_r, av_r, props, model.evaluate_Q()
                )
            )
            if ts_writer:
                ts_writer.add_scalar("DQN training loss", loss.item(), i)
                grad_norm_now = nn.utils.clip_grad_norm_(model.Q.parameters(), 10000000)
                ts_writer.add_scalar("DQN grad norm", grad_norm_now, i)
                ts_writer.add_scalar("DQN dis r", dis_r, i)
                ts_writer.add_scalar("DQN av r", av_r, i)
                ts_writer.add_scalar("DQN prop 1", props, i)

        # sample new data from env
        with torch.no_grad():
            (
                states0,
                actions0,
                rewards0,
                team_states0,
                opponent_states0,
                team_rewards0,
                competitions0,
            ) = ihs_semisyn_data.sim_traj(
                n_indiv=1,
                n_teams=2,
                n_weeks=2,
                indiv_policy=TrainedIndivPolicy(model, epsilon=0.2),
                match_policy=RandomMatchPolicy(),
                phi=phi,
                seed=seed,
            )
            states1 = np.concatenate(states0, axis=0)[:, :-1].reshape(
                -1, states0[0].shape[-1]
            )
            actions1 = np.concatenate(actions0, axis=0).reshape(-1, 1)
            rewards1 = np.concatenate(rewards0, axis=0).reshape(-1, 1)
            next_states1 = np.concatenate(states0, axis=0)[:, 1:].reshape(
                -1, states0[0].shape[-1]
            )

            buffer.append(states1, actions1, rewards1, next_states1)

    # evaluation
    dis_rs = []
    av_rs = []
    props = []
    with torch.no_grad():
        for i in range(10):
            N_eval = 100
            T_eval = 5
            (
                states0,
                actions0,
                rewards0,
                team_states0,
                opponent_states0,
                team_rewards0,
                competitions0,
            ) = ihs_semisyn_data.sim_traj(
                n_indiv=N_eval,
                n_teams=10,
                n_weeks=T_eval,
                indiv_policy=TrainedIndivPolicy(model, epsilon=0.0),
                match_policy=RandomMatchPolicy(),
                phi=phi,
                seed=seed,
                eval_mode=True,
            )
            rewards = np.concatenate(rewards0, axis=0)
            actions = np.concatenate(actions0, axis=0)
            prop = np.mean(actions.flatten())
            av_r = np.mean(rewards)
            dis_r = 0
            for t in range(rewards.shape[1]):
                dis_r += gamma**t * np.mean(rewards[:, t])
            dis_rs.append(dis_r)
            av_rs.append(av_r)
            props.append(prop)

        return model, dis_rs, av_rs, props


def run_online_dqn_learning_star(args):
    return run_online_dqn_learning(*args)


def run_exp_test():

    phis = [0.2, 0.5, 0.8]
    cbcs = [1, 2, 3, 4]

    cbcs = [1]
    phis = [0.0]

    # job_list = [
    #     (["step"], "mood"),
    #     (["sleep"], "mood"),
    #     (["step"], "sleep"),
    #     (["sleep"], "step"),
    #     (["mood"], "step"),
    #     (["mood"], "sleep"),
    # ]

    job_list = [
        (["step"], "sleep"),
    ]

    SEED = 42
    CORES = 4

    model_parameters = {
        "in_dim": 1,
        "out_dim": 2,
        "hidden_dims": [8, 8],
        "gamma": 0.9,
    }
    train_parameters = {
        "batch_size": 64,
        "iters": 5000,
        "target_update_freq": 100,
        "learning_rate": 0.001,
        "grad_norm": 10,
    }
    # train_parameters = {
    #     "batch_size": 256,
    #     "iters": 10,
    #     "target_update_freq": 10,
    #     "learning_rate": 0.001,
    #     "grad_norm": 5,
    # }

    res = pd.DataFrame()
    jobs = []
    for phi in phis:
        for cbc in cbcs:
            for state_vars, reward_var in job_list:
                jobs.append(
                    [
                        SEED,
                        state_vars,
                        reward_var,
                        phi,
                        cbc,
                        model_parameters,
                        train_parameters,
                    ]
                )
    with mp.Pool(CORES) as p:
        outs = p.map(run_online_dqn_learning_star, jobs)

    for i, out in enumerate(outs):
        model, dis_rs, av_rs, props = out
        _, state_vars, reward_var, phi, cbc, _, _ = jobs[i]
        res0 = pd.DataFrame(
            {
                "state_var": [state_vars] * len(dis_rs),
                "reward_var": [reward_var] * len(dis_rs),
                "phi": [phi] * len(dis_rs),
                "cbc": [cbc] * len(dis_rs),
                "av_r": av_rs,
                "dis_r": dis_rs,
                "prop": props,
            }
        )
        res = pd.concat([res, res0], axis=0, ignore_index=True)
        print(res)
    # res = pd.DataFrame()
    # for phi in phis:
    #     for cbc in cbcs:
    #         for state_vars, reward_var in job_list:

    #             outs = run_online_dqn_learning(
    #                 SEED,
    #                 state_vars,
    #                 reward_var,
    #                 phi,
    #                 cbc,
    #                 model_parameters,
    #                 train_parameters,
    #             )
    #             model, dis_rs, av_rs, props = outs
    #             res0 = pd.DataFrame(
    #                 {
    #                     "state_var": [state_vars] * len(dis_rs),
    #                     "reward_var": [reward_var] * len(dis_rs),
    #                     "phi": [phi] * len(dis_rs),
    #                     "cbc": [cbc] * len(dis_rs),
    #                     "av_r": av_rs,
    #                     "dis_r": dis_rs,
    #                     "prop": props,
    #                 }
    #             )
    #             res = pd.concat([res, res0], axis=0, ignore_index=True)
    #             print(res)
    res.to_csv(os.path.join("../results", "results_online_dqn.csv"))


if __name__ == "__main__":
    # run_exp1() # only for state_var == reward_var
    run_exp_test()
