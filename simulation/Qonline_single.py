# -*- coding: utf-8 -*-
"""
Single setting: Online Q learning to estimate the optimal policy

@author: test
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, pickle, platform, os
plat = platform.platform()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
cluster_rl_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(cluster_rl_dir)
from functions.generate_joint_data import *
from joblib import Parallel, delayed
import functions.utilities as uti
#%%
print(sys.argv)
seed = int(sys.argv[1])
num_teams = int(sys.argv[2])
team_size = int(sys.argv[3])
sd_cluster_ex_noise = float(sys.argv[4])
sd_white =float(sys.argv[5])

# parameters for evaluating the "optimal" policy
num_weeks_eva = int(sys.argv[6])
folds_eva=int(sys.argv[7])
setting=sys.argv[8]

ctn_state_sd = float(sys.argv[9])
horizon=int(sys.argv[10])
gamma_1=float(sys.argv[11])
gamma_2=float(sys.argv[12])
nthread=int(sys.argv[13])
sharpness=float(sys.argv[14])
individual_action_effect=float(sys.argv[15])
within_team_effect=float(sys.argv[16])
num_teams_eva=int(sys.argv[17])
corr_type=sys.argv[18]
corr_eva=int(sys.argv[19])
num_weeks=int(sys.argv[20])
only_training=int(sys.argv[21])
include_Mit=int(sys.argv[22])
use_replay_buffer=int(sys.argv[23])
early_stopping_criterion=sys.argv[24] # reward, loss
update_target_every=int(sys.argv[25])
early_stopping_patience=int(sys.argv[26])
state_combine_friends=int(sys.argv[27])
history_length=int(sys.argv[28])
transition_state_type=sys.argv[29]
delete_week_end=int(sys.argv[30])
horizon_eva=int(sys.argv[31])
include_weekend_indicator=int(sys.argv[32])
train_corr_eva=int(sys.argv[33])
reward_buffer_len = int(sys.argv[34])
early_stop=float(sys.argv[35])
state_ex_noise=float(sys.argv[36])
hidde_nodes_lens=int(sys.argv[37])
hidden_nodes= [int(sys.argv[i]) for i in range(38, 38+hidde_nodes_lens)]
autoregressive_coef = 0.8
action_space=[0,1]
basis = 'nn'
p = 1
method="online"
gamma=0.9
team_size_eva=1
print('finish params', 'gamma', gamma)
#%%
basis_name = "nn"+'_hidden_nodes'+str(hidden_nodes)+'replay_buffer'+str(use_replay_buffer)

if setting == "tab4"  or setting =="tab6" or setting=="ctn2":
    setting_name = setting  + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)+'gm2'+str(gamma_2)
elif setting == "tab5":
    setting_name = setting  + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)+'gm2'+str(gamma_2) +'sharp'+str(sharpness)
elif setting == "ctn1":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)
elif setting == "ctn3":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)
elif setting == "ctn4" or setting=="ctn6" or setting=="ctn61"  or setting =="ctn8" or setting == "ctn41":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) + '/indEff'+str(individual_action_effect)+'/teamEff'+str(within_team_effect)
elif setting=="ctn63" or setting=="ctn64":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1)+'include_Mit'+str(include_Mit) +'/teamEff'+str(within_team_effect)
elif setting == "ctn5":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1)+'_gm2'+str(gamma_2) + '/indEff'+str(individual_action_effect)+'/teamEff'+str(within_team_effect)
elif setting == "ctn0":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)
else:
    setting_name = setting
if setting != "ctn0":    
    if state_combine_friends:
        setting_name += "/state_combine_friends1"
    
    if transition_state_type == "weekly":
        setting_name += "/weeklyTM"
        if delete_week_end:
            setting_name +="_delete_end"
        elif include_weekend_indicator:
            setting_name += "weekendID"
is_old_path=0
corr_type_name = corr_type
if corr_type != "r_autoex_s_exsubject":
    sd_name = '/sd_cluster_ex_noise'+str(sd_cluster_ex_noise)+'sd_white'+str(sd_white)
else:
    sd_name = '/sd_cluster_ex_noise'+str(sd_cluster_ex_noise)+'sd_white'+str(sd_white) + "state_ex_noise"+str(state_ex_noise)

def setpath():
    if is_old_path:
        if not os.path.exists('results'):
            os.makedirs('results', exist_ok=True)
        path_name = 'results/'+corr_type_name+'/Setting_'+setting_name+\
          '/'+basis_name+'/num_teams_'+str(num_teams)+\
            '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks)+\
                sd_name+'/seed'+str(seed)
    else:
        if not os.path.exists('results'):
            os.makedirs('results', exist_ok=True)
        path_name = 'results/'+corr_type_name+'/Setting_'+setting_name+\
           '/'+basis_name+'/num_teams_'+str(num_teams)+\
            '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks) +'_horizon'+str(horizon)+'/history_length'+str(history_length)+\
                sd_name+\
                '/early_stopping_'+early_stopping_criterion+'_patience'+str(early_stopping_patience)+'/update_targe'+str(update_target_every)+'/train_corr_eva'+str(train_corr_eva)+\
                    '/reward_buffer_len'+str(reward_buffer_len)+'/early_stop'+str(early_stop)+'/seed'+str(seed)
    print('path_name',path_name)
    sys.stdout.flush()
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    
setpath()
q_name = 'q_network_params.pth'
# file_name="seed_"+str(seed)+".dat"
file_name="seed_"+str(seed)+'n_test'+str(num_teams_eva)+'n_weeks'+str(num_weeks_eva)+'horizon_eva'+str(horizon_eva)+'corr_eva'+str(corr_eva)+".dat"

if os.path.exists(file_name):
    exit()


stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "a")
print("/nName of Python script:", sys.argv[0])
sys.stdout.flush()
#%% transition and rewards
def round_to_nearest(value, possible_values):
    """Rounds a value to the nearest one in the given set possible_values."""
    # possible_values = np.array([4, 6, 8, 12])
    if possible_values==None:
        return value
    if np.isscalar(value):
        value = np.array([value])
    idx = np.argmin(np.abs(possible_values - value), axis=-1)
    return possible_values[idx].reshape(value.shape)

if setting == "ctn0":
    include_team_effect=0
    only_states=1
    instt=0
    round_Mit=0
    state_dim=1
    state_space = None
    possible_values = None
    
    def reward_function(St, At, matched_team_current_states=None, team_current_states=None, i=0, t=0, alpha=50, beta=0.05):
        Sijt = np.atleast_1d(St)
        At = np.atleast_1d(At)
        def r_base(s, a):
            return  0.25*s**2 * (2.0 * a - 1.0) + s
        
        r_list = []
        for s_i, a in zip(Sijt, At):
            r_list.append(r_base(s_i,a))
        return np.hstack(r_list).reshape(-1,1)
    
    def transition_function(Sijt, Aijt, matched_team_current_states=None, 
                            team_current_states=None, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect
                            , gamma=0.3, delta=0.1):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        next_s = []
        for s_i, At in zip(Sijt, Aijt):
            epsilon = np.random.normal(0, ctn_state_sd)  
            next_s.append(0.5 *  (2.0 * At - 1.0) *s_i + epsilon)
        return np.hstack(next_s).reshape(-1, 1)
    def init_state(team_size):
        return np.random.normal(0, 0.5, team_size) 
    
elif setting == "ctn4":
    include_team_effect=1
    only_states=0
    instt=0
    round_Mit=0
    possible_values = None 
    state_dim=2
    
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = St[...,0]  *(2*At-1)
        
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        """
        update states at a fixed time t for a whole team
        
        Parameters
        -----------
        Sijt: Sleeping hours of subject j in team i at time t.
            2D ndarray (subjects within a team, number of features).
        Aijt:2D ndarray (subjects within a team, 1) individual action. Aijt=1, send message 1; Aijt=0, send message 2.
        matched_team: the team matched with team i at time t.
        individual_action_effect: individual action effect.
        gamma_1: Influence of the competing team's sleep on the current subject if they belong to different institutions.
        gamma_2: Influence of the competing team's sleep on the current subject if they belong to the same institution.
        optimal_sleep: optimal sleeping hours
        ctn_state_sd: sd of white noise
        """
        
        s_i = Sijt 
        s_j = matched_team_current_states
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        # competition effect
        competition_effect = gamma_1 * (s_j - s_i)
        # within team effect
        team_effect = within_team_effect * (team_current_states - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + team_effect + competition_effect + epsilon
        next_s_i[next_s_i>13] = 13
        next_s_i[next_s_i<0] = 0
        return next_s_i
    
    def init_state(team_size):
        return np.random.uniform(0, 13, team_size)
#%%
def get_training_env():
    if train_corr_eva:
        if corr_type=="r_autoex":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise)
            training_reward_function = uti.autoex_reward(cluster_noise, reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data
            training_transition=transition_function
        elif corr_type=="r_autoexsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, (1, 1 * horizon+1))
            # BUG NOTICE: the reward function for each cluster should not share the same object,
            # which can be caused by defining the cluster reward function with the same name inside the following for
            # loop for all teams
            training_reward_function = uti.autoex_reward(cluster_noise, reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data
            training_transition=transition_function
        elif corr_type == "r_exsubject":
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return reward_function(St, At, matched_team_current_states, team_current_states) + noise + np.random.normal(0, sd_white)
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
    
        elif corr_type == "r_ex":
            # print('corr_type', corr_type)
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, 1)
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                return reward_function(St, At, matched_team_current_states, team_current_states) + cluster_noise + np.random.normal(0, sd_white)
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
        
        elif corr_type == "r_autoex_s_exsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise)
            training_reward_function = uti.autoex_reward(cluster_noise, reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                noise = np.random.normal(0, state_ex_noise)
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
    
        else:
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
        
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
    else:
        if corr_type in ["r_ex", "r_autoex", "r_exsubject", "uncorrelated"]:
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return reward_function(St, At, matched_team_current_states, team_current_states) + noise  + np.random.normal(0, sd_white)
            
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
        elif corr_type == "r_autoex":
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return reward_function(St, At, matched_team_current_states, team_current_states) + noise  + np.random.normal(0, sd_white/np.sqrt(1-autoregressive_coef**2))
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
        elif corr_type == "r_autoex_s_exsubject":
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return reward_function(St, At, matched_team_current_states, team_current_states) + noise  + np.random.normal(0, sd_white/np.sqrt(1-autoregressive_coef**2))
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                noise = np.random.normal(0, state_ex_noise)
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
        else:
            def training_reward_function(St, At, matched_team_current_states=None, team_current_states=None, t=0, i=None):
                return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
        
            def training_transition(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, t=0):
                noise = np.random.normal(0, sd_cluster_ex_noise)
                return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise

    return training_transition, training_reward_function
#%% training params
# Initialize parameters
epsilon = 1.0  # initial epsilon
min_epsilon = 0.01  # minimum value of epsilon
decay_rate = 0.001  # decay rate
#%%
# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes):
        super(QNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_nodes[0]), nn.ReLU()]
        for i in range(len(hidden_nodes) - 1):
            layers.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes[-1], action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

# Initialize Q-network and target network
state_size = p  # dimension of your state
action_size = 2  # number of possible actions
q_network = QNetwork(state_size, action_size, hidden_nodes)

# Set a tolerance for the loss 
loss_tol = 1e-5
old_loss = np.inf
#%%
if os.path.exists('q_network_params.pth'):
    print('load q network')
    q_network.load_state_dict(torch.load('q_network_params.pth'))
else:
    target_network = QNetwork(state_size, action_size, hidden_nodes)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    # Training loop
    for i_episode in range(5000):  # number of episodes
        for i in range(num_teams):
            episode_reward=0
            state = np.random.normal(0, 0.5, 1)  # initialize state
            training_transition, training_reward_function = get_training_env()
            for t in range(horizon):  # maximum steps per episode
                state_tensor = torch.FloatTensor(state)
    
                # Select action using epsilon-greedy policy
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice([0, 1])  # explore
                else:
                    with torch.no_grad():
                        action = q_network(state_tensor).argmax().item()  # exploit
    
                # Take action, observe reward and next state
                next_state = training_transition(state, action, t)
                reward = training_reward_function(state, action, t=t, i=0)
                episode_reward+= reward
                # Compute Q-value of next state
                next_state_tensor = torch.FloatTensor(next_state)
                with torch.no_grad():
                    next_q_values = target_network(next_state_tensor)
    
                # Compute target Q-value
                reward_tensor = torch.tensor(reward, dtype=torch.float32)  # convert reward to Tensor
                target_q_value = reward_tensor + gamma * next_q_values.max()
    
                # Compute current Q-value
                q_value = q_network(state_tensor).gather(0, torch.tensor(action_space.index(action), dtype=torch.int64).unsqueeze(-1)).squeeze(-1)
    
                # Compute loss
                loss = (q_value - target_q_value).pow(2)
    
                # Check if the loss is less than the tolerance
                # if abs(old_loss - loss.item()) < loss_tol:
                #     print("Converged at episode: ", i_episode)
                #     break
    
                old_loss = loss.item()
    
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                # Update state
                state = next_state.flatten()

        # if the inner loop did not break, continue with the next episode
        print('====== i_episode', i_episode, 'abs(old_loss - loss.item())',abs(old_loss - loss.item()), 'reward', episode_reward)
        sys.stdout.flush()
        # if abs(old_loss - loss.item()) >= loss_tol:
        if i_episode % update_target_every == 0:
            target_network.load_state_dict(q_network.state_dict())
        epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * i_episode))
        # else:
            # break

    #%%

    torch.save(q_network.state_dict(), 'q_network_params.pth')
if only_training:
    exit()
#%%
def optimal_policy(St):
    state_tensor = torch.FloatTensor(St)
    with torch.no_grad():
        action = q_network(state_tensor).argmax().item()  
    return action

burnin=1000
def renew_teams(num_teams, num_weeks, team_size, horizon, corr_eva=1, 
                is_indi=True, is_training=True, is_record_all=False, transition_state_type="weekly"):
    if corr_eva:
        if corr_type=="r_autoex":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            # print(cluster_noise)
            # BUG NOTICE: the reward function for each cluster should not share the same object,
            # which can be caused by defining the cluster reward function with the same name inside the following for
            # loop for all teams
            cluster_reward_list = [uti.autoex_reward(cluster_noise[i], reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]
        elif corr_type=="r_autoexsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, (num_teams, num_weeks * horizon+1))
            # BUG NOTICE: the reward function for each cluster should not share the same object,
            # which can be caused by defining the cluster reward function with the same name inside the following for
            # loop for all teams
            cluster_reward_list = [uti.autoex_reward(cluster_noise[i], reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]

        elif corr_type == "r_exsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, (num_teams, num_weeks*horizon))
            def make_cluster_reward(j):
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states) + cluster_noise[j, t] + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            cluster_reward_list = [make_cluster_reward(i) for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]
        elif corr_type == "r_ex":
            # print('corr_type', corr_type)
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            def make_cluster_reward(j):
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states) + cluster_noise[j] + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            cluster_reward_list = [make_cluster_reward(i) for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]
        elif corr_type == "s_exsubject":
            states_noise = np.random.normal(0, sd_cluster_ex_noise, (num_teams, max(num_weeks * horizon+1, burnin+1)))
            def make_cluster_transition(i):
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i, t+1] + np.random.normal(0, sd_white)
            
                return cluster_transition
            cluster_reward_list = [reward_function for i in range(num_teams)]
            cluster_transition_list = [make_cluster_transition(i) for i in range(num_teams)]
        elif corr_type == "s_ex":
            states_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            def make_cluster_transition(i):
            
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i] + np.random.normal(0, sd_white)
            
                return cluster_transition
            cluster_reward_list = [reward_function for i in range(num_teams)]
            cluster_transition_list = [make_cluster_transition(i) for i in range(num_teams)]
        elif corr_type=="rs_autoexsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, (num_teams, max(num_weeks * horizon+1, burnin+1)))
            # BUG NOTICE: the reward function for each cluster should not share the same object,
            # which can be caused by defining the cluster reward function with the same name inside the following for
            # loop for all teams
            cluster_reward_list = [uti.autoex_reward(cluster_noise[i], reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data for i in range(num_teams)]
            states_noise = np.random.normal(0, sd_cluster_ex_noise, (num_teams, max(num_weeks * horizon+1, burnin+1)))
            def make_cluster_transition(i):
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i, t+1]
            
                return cluster_transition
            cluster_reward_list = [reward_function for i in range(num_teams)]
            cluster_transition_list = [make_cluster_transition(i) for i in range(num_teams)]
        elif corr_type=="r_autoex_s_exsubject":
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            # print(cluster_noise)
            # BUG NOTICE: the reward function for each cluster should not share the same object,
            # which can be caused by defining the cluster reward function with the same name inside the following for
            # loop for all teams
            cluster_reward_list = [uti.autoex_reward(cluster_noise[i], reward_function,
                                                autoregressive_coef, std_dev=sd_white).generate_autoregressive_data for i in range(num_teams)]
            # print("r_autoex_s_exsubject", 'state_ex_noise', state_ex_noise)
            states_noise = np.random.normal(0, state_ex_noise, (num_teams, max(num_weeks * horizon+1, burnin+1)))
            def make_cluster_transition(i):
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i, t+1]
                return cluster_transition
            cluster_transition_list = [make_cluster_transition(i) for i in range(num_teams)]
        elif corr_type == "uncorrelated":
            cluster_reward_list = [reward_function for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]
        else:
            raise ValueError('Invalid corr_type')
        
    else:
        if corr_type in ["r_ex", "r_autoex", "r_exsubject", "uncorrelated"]:
            def make_cluster_reward():
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(St, At, matched_team_current_states, team_current_states) #+ noise + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
                return cluster_transition
        elif corr_type == "rs_autoexsubject":
            def make_cluster_reward():
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(St, At, matched_team_current_states, team_current_states) #+ noise + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, sd_cluster_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
        elif corr_type=="r_autoex_s_exsubject":
            def make_cluster_reward():
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(St, At, matched_team_current_states, team_current_states) #+ noise + np.random.normal(0, sd_white)
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, state_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
            
        else:
            def make_cluster_reward():
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states)# + np.random.normal(0, sd_white)
                return cluster_reward_function
        
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, sd_cluster_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
        # def make_cluster_transition():
        #     def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
        #         return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
        #     return cluster_transition
        # def make_cluster_reward():
        #     def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
        #         return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
        #     return cluster_reward_function
    
        cluster_reward_list = [make_cluster_reward() for _ in range(num_teams)]
        cluster_transition_list = [make_cluster_transition() for _ in range(num_teams)]
        
    def make_init_state(i):
        if corr_eva:
            def team_init_state(team_size):
                if corr_type in ['r_ex', "r_exsubject", 'r_autoex', 'r_autoexsubject', "uncorrelated"]:
                    noise=0
                elif corr_type in ["s_exsubject", "rs_autoexsubject", "r_autoex_s_exsubject"]:
                    noise=states_noise[i, 0] if corr_eva else 0
                elif corr_type == "s_ex":
                    noise = states_noise[i]
                else:
                    raise ValueError('Invalid corr_type')
                if instt:
                    instt_tmp = 0 if i < num_teams/2 else 1
                    return np.concatenate(((init_state(team_size)+noise).reshape((-1,1)), np.repeat(instt_tmp, team_size).reshape((-1,1))), axis=-1)
                else:
                    return init_state(team_size)+noise
        else:
            def team_init_state(team_size):
                if corr_type in ['r_ex', "r_exsubject", 'r_autoex', 'r_autoexsubject', "uncorrelated"]:
                    noise=0
                elif corr_type == "r_autoex_s_exsubject":
                    noise = np.random.normal(0, state_ex_noise)
                else:
                    noise = np.random.normal(0, sd_cluster_ex_noise) 
                if instt:
                    instt_tmp = 0 if i < num_teams/2 else 1
                    return np.concatenate(((init_state(team_size)+noise).reshape((-1,1)), np.repeat(instt_tmp, team_size).reshape((-1,1))), axis=-1)
                else:
                    return init_state(team_size)+noise
        return team_init_state
    team_init_statefun_list = [make_init_state(i) for i in range(num_teams)]
    

    
    teams = [Team(i, team_size, 
                reward_function=cluster_reward_list[i],
                init_state=team_init_statefun_list[i],
                transition_function=cluster_transition_list[i],
                is_indi=is_indi, is_training=is_training,
                is_record_all=is_record_all,
                transition_state_type=transition_state_type,
                # record_history=record_history,
                # return_trajectories=return_trajectories,
                instt=instt, horizon=horizon) for i in range(num_teams)]
    return teams
def evaluate_policy(target_policy, gamma=0.9, nthread=1, corr_eva=corr_eva, horizon_eva=horizon_eva):
    if round_Mit:
        def round_Mit_function(states):
            return round_to_nearest(states, possible_values)
    else:
        round_Mit_function = None
            
    def run_one(fold):
                
        teams = renew_teams(num_teams_eva, num_weeks_eva, team_size=team_size_eva, horizon = horizon_eva,
                            is_training=False, corr_eva = corr_eva,
                            transition_state_type=transition_state_type)
        rollouts, trajectories = simulate_teams(teams, num_weeks_eva, individual_policy=target_policy,
                                      action_space=action_space,
                                      only_states=only_states,
                                      collect_for_training=0,
                                      include_Mit=include_Mit,
                                      include_team_effect=include_team_effect,
                                      state_combine_friends=state_combine_friends,
                                      instt=instt, round_Mit_function=round_Mit_function,
                                      history_length=history_length,
                                      horizon_eva=horizon_eva,
                                      transition_state_type=transition_state_type,
                                      include_weekend_indicator=include_weekend_indicator,
                                      burnin=0,
                                        )
        if 'av_rewards' in rollouts[0]:
            av_rs=np.array([rollout['av_rewards'] for rollout in rollouts])
            dis_rs = np.array([rollout['dis_rewards'] for rollout in rollouts])
            av_r=np.mean(av_rs)
            dis_r= np.mean(dis_rs)
        else:
            rewards = np.concatenate([rollout['rewards'] for rollout in rollouts])
            av_r = np.mean(rewards)
            dis_r = 0
            for t in range(rewards.shape[1]):
                dis_r += gamma**t * np.mean(rewards[:, t])
        # print(av_r,dis_r)
        return av_r, dis_r
    if nthread>1:
        res = Parallel(n_jobs=nthread)(delayed(run_one)(fold) for fold in range(folds_eva))
    else:
        res = []
        for fold in range(folds_eva):
            tmp = run_one(fold)
            res.append(tmp)
    av_r = np.mean([r[0] for r in res])
    dis_r = np.mean([r[1] for r in res])
    print('av_r', av_r, 'dis_r', dis_r)
    return av_r, dis_r


av_r, dis_r = evaluate_policy(target_policy=optimal_policy,nthread=nthread)
sys.stdout.flush()
#%%
with open(file_name, "wb") as f:
    pickle.dump({'av_r':av_r, 'dis_r':dis_r}, f)