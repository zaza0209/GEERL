# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:44:29 2024

compare the value of the policies derived by Q function with different variance
"""

import numpy as np
import sys, pickle, platform, os
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
cluster_rl_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(cluster_rl_dir)
from functions.generate_joint_data import *
import functions.utilities as uti
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
root_path = os.getcwd()
print('rootpath', root_path)
#%%
print(sys.argv)
seed = int(sys.argv[1])

sd_cluster_ex_noise = float(sys.argv[2])
sd_white =float(sys.argv[3])

# parameters for evaluating the "optimal" policy
num_weeks_eva = int(sys.argv[4])
folds_eva=int(sys.argv[5])

setting=sys.argv[6]

ctn_state_sd = float(sys.argv[7])
gamma_1=float(sys.argv[8])
gamma_2=float(sys.argv[9])
nthread=int(sys.argv[10])
sharpness=float(sys.argv[11])
individual_action_effect=float(sys.argv[12])
within_team_effect=float(sys.argv[13])

num_teams_eva=int(sys.argv[14])
include_Mit=int(sys.argv[15])
only_states=int(sys.argv[16])
transition_state_type=sys.argv[17]
delete_week_end=int(sys.argv[18])
horizon_eva=int(sys.argv[19])
include_weekend_indicator=int(sys.argv[20])
state_ex_noise=float(sys.argv[21])
Q_sd=float(sys.argv[22])
refit=int(sys.argv[23])

burnin=1000
autoregressive_coef = 0.8
team_size_eva=1
corr_eva=0
gamma = 0.9 
np.random.seed(seed)
corr_type="uncorrelated"
#%%
if setting == "tab4"  or setting =="tab6" or setting=="ctn2":
    setting_name = setting  + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)+'gm2'+str(gamma_2)
elif setting == "tab5":
    setting_name = setting  + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)+'gm2'+str(gamma_2) +'sharp'+str(sharpness)
elif setting == "ctn1":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)
elif setting == "ctn3":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'gm1'+str(gamma_1)
elif setting == "ctn4" or setting=="ctn6" or setting=="ctn61" or setting=="ctn63" or setting =="ctn8" or setting == "ctn41":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) + '/indEff'+str(individual_action_effect)+'/teamEff'+str(within_team_effect)
    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1)+'_gm2'+str(gamma_2) + '/indEff'+str(individual_action_effect)+'/teamEff'+str(within_team_effect)
elif setting == "ctn0":
    setting_name = setting + 'state_noise'+str(ctn_state_sd)
else:
    setting_name = setting

setting_name += "Q_sd"+str(Q_sd)
if setting[:3] == "ctn":
    setting_name += '/sd_cluster_ex_noise'+str(sd_cluster_ex_noise)+'sd_white'+str(sd_white) + "state_ex_noise"+str(state_ex_noise)

def setpath():
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    path_name = 'D:/OneDrive/PhD/ATE/code/ClusterRL/individual_level/value_comparison'+\
        '/results/Setting_'+setting_name+\
                "/burnin"+str(burnin)+'/seed'+str(seed)
        
    print('path_name',path_name)
    sys.stdout.flush()
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)

        
setpath()
print(os.getcwd())
file_name =  "seed_"+str(seed)+'n_test'+str(num_teams_eva)+'n_weeks'+str(num_weeks_eva)+'horizon_eva'+str(horizon_eva)+'corr_eva'+str(corr_eva)+".dat"

if not refit:
    if os.path.exists(file_name):
        exit()
        

def save_data(GEE, file_name):
    with open(file_name, "wb") as f:
        pickle.dump({'q_function_list':GEE.q_function_list,
                     'coef_sd':GEE.coef_sd,
                     'iternum':GEE.iternum,
                     'converge':GEE.converge,
                     'running_time':GEE.run_time,
                     'rho':GEE.dep_params}, f)
        

stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("/nName of Python script:", sys.argv[0])

#%% transition and reward

'''
Possible settings:
    1. varing matching effect: [gamma_1, gamma_2]=[0.3, -0.3], [0.3, 0.3], [0.3, 0]
    2. varing individual action effect: individual_action_effect = [0.3, 0.6]
    3. varing within team correlation: 
        3.1. correlated states: exchangeable state variables (rho=[0.2, 0.8])
        3.2. correlated rewards: exchangeable between individuals (sd_cluster_ex_noise) 
            + autoregressive over time (autoregressive_coef, sd_white_noise) 
            -> varing corr(TD_{ij}, TD_{rl}) = rho for i != r
    4. varing sample size: num of teams (100), team size (10), time horizon (7 days a week, 20 weeks)

'''
    
    
def round_to_nearest(value, possible_values):
    """Rounds a value to the nearest one in the given set possible_values."""
    # possible_values = np.array([4, 6, 8, 12])
    if np.isscalar(value):
        value = np.array([value])
    idx = np.argmin(np.abs(possible_values - value), axis=-1)
    return possible_values[idx].reshape(value.shape)


if setting == "tab0":
    round_Mit=1
    instt=0
    possible_values = np.array([0, 1])
    state_dim=1
    
    def reward_function(Sijt, action, matched_team=None, alpha=10, beta=0.05):
        r_table = {(0,0):2, (0, 1):0, (1, 0):-1, (1, 1):-1}
        r_list=[]
        for s_i,a in zip(Sijt, action):
            if not isinstance(s_i, int):
                s_i = int(s_i)
            a = int(a)
            if (s_i, a) in r_table:
                r_list.append(r_table[(s_i, a)])
            else:
                raise ValueError('wrong states or action')
        return np.hstack(r_list).reshape(-1, 1)

    def transition_function(Sijt, action, matched_team=None, gamma=0.05, delta=0.1):
        ns_table = {(0,0):0, (0, 1):1, (1,0):0, (1,1):0}
        next_s=[]
        for s_i, a in zip(Sijt, action):
            if not isinstance(s_i, int):
                s_i = int(s_i)
            a = int(a)
            if (s_i, a) in ns_table:
                next_s.append(ns_table[(s_i, a)])
            else:
                raise ValueError('wrong states or action')
        return np.hstack(next_s).reshape(-1,1)
    
    def init_state(team_size):
        return np.random.randint(0,2, team_size)  
    Q_true = np.array([[2/(1-gamma), 2*gamma**2/(1-gamma) -gamma],
                       [2*gamma/(1-gamma)-1, 2*gamma/(1-gamma)-1]])
    Q = Q_true + np.random.normal(0, Q_sd, size=(2,2))
    
    def target_policy(St):
        a_list = []
        St = np.atleast_1d(St)
        for s in St:
            max_q = np.max(Q[int(s)])
            a_list.append(np.random.choice(np.where(Q[int(s)]==max_q)[0]))
        return np.hstack(a_list)
        

elif setting == "ctn62":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = 0.25*St[...,0]**2*(2*At-1) + St[..., 0]
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        s_i = Sijt 
        s_j = matched_team_current_states #round_to_nearest(matched_team_current_states, possible_values) 
        team_current_states = team_current_states#round_to_nearest(team_current_states, possible_values)
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


elif setting == "ctn64":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    instt=0
    round_Mit=0
    include_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = 0.25*St[...,0]**(2*At-1) + St[..., 0]
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        s_i = Sijt 
        s_j = matched_team_current_states #round_to_nearest(matched_team_current_states, possible_values) 
        team_current_states = team_current_states#round_to_nearest(team_current_states, possible_values)
        # Individual action effect
        # individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        # competition effect
        competition_effect = gamma_1 * (s_j - s_i)
        # within team effect
        team_effect = within_team_effect * (2.0 * Aijt.reshape(-1,1) - 1.0)*(team_current_states - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + team_effect + epsilon +competition_effect
        next_s_i[next_s_i>13] = 13
        next_s_i[next_s_i<-13] = -13
        return next_s_i
    
    def init_state(team_size):
        return np.random.uniform(-10, 10, team_size)

    
elif setting == "ctn8":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values = np.arange(0, 13)
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = St[...,0]*(2*At-1) + 0.25*St[..., 0]
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        s_i = Sijt 
        s_j = round_to_nearest(matched_team_current_states, possible_values) 
        team_current_states = round_to_nearest(team_current_states, possible_values)
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
    
elif setting == "ctn9":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1  
    only_states=0
    instt=0
    round_Mit=0
    possible_values = None #np.arange(0, 13)
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = St[...,0]*(2*At-1) + 0.25*St[..., 0]
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        s_i = Sijt 
        s_j = matched_team_current_states # round_to_nearest(matched_team_current_states, possible_values) 
        team_current_states = team_current_states#round_to_nearest(team_current_states, possible_values)
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
        return np.random.normal(0, 0.5, team_size)  
elif setting == "ctn10":
    '''
    p=2
    '''
    include_team_effect=1  
    only_states=0
    instt=0
    round_Mit=0
    possible_values = None #np.arange(0, 13)
    def reward_function(St, At, matched_team_current_states, team_current_states, i=0, t=0):
        St = np.atleast_1d(St)
        rewards = 2*St[...,0]+St[...,1] - 0.25*(2.0 * At - 1.0)
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect):
        s1 = 0.75*(2.0 * Aijt.flatten() - 1.0) *Sijt[...,0]+0.25*Sijt[...,1]
        s2 = 0.75*(2.0 * Aijt.flatten() - 1.0) *Sijt[...,1]+0.25*Sijt[...,0]
        
        # Noise
        epsilon_shape = Sijt.shape[:-1] # Add an extra dimension for broadcasting
        s1 += np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        s2 += np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
    
        
        # Calculate next state
        next_s_i = np.concatenate((s1.reshape(-1,1), s2.reshape(-1,1)), axis=-1)
        return next_s_i
    
    def init_state(team_size):
        return np.random.normal(0, 0.5, (team_size, 2))  
if within_team_effect == 0:
    include_team_effect=0
#%% fit the Q model if not fitted
is_onpolicy =0 
if is_onpolicy:
    action_space =[1]
else:
    action_space =[0,1]

#%% generate data
def renew_teams(num_teams, num_weeks, team_size, horizon=1, corr_eva=1, 
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
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i, t+1] 
            
                return cluster_transition
            
            def make_cluster_reward(j):
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
                return cluster_reward_function
            cluster_reward_list = [make_cluster_reward(i) for i in range(num_teams)]
            cluster_transition_list = [make_cluster_transition(i) for i in range(num_teams)]
        elif corr_type == "s_ex":
            states_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            def make_cluster_transition(i):
            
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + states_noise[i] #+ np.random.normal(0, sd_white)
            
                return cluster_transition
            def make_cluster_reward(j):
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
            
            cluster_reward_list = [make_cluster_reward(i) for i in range(num_teams)]
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
                    return reward_function(St, At, matched_team_current_states, team_current_states)# + noise + np.random.normal(0, sd_white)
            
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
                    return reward_function(St, At, matched_team_current_states, team_current_states)# + noise + np.random.normal(0, sd_white)
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, state_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
            
        else:
            def make_cluster_reward():
                def cluster_reward_function(St, At, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(St, At, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
                return cluster_reward_function
        
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, sd_cluster_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
    
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
                            is_training=False, corr_eva =corr_eva,
                            transition_state_type=transition_state_type)
        rollouts, trajectories = simulate_teams(teams, num_weeks_eva, individual_policy=target_policy,
                                      action_space=action_space,
                                      only_states=only_states,
                                      include_Mit=include_Mit,
                                      include_team_effect=include_team_effect,
                                      instt=instt, round_Mit_function=round_Mit_function,
                                      horizon_eva=horizon_eva,
                                      transition_state_type=transition_state_type,
                                      include_weekend_indicator=include_weekend_indicator,
                                      burnin=burnin,
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
#%% if random policy
av_r, dis_r = evaluate_policy(target_policy, nthread=nthread, corr_eva=corr_eva)
sys.stdout.flush()
with open(file_name, "wb") as f:
    pickle.dump({'av_r':av_r, 'dis_r':dis_r}, f)
