# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:57:21 2023

reward autoregressive + exchangeable correlation; CV learning

"""

import numpy as np
import sys, pickle, platform, os
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
cluster_rl_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(cluster_rl_dir)

import functions.GEE_Q as GEE_Q
import functions.GEE_Q_old as GEE_Q_old
from functions.generate_joint_data import simulate_teams, Team
import functions.utilities as uti
import functions.utilities_old as uti_old
import functions.cov_struct as cov_structs
from functions.offlineFQI import *
from collections import namedtuple
# import math
import gc
import json
# import itertools
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from copy import deepcopy
root_path = os.getcwd()
#%%
print(sys.argv)
seed = int(sys.argv[1])
num_teams = int(sys.argv[2])
team_size = int(sys.argv[3])
num_weeks = int(sys.argv[4])
method = sys.argv[5]
sd_cluster_ex_noise = float(sys.argv[6])
sd_white =float(sys.argv[7])

# parameters for evaluating the "optimal" policy
num_weeks_eva = int(sys.argv[8])
folds_eva=int(sys.argv[9])
basis = sys.argv[10]
cv_criterion = sys.argv[11]
setting=sys.argv[12]

ctn_state_sd = float(sys.argv[13])
horizon=int(sys.argv[14])
gamma_1=float(sys.argv[15])
gamma_2=float(sys.argv[16])
nthread=int(sys.argv[17])
sharpness=float(sys.argv[18])
individual_action_effect=float(sys.argv[19])
within_team_effect=float(sys.argv[20])
only_cv = int(sys.argv[21])
num_teams_eva=int(sys.argv[22])
corr_type=sys.argv[23]
num_batches=int(sys.argv[24])
cv_loss=sys.argv[25]
accelerate_method=sys.argv[26]
new_cov=int(sys.argv[27])
new_GEE=int(sys.argv[28])
new_uti=int(sys.argv[29])
cv_seed=sys.argv[30]
include_Mit=int(sys.argv[31])
only_states=int(sys.argv[32])
state_combine_friends=int(sys.argv[33])
history_length=int(sys.argv[34])
transition_state_type=sys.argv[35]
delete_week_end=int(sys.argv[36])
horizon_eva=int(sys.argv[37])
include_weekend_indicator=int(sys.argv[38])
cv_in_training=int(sys.argv[39])
state_ex_noise=float(sys.argv[40])
optimal_GEE=int(sys.argv[41])
combine_actions=int(sys.argv[42])
refit=int(sys.argv[43])

burnin=1000
autoregressive_coef = 0.8
hidden_nodes=[64]
update_target_every=10
team_size_eva=1
corr_eva=0
gamma = 0.9 
np.random.seed(seed)


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
elif setting=="ctn63" or  setting=="ctn64":
    if method != "random":
        if only_states:
            setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) +'only_state'+str(only_states) +'/teamEff'+str(within_team_effect)
        else:
            setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) +'include_Mit'+str(include_Mit)+'/teamEff'+str(within_team_effect)

    else:
        setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) +'/teamEff'+str(within_team_effect)
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
    
if basis != "one_hot":
    basis_name =  basis+"cv_"+str(cv_criterion)
    if cv_criterion in ["min", "1se"]:
        basis_name += '/loss'+cv_loss
        if cv_in_training:
            basis_name += "_in_training"
    else:
        basis_name += "/lossNone"
else:
    basis_name = basis
    
if optimal_GEE:
    method_name = method+"_optimalGEE"
else:
    method_name = method
    
if accelerate_method == "split_clusters":
    batch_name = str(num_batches) + accelerate_method
elif accelerate_method == "batch_processing":
    batch_name = str(num_batches)
else:
    raise ValueError("Invalid accelerate_method")

is_old_path=0
if new_cov==0 or new_GEE==0 or new_uti==0:
    corr_type_name = "new_cov"+str(new_cov)+"new_GEE"+str(new_GEE) +'new_uti'+str(new_uti)+corr_type
else:
    corr_type_name = corr_type
if corr_type != "r_autoex_s_exsubject":
    sd_name = '/sd_cluster_ex_noise'+str(sd_cluster_ex_noise)+'sd_white'+str(sd_white)
else:
    sd_name = '/sd_cluster_ex_noise'+str(sd_cluster_ex_noise)+'sd_white'+str(sd_white) + "state_ex_noise"+str(state_ex_noise)

def setpath(basis_name):
    if is_old_path:
        if not os.path.exists('results'):
            os.makedirs('results', exist_ok=True)
        path_name = 'results/'+corr_type_name+'/Setting_'+setting_name+\
          '/'+ method +'/'+basis_name+'/num_teams_'+str(num_teams)+\
            '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks)+\
                sd_name+'/seed'+str(seed)
    else:
        if not os.path.exists('results'):
            os.makedirs('results', exist_ok=True)
        if method in ["random", "action=1", "action=0"]:
            def generate_random_setting_name():
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
                elif setting=="ctn63" or  setting=="ctn64":
                    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1) +'/teamEff'+str(within_team_effect)
                elif setting == "ctn5":
                    setting_name = setting + 'state_noise'+str(ctn_state_sd)+'/gm1'+str(gamma_1)+'_gm2'+str(gamma_2) + '/indEff'+str(individual_action_effect)+'/teamEff'+str(within_team_effect)
                elif setting == "ctn0":
                    setting_name = setting + 'state_noise'+str(ctn_state_sd)
                else:
                    setting_name = setting
                if state_combine_friends:
                    setting_name += "/state_combine_friends1"
                return setting_name
            path_name = 'results/'+corr_type_name+'/Setting_'+generate_random_setting_name()+\
              '/'+ method +'/team_size_'+str(team_size)+'/horizon'+str(horizon)+\
                    sd_name+'/seed'+str(seed)
        elif method == "nn":
            basis_name_nn = "offline_nn"+'_hidden_nodes'+str(hidden_nodes)
            path_name = 'results/'+corr_type_name+'/Setting_'+setting_name+\
               '/'+basis_name_nn+'/num_teams_'+str(num_teams)+\
                '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks) +'_horizon'+str(horizon)+'/history_length'+str(history_length)+\
                    sd_name+\
                    '/update_targe'+str(update_target_every)+"/burnin"+str(burnin)+'/seed'+str(seed)
        else:
            path_name = 'results/'+corr_type_name+'/Setting_'+setting_name+\
              '/'+ method_name +'/num_teams_'+str(num_teams)+\
                '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks) +'_horizon'+str(horizon)+'/history_length'+str(history_length)+'/batches'+batch_name+\
                    sd_name+\
                        "/burnin"+str(burnin)+'/seed'+str(seed)+'/'+basis_name
            
    # print('path_name',path_name)
    sys.stdout.flush()
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    print(os.getcwd())
    sys.stdout.flush()

def find_cv_file():
    path_name =root_path+'/results/'+corr_type_name+'/Setting_'+setting_name+\
      '/'+ method_name +'/num_teams_'+str(num_teams)+\
        '/team_size_'+str(team_size)+'/num_weeks_'+ str(num_weeks) +'_horizon'+str(horizon)+'/history_length'+str(history_length)+'/batches'+batch_name+\
            sd_name+\
                "/burnin"+str(burnin)+'/seed'+str(seed)
    return path_name
    
        
setpath(basis_name)

if cv_seed == "None":
    file_name =  "seed_"+str(seed)+'n_test'+str(num_teams_eva)+'n_weeks'+str(num_weeks_eva)+'horizon_eva'+str(horizon_eva)+'corr_eva'+str(corr_eva)+".dat"
else:
    file_name =  "seed_"+str(seed)+'n_test'+str(num_teams_eva)+'corr_eva'+str(corr_eva)+'cv_seed'+cv_seed+".dat"

qmodel_file = 'qmodel.dat' if cv_seed == "None" else 'qmodel_cv_seed'+cv_seed+'.dat'

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
        
def data_save_path():
    sub_dirs = ['data',corr_type, 'Setting_' + setting_name,'num_teams_'+str(num_teams),
                'team_size_'+str(team_size),'num_weeks_'+ str(num_weeks)+'_horizon'+str(horizon),
                sd_name[1:],
               "seed"+str(seed)]
    data_path = root_path + "/" + "/".join(sub_dirs)
    return data_path

stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("/nName of Python script:", sys.argv[0])
print('num_teams', num_teams, 'team_size', team_size, 'num_weeks', num_weeks, 'sd_cluster_ex_noise',sd_cluster_ex_noise, 'sd_white', sd_white, 'method', method)

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



if setting == "tab1":
    round_Mit=1
    instt=0
    possible_values = np.array([0, 1])
    state_dim=1
    # state0 = np.random.randint(0, 2, num_teams)
    
    
    def reward_function(Sijt, action, matched_team, alpha=10, beta=0.05):
        r_table = {(0,0):1, (0, 1):0, (1, 0):0, (1, 1):1}
        s_j = np.mean(matched_team.states, axis=0)
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        if not isinstance(s_j, int):
            s_j= int(s_j)
        r_list=[]
        for s_i in Sijt:
            if not isinstance(s_i, int):
                s_i = int(s_i)
            if (s_i, s_j) in r_table:
                r_list.append(r_table[(s_i, s_j)])
            else:
                raise ValueError('wrong states')
        return np.hstack(r_list).reshape(-1, 1)

    def transition_function(Sijt, action, matched_team, gamma=0.05, delta=0.1):
        s_j = np.mean(matched_team.states, axis=0)
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        if not isinstance(s_j, int):
            s_j= int(s_j)
        next_s=[]
        
        for s_i in Sijt:
            if not isinstance(s_i, int):
                s_i = int(s_i)
            if s_j != s_i:
                if s_i == 0:
                    next_s.append(1) 
                else:
                    next_s.append(0)
            if s_j == s_i:
                next_s.append(s_i)
        return np.hstack(next_s).reshape(-1,1)

elif setting == "tab2":
    include_team_effect=0
    round_Mit=1
    instt=0
    possible_values = np.array([0, 1])
    state_dim=1
    # state0 = np.random.randint(0, 2, num_teams)
    
    
    def reward_function(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, i=0, t=0, alpha=50, beta=0.05):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        def r_base(s, a):
            return  0.25*s**2 * (2.0 * a - 1.0) + s
        
        r_list = []
        # print('Sijt', Sijt, 'shape', Sijt.shape)
        # print(Aijt, Aijt.shape)
        for s_i, a in zip(Sijt, Aijt):
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
        for s_i, Aijt in zip(Sijt, Aijt):
            if not isinstance(s_i, int):
                s_i = int(s_i)
            if not isinstance(Aijt, int):
                Aijt = int(Aijt)
            if Aijt != s_i:
                if Aijt == 0:
                    next_s.append(1) 
                else:
                    next_s.append(0)
            if Aijt == s_i:
                next_s.append(s_i)
        return np.hstack(next_s).reshape(-1, 1)
    def init_state(team_size):
        return np.random.randint(0,2, team_size)  
    
elif setting =="tab4":
    instt=1
    round_Mit=1
    possible_values = np.arange(0, 13)#np.array([4, 8, 12])
    state_dim=2
    instt_indicator = np.concatenate((np.repeat([0], int(num_teams/2)),np.repeat([1], num_teams - int(num_teams/2)))).reshape((-1,1))
    # instt_indicator_test= np.concatenate((np.repeat([0], int(n_test/2)),np.repeat([1], n_test - int(n_test/2)))).reshape((-1,1))
        
    # s0 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([0], 13).reshape((-1,1))), axis=1)
    # s1 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([1], 13).reshape((-1,1))), axis=1)
    state0 = np.random.randint(0, 13, num_teams)
    state0 = np.concatenate((state0.reshape((-1,1)), instt_indicator), axis=-1)
    state0[:, 0] = np.concatenate([round_to_nearest(val,  possible_values) for val in state0[:, 0]])
    
    def reward_function(Sijt, actions, matched_team, optimal_sleep=10, max_reward=20000, sharpness=1):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        actions: Actions taken (not used in this function, but presumably relevant elsewhere).
        matched_team: The team matched with (not used in this function, but presumably relevant elsewhere).
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        sharpness: The sharpness parameter determines how rapidly the reward decreases as the sleep deviates from the optimal value.
        """
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]
        return rewards if rewards.size > 1 else rewards.item()
    
    
    def transition_function(Sijt, Aijt, matched_team, ctn_state_sd = ctn_state_sd,
                            optimal_sleep=10, individual_action_effect=0.3,
                            gamma_1=gamma_1, gamma_2=gamma_2):
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
        Mit = np.mean(matched_team.states, axis=0)
        
        gamma_list = [gamma_1, gamma_2]
        # Split the feature and observation dimensions
        s_i, instt_i = Sijt[..., :-1], Sijt[..., -1:]
        s_j, instt_j = Mit[..., :-1], Mit[..., -1:]
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        
        # Matching effect
        matching_mask = np.equal(instt_i, instt_j[..., np.newaxis,:])
        # Convert boolean mask to integer indices (0 for False, 1 for True)
        matching_indices = matching_mask.astype(int)
        
        # Use integer indices to get gamma values from gamma_array
        gamma_array = np.array(gamma_list)
        gamma_values = gamma_array[matching_indices]

        # gamma_values = np.array([gamma_list[int(i)] for i in np.equal(instt_i_expanded, instt_j_expanded)])
        competition_effect = gamma_values * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + competition_effect + epsilon
        
        # Ensure sleep remains within realistic bounds
        next_s_i = round_to_nearest(next_s_i, possible_values)
        
        # Concatenate the last dimension back
        next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_sleep_i
    
elif setting =="tab5":
    instt=1
    round_Mit=1
    possible_values = np.arange(0, 13)#np.array([4, 8, 12])
    state_dim=2
    instt_indicator = np.concatenate((np.repeat([0], int(num_teams/2)),np.repeat([1], num_teams - int(num_teams/2)))).reshape((-1,1))
    # instt_indicator_test= np.concatenate((np.repeat([0], int(n_test/2)),np.repeat([1], n_test - int(n_test/2)))).reshape((-1,1))
        
    # s0 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([0], 13).reshape((-1,1))), axis=1)
    # s1 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([1], 13).reshape((-1,1))), axis=1)
    state0 = np.random.randint(0, 13, num_teams)
    state0 = np.concatenate((state0.reshape((-1,1)), instt_indicator), axis=-1)
    state0[:, 0] = np.concatenate([round_to_nearest(val,  possible_values) for val in state0[:, 0]])
    
    def reward_function(Sijt, actions, matched_team, optimal_sleep=10, max_reward=20000, sharpness=sharpness):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        actions: Actions taken (not used in this function, but presumably relevant elsewhere).
        matched_team: The team matched with (not used in this function, but presumably relevant elsewhere).
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        sharpness: The sharpness parameter determines how rapidly the reward decreases as the sleep deviates from the optimal value.
        """
        Sijt = np.atleast_1d(Sijt)
        rewards = max_reward * np.exp(-0.5 * ((Sijt[...,0] - optimal_sleep) / sharpness)**2)
        return rewards if rewards.size > 1 else rewards.item()
    
    
    def transition_function(Sijt, Aijt, matched_team, ctn_state_sd = ctn_state_sd,
                            optimal_sleep=10, individual_action_effect=0.3,
                            gamma_1=gamma_1, gamma_2=gamma_2):
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
        Mit = np.mean(matched_team.states, axis=0)
        
        gamma_list = [gamma_1, gamma_2]
        # Split the feature and observation dimensions
        s_i, instt_i = Sijt[..., :-1], Sijt[..., -1:]
        s_j, instt_j = Mit[..., :-1], Mit[..., -1:]
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        
        # Matching effect
        matching_mask = np.equal(instt_i, instt_j[..., np.newaxis,:])
        # Convert boolean mask to integer indices (0 for False, 1 for True)
        matching_indices = matching_mask.astype(int)
        
        # Use integer indices to get gamma values from gamma_array
        gamma_array = np.array(gamma_list)
        gamma_values = gamma_array[matching_indices]

        # gamma_values = np.array([gamma_list[int(i)] for i in np.equal(instt_i_expanded, instt_j_expanded)])
        competition_effect = gamma_values * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + competition_effect + epsilon
        
        # Ensure sleep remains within realistic bounds
        next_s_i = round_to_nearest(next_s_i, possible_values)
        
        # Concatenate the last dimension back
        next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_sleep_i
    
elif setting == "tab6":
    instt=1
    round_Mit=1
    possible_values = np.arange(0, 13)#np.array([4, 8, 12])
    state_dim=2
    instt_indicator = np.concatenate((np.repeat([0], int(num_teams/2)),np.repeat([1], num_teams - int(num_teams/2)))).reshape((-1,1))
    
    def reward_function(Sijt, actions, matched_team, optimal_sleep=10, max_reward=20000):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        steepness: Controls how steep the sigmoid curve is.
        """
        Sijt = np.atleast_1d(Sijt)
        rewards = np.abs(Sijt[...,0] - optimal_sleep)
        return rewards if rewards.size > 1 else rewards.item()

    def transition_function(Sijt, Aijt, matched_team, ctn_state_sd = ctn_state_sd,
                            optimal_sleep=10, individual_action_effect=0.3,
                            gamma_1=gamma_1, gamma_2=gamma_2):
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
        Mit = np.mean(matched_team.states, axis=0)
        
        gamma_list = [gamma_1, gamma_2]
        # Split the feature and observation dimensions
        s_i, instt_i = Sijt[..., :-1], Sijt[..., -1:]
        s_j, instt_j = Mit[..., :-1], Mit[..., -1:]
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        
        # Matching effect
        matching_mask = np.equal(instt_i, instt_j[..., np.newaxis,:])
        # Convert boolean mask to integer indices (0 for False, 1 for True)
        matching_indices = matching_mask.astype(int)
        
        # Use integer indices to get gamma values from gamma_array
        gamma_array = np.array(gamma_list)
        gamma_values = gamma_array[matching_indices]

        # gamma_values = np.array([gamma_list[int(i)] for i in np.equal(instt_i_expanded, instt_j_expanded)])
        competition_effect = gamma_values * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + competition_effect + epsilon
        
        # Ensure sleep remains within realistic bounds
        next_s_i = round_to_nearest(next_s_i, possible_values)
        
        # Concatenate the last dimension back
        next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_sleep_i

elif setting == "tab7":
    instt=0
    round_Mit=1
    possible_values = np.arange(0, 13)#np.array([4, 8, 12])
    state_dim=2
    # instt_indicator_test= np.concatenate((np.repeat([0], int(n_test/2)),np.repeat([1], n_test - int(n_test/2)))).reshape((-1,1))
        
    # s0 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([0], 13).reshape((-1,1))), axis=1)
    # s1 = np.concatenate((np.arange(0, 13).reshape((-1,1)), np.repeat([1], 13).reshape((-1,1))), axis=1)
    # state0 = np.random.randint(0, 13, num_teams)
    # state0 = np.concatenate((state0.reshape((-1,1)), instt_indicator), axis=-1)
    # state0[:, 0] = np.concatenate([round_to_nearest(val,  possible_values) for val in state0[:, 0]])
    
    def reward_function(Sijt, actions, matched_team, optimal_sleep=10, max_reward=20000, steepness=1):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        steepness: Controls how steep the sigmoid curve is.
        """
        Sijt = np.atleast_1d(Sijt)
        # Sigmoid function centered at optimal_sleep
        # rewards = max_reward / (1 + np.exp(-steepness * (Sijt - optimal_sleep)))
        rewards = Sijt
        return rewards if rewards.size > 1 else rewards.item()

    def transition_function(Sijt, Aijt, matched_team, ctn_state_sd = ctn_state_sd,
                            optimal_sleep=10, individual_action_effect=0.3,
                            gamma_1=0.3):
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
        s_j = np.mean(matched_team.states, axis=0)
        
        # Split the feature and observation dimensions
        s_i = Sijt[...,0]
        if round_Mit:
            s_j = round_to_nearest(s_j, possible_values)
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i).reshape(-1,1)

        # gamma_values = np.array([gamma_list[int(i)] for i in np.equal(instt_i_expanded, instt_j_expanded)])
        competition_effect = gamma_1 * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect.flatten() + competition_effect + epsilon
        # Ensure sleep remains within realistic bounds
        next_s_i = round_to_nearest(next_s_i.reshape(-1,1), possible_values)
        
        # Concatenate the last dimension back
        # next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_s_i
    
elif setting == "tab8":
    instt=0
    possible_values = np.array([-10, 10])#np.arange(0, 13)#np.array([4, 8, 12])
    round_Mit=1
    state_dim=1
    def reward_function(Sijt, action, matched_team, alpha=50, beta=0.05):
        s_j = np.mean(matched_team.states, axis=0)
        def r_base(s):
            return s # np.abs(s-s_j) 
        
        r_list = []
        for s_i in Sijt:
            r_list.append(r_base(s_i))
        return np.hstack(r_list).reshape(-1,1)

    def transition_function(Sijt, action, matched_team, gamma=0.3, delta=0.1):
        """
    
        Parameters
        ----------
        gamma : gamma determines the degree to which team i's state is influenced by its interaction with team j.
            The default is 0.05.
        delta : damping factor. The default is 0.1. Ensure there is a stationary point in the state trajectories.
    
        """
        s_j = np.mean(matched_team.states, axis=0)
        s_j = round_to_nearest(s_j, possible_values)
        next_s = []
        for s_i in Sijt:
            epsilon = np.random.normal(0, ctn_state_sd)  
            n_s = round_to_nearest(s_i + gamma * (s_j - s_i) -delta * s_i + epsilon, possible_values)
            next_s.append(n_s)
        return np.hstack(next_s).reshape(-1, 1)
    
elif setting == "ctn0":
    include_team_effect=0
    instt=0
    round_Mit=0
    state_dim=1
    state_space = None
    possible_values = None
    
    def reward_function(Sijt, Aijt, matched_team_current_states=None, team_current_states=None,
                        i=0, t=0, alpha=50, beta=0.05, week=None,
                         competition_indicator = None,
                         next_week = None):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        def r_base(s, a):
            return  0.25*s**2 * (2.0 * a - 1.0) + s
        
        r_list = []
        # print('Sijt', Sijt, 'shape', Sijt.shape)
        # print(Aijt, Aijt.shape)
        for s_i, a in zip(Sijt, Aijt):
            r_list.append(r_base(s_i,a))
        return np.hstack(r_list).reshape(-1,1)
    
    def transition_function(Sijt, Aijt, matched_team_current_states=None, 
                            team_current_states=None, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect
                            , gamma=0.3, delta=0.1, week=None,
                             competition_indicator = None,
                             next_week = None):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        next_s = []
        for s_i, Aijt in zip(Sijt, Aijt):
            epsilon = np.random.normal(0, ctn_state_sd)  
            next_s.append(0.5 *  (2.0 * Aijt - 1.0) *s_i + epsilon)
        return np.hstack(next_s).reshape(-1, 1)
    def init_state(team_size):
        return np.random.normal(0, 0.5, team_size)  
    

elif setting == "ctn1":
    include_team_effect=0
    instt=0
    round_Mit=0
    state_dim=1
    state_space = None
    possible_values = None
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0, alpha=50, beta=0.05):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        # s_j = np.mean(matched_team.states, axis=0)
        # def r_base(s, a):
        #     return  s * (2.0 * a - 1.0) 
        
        r_list = []
        for s_i, Aijt in zip(Sijt, Aijt):
            r_list.append(s_i)
        return np.hstack(r_list).reshape(-1,1)
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states, t=0,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1,
                            within_team_effect = within_team_effect
                            , gamma=0.3, delta=0.1):
        # s_j = matched_team_current_states
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        next_s = []
        for s_i, Aijt in zip(Sijt, Aijt):
            epsilon = np.random.normal(0, ctn_state_sd)  
            next_s.append(0.3 * s_i*(2.0 * Aijt - 1.0) + epsilon)
        return np.hstack(next_s).reshape(-1, 1)
    
    def init_state(team_size):
        return np.random.normal(0, 0.5, team_size)  
    

elif setting == "ctn2":
    
    include_team_effect=0
    instt=0
    round_Mit=0
    state_dim=1
    state_space = None
    possible_values = None
    
    def reward_function(Sijt, Aijt, matched_team_current_states=None, team_current_states=None, i=0, t=0, alpha=50, beta=0.05):
        Sijt = np.atleast_1d(Sijt)
        Aijt = np.atleast_1d(Aijt)
        def r_base(s, a):
            return s* (2.0 * a - 1.0) 
        
        r_list = []
        # print('Sijt', Sijt, 'shape', Sijt.shape)
        # print(Aijt, Aijt.shape)
        for s_i, a in zip(Sijt, Aijt):
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
        for s_i, Aijt in zip(Sijt, Aijt):
            epsilon = np.random.normal(0, ctn_state_sd)  
            next_s.append(0.3 *  (2.0 * Aijt - 1.0) *s_i + epsilon)
        return np.hstack(next_s).reshape(-1, 1)
    def init_state(team_size):
        return np.random.normal(0, 0.5, team_size)  
    
elif setting == "ctn3":
    instt=0
    round_Mit=0
    possible_values = None 
    state_dim=2
    
    def reward_function(Sijt, actions, matched_team):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        actions: Actions taken (not used in this function, but presumably relevant elsewhere).
        matched_team: The team matched with (not used in this function, but presumably relevant elsewhere).
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        sharpness: The sharpness parameter determines how rapidly the reward decreases as the sleep deviates from the optimal value.
        """
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt *(2*actions-1)
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team, ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=0.3,
                            gamma_1=gamma_1):
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
        
        # Split the feature and observation dimensions
        s_i = Sijt 
        s_j = np.mean(matched_team.states, axis=0)
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        competition_effect = gamma_1 * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + competition_effect + epsilon
        next_s_i[next_s_i>13] = 13
        next_s_i[next_s_i<0] = 0
        # next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        
        return next_s_i
    
    def init_state(team_size):
        return np.random.uniform(0, 13, team_size)


elif setting == "ctn4":
    include_team_effect=0
    # only_states=0
    instt=0
    round_Mit=0
    possible_values = None 
    state_dim=2
    
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]  *(2*Aijt-1)
        
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
        individual_effect = -0.3 * (2.0 * Aijt.reshape(-1,1) - 1.0) * ( s_i)
        # competition effect
        competition_effect = gamma_1 * (s_j - s_i)
        # within team effect
        team_effect = within_team_effect * (team_current_states - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = individual_effect+ epsilon# + team_effect + competition_effect 
        # next_s_i[next_s_i>13] = 13
        # next_s_i[next_s_i<0] = 0
        # next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_s_i
    
    def init_state(team_size):
        return np.random.normal(0, 0.5, team_size)  

elif setting == "ctn41":
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values = None 
    state_dim=2
    
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0] # *(2*Aijt-1)
        
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
        next_s_i = individual_effect + team_effect + competition_effect + epsilon
        next_s_i[next_s_i>13] = 13
        next_s_i[next_s_i<0] = 0
        # next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_s_i
    
    def init_state(team_size):
        return np.random.uniform(0, 13, team_size)
    
elif setting == "ctn5":
    include_team_effect=1
    instt=1
    round_Mit=0
    possible_values = None 
    state_dim=2
    instt_indicator = np.concatenate((np.repeat([0], int(num_teams/2)),np.repeat([1], num_teams - int(num_teams/2)))).reshape((-1,1))
    
    
    def reward_function(Sijt, actions, matched_team_current_states, team_current_states):
        """
        Sijt: Sleeping hours matrix with the last dimension being the individual sleep values.
        actions: Actions taken (not used in this function, but presumably relevant elsewhere).
        matched_team: The team matched with (not used in this function, but presumably relevant elsewhere).
        optimal_sleep: The amount of sleep that yields the maximum reward.
        max_reward: Maximum achievable reward (when a team gets optimal sleep).
        sharpness: The sharpness parameter determines how rapidly the reward decreases as the sleep deviates from the optimal value.
        """
        rewards = Sijt[...,0]  *(2*actions-1)
        return rewards if rewards.size > 1 else rewards.item()
    
    def transition_function(Sijt, Aijt, matched_team_current_states, 
                            team_current_states,
                            ctn_state_sd = ctn_state_sd,
                            optimal_sleep=0, individual_action_effect=individual_action_effect,
                            gamma_1=gamma_1, gamma_2=gamma_2,
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
        Mit = matched_team_current_states
        
        gamma_list = [gamma_1, gamma_2]
        # Split the feature and observation dimensions
        s_i, instt_i = Sijt[..., :-1], Sijt[..., -1:]
        s_j, instt_j = Mit[..., :-1], Mit[..., -1:]
        team_current_states = team_current_states[...,:-1]
        # Individual action effect
        individual_effect = individual_action_effect * (2.0 * Aijt.reshape(-1,1) - 1.0) * (optimal_sleep - s_i)
        
        # within team effect
        team_effect = within_team_effect * (team_current_states - s_i)
        
        # Matching effect
        matching_mask = np.equal(instt_i, instt_j[..., np.newaxis,:])
        
        
        # Convert boolean mask to integer indices (0 for False, 1 for True)
        matching_indices = matching_mask.astype(int)
        
        # Use integer indices to get gamma values from gamma_array
        gamma_array = np.array(gamma_list)
        gamma_values = gamma_array[matching_indices]
      
        # gamma_values = np.array([gamma_list[int(i)] for i in np.equal(instt_i_expanded, instt_j_expanded)])
        competition_effect = gamma_values * (s_j - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i + individual_effect + team_effect + competition_effect + epsilon
        next_s_i[next_s_i>13] = 13
        next_s_i[next_s_i<0] = 0
        next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_sleep_i
    
    def init_state(team_size):
        return np.random.uniform(0, 13, team_size)
elif setting == "ctn6":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values = np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = 0.25*Sijt[...,0]**2*(2*Aijt-1) + 4*Sijt[..., 0]
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

elif setting == "ctn61":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]*(2*Aijt-1) + Sijt[..., 0]
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
        return np.random.normal(size=team_size)  #np.random.uniform(0, 13, team_size)

elif setting == "ctn62":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = 0.25*Sijt[...,0]**2*(2*Aijt-1) + Sijt[..., 0]
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

elif setting == "ctn63":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]*(2*Aijt-1) + Sijt[..., 0]
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
        team_effect = within_team_effect * np.sign(team_current_states - s_i) * s_i
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = s_i - 0.3*s_i + individual_effect + team_effect + epsilon +competition_effect
        # next_s_i[next_s_i>13] = 13
        # next_s_i[next_s_i<0] = 0
        return next_s_i
    
    def init_state(team_size):
        return np.random.uniform(0, 13, team_size)

elif setting == "ctn64":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    # only_states=0
    instt=0
    round_Mit=0
    include_Mit=0
    possible_values =None # np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = 0.25*Sijt[...,0]**(2*Aijt-1) + Sijt[..., 0]
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

elif setting == "ctn64":
    '''
    tabular counterpart for ctn4 setting
    '''
    include_team_effect=1
    only_states=0
    instt=0
    round_Mit=0
    include_Mit=include_Mit
    possible_values =None # np.arange(0, 13)
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = 0.25*Sijt[...,0]**(2*Aijt-1) + Sijt[..., 0]
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
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]*(2*Aijt-1) + 0.25*Sijt[..., 0]
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
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = Sijt[...,0]*(2*Aijt-1) + 0.25*Sijt[..., 0]
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
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = 2*Sijt[...,0]+Sijt[...,1] - 0.25*(2.0 * Aijt - 1.0)
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
elif setting == "ctn4_p2":
    include_team_effect=0
    instt=0
    round_Mit=0
    possible_values = None 
    state_dim=2
    
    def reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, i=0, t=0):
        Sijt = np.atleast_1d(Sijt)
        rewards = np.sum(Sijt[...,:])  *(2*Aijt-1)
        
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
        individual_effect = 0.9 * (2.0 * Aijt.reshape(-1,1) - 1.0) * s_i  -\
            0.0 * (2.0 * Aijt.reshape(-1,1) - 1.0) * s_i**2 +\
            0.0 *(2* Aijt.reshape(-1,1) - 1.0) * np.vstack((s_i[:, 1], s_i[:, 0])).T
        # competition effect
        competition_effect = gamma_1 * (s_j - s_i)
        # within team effect
        team_effect = within_team_effect * (team_current_states - s_i)
        
        # Noise
        epsilon_shape = s_i.shape[:-1] + (1,)  # Add an extra dimension for broadcasting
        epsilon = np.random.normal(0, ctn_state_sd, size=epsilon_shape) 
        
        # Calculate next state
        next_s_i = individual_effect+ epsilon# + team_effect + competition_effect 
        # next_s_i[next_s_i>13] = 13
        # next_s_i[next_s_i<0] = 0
        # next_sleep_i = np.concatenate((next_s_i, instt_i), axis=-1)
        return next_s_i
    
    def init_state(team_size):
        return np.random.normal(-5, 0.5, (team_size, 2))  
    
if state_combine_friends:
    include_team_effect=0
if within_team_effect == 0:
    include_team_effect=0
#%% fit the Q model if not fitted
def target_policy(Sijt):
    Sijt = np.atleast_1d(Sijt)
    return np.ones(len(Sijt))

is_onpolicy =0 
if is_onpolicy:
    my_target_policy = target_policy
    action_space =[1]
else:
    my_target_policy =None
    action_space =[0,1]

#%% generate data
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
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + cluster_noise[j, t] + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            cluster_reward_list = [make_cluster_reward(i) for i in range(num_teams)]
            cluster_transition_list = [transition_function for i in range(num_teams)]
        elif corr_type == "r_ex":
            # print('corr_type', corr_type)
            cluster_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            def make_cluster_reward(j):
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + cluster_noise[j] + np.random.normal(0, sd_white)
            
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
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
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
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
            
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
            # cluster_transition_list = [transition_function for i in range(num_teams)]

            # cluster_noise = np.random.normal(0, sd_cluster_ex_noise, num_teams)
            # # BUG NOTICE: the reward function for each cluster should not share the same object,
            # # which can be caused by defining the cluster reward function with the same name inside the following for
            # # loop for all teams
            # cluster_reward_list = [uti.autoex_reward(cluster_noise[i], reward_function,
            #                                     autoregressive_coef, std_dev=sd_white).generate_autoregressive_data for i in range(num_teams)]
            # print("r_autoex_s_exsubject", 'state_ex_noise', state_ex_noise)
            states_noise = np.random.normal(0, state_ex_noise, (num_teams, max(num_weeks * horizon+1, burnin+1)))
            def make_cluster_transition(i):
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t, week=None,
                 competition_indicator = None,
                 next_week = None):
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
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states)# + noise + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) 
                return cluster_transition
        elif corr_type == "rs_autoexsubject":
            def make_cluster_reward():
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) #+ noise + np.random.normal(0, sd_white)
            
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t):
                    noise = np.random.normal(0, sd_cluster_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
        elif corr_type=="r_autoex_s_exsubject":
            def make_cluster_reward():
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None, week=None,
                 competition_indicator = None,
                 next_week = None):
                    # noise = np.random.normal(0, sd_cluster_ex_noise)
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states)# + noise + np.random.normal(0, sd_white)
                return cluster_reward_function
            def make_cluster_transition():
                def cluster_transition(Sijt, Aijt, matched_team_current_states, team_current_states, t, week=None,
                 competition_indicator = None,
                 next_week = None):
                    noise = np.random.normal(0, state_ex_noise)
                    return transition_function(Sijt, Aijt, matched_team_current_states, team_current_states) + noise
                return cluster_transition
            
        else:
            def make_cluster_reward():
                def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
                    return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
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
        #     def cluster_reward_function(Sijt, Aijt, matched_team_current_states, team_current_states, t, i=None):
        #         return reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + np.random.normal(0, sd_white)
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

def concatenate_history(states, history_length):
    """Concatenate history of states."""
    new_states = []
    for i in range(len(states)):
        # Gather history for the current state
        history = [states[max(i - j, 0)] for j in range(history_length)]
        
        # Concatenate history and current state
        state_with_history = np.concatenate(history, axis=-1)
        new_states.append(state_with_history)
    
    return np.array(new_states)


def process_data(rollouts, only_states=0, include_Mit=0, history_length=1):
    all_possible_states=[]
    if only_states:
        for rollout in rollouts:
            rollout["states"] = rollout["states"][:,:,0]
            rollout["states"] = rollout["states"][..., np.newaxis]
            rollout["next_states"] = rollout["next_states"][:,:,0]
            rollout["next_states"]= rollout["next_states"][..., np.newaxis]
            
            if possible_values is not None:
                tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                all_possible_states.append(tmp_states_values)
            del rollout["Mits"], rollout["next_Mits"], rollout["team_states"], rollout["next_team_states"], rollout["team_id"], rollout['matched_id']

            
    elif state_combine_friends:
        for rollout in rollouts:
            states_tmp=np.empty((0, team_size))
            next_states_tmp=np.empty((0, team_size))
            for i in range(team_size):
                for j in range(rollout['states'].shape[1]):
                    s = np.concatenate((rollout['states'][i, j].flatten(), rollout['states'][:i, j].flatten(), rollout['states'][i+1:,j].flatten())).reshape(1, -1)
                    states_tmp = np.concatenate((states_tmp, s), axis=0)
                    
                    s = np.concatenate((rollout['next_states'][i, j].flatten(), rollout['next_states'][:i, j].flatten(), rollout['next_states'][i+1:, j].flatten())).reshape(1,-1)
                    next_states_tmp = np.concatenate((next_states_tmp, s), axis=0)
            
            rollout['states']= states_tmp.reshape(team_size, -1, team_size)
            rollout['next_states']= next_states_tmp.reshape(team_size, -1, team_size)
            
            if include_Mit:
                Mit = rollout["Mits"]
                if round_Mit:
                    Mit = round_to_nearest(Mit, possible_values)
                rollout["states"] = np.concatenate((np.tile(Mit[np.newaxis,...], (team_size, 1, 1)), rollout["states"]), axis=-1)

                next_Mit = rollout["next_Mits"]
                if round_Mit:
                    next_Mit = round_to_nearest(next_Mit, possible_values)
                rollout["next_states"] = np.concatenate((np.tile(next_Mit[np.newaxis,...], (team_size, 1, 1)), rollout["next_states"]), axis=-1)
         
            
            del rollout["Mits"], rollout["next_Mits"], rollout["team_states"], rollout["next_team_states"], rollout["team_id"], rollout['matched_id']
            if possible_values is not None:
                tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                all_possible_states.append(tmp_states_values)

    elif include_team_effect:
        if instt:
            raise NotImplementedError()
            for rollout in rollouts:
                states =  rollout["states"][:,:,:-1]
                instt_a =  rollout["states"][:,:,-1]
                instt_b = rollout["Mits"][:, -1]
                team_values = np.mean(rollout["states"][:,:,:-1], axis=0)
                Mits = rollout["Mits"][:, :-1]
                if round_Mit:
                    Mits = round_to_nearest(Mits, possible_values)
                matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
                matching_indices = matching_mask.astype(int)
                rollout["states"] = np.concatenate((np.tile(Mits, (team_size, 1, 1)), np.tile(team_values, (team_size, 1, 1)), states, matching_indices[...,np.newaxis]), axis=-1) #
                
                next_states =  rollout["next_states"][:,:,:-1]
                instt_a =  rollout["next_states"][:,:,-1]
                instt_b = rollout["next_Mits"][:, -1]
                next_team_values = np.mean(rollout["next_states"][:,:,:-1], axis=0)
                next_Mits = rollout["next_Mits"][:, :-1]
                if round_Mit:
                    next_Mits = round_to_nearest(next_Mits, possible_values)
                matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
                matching_indices = matching_mask.astype(int)
                rollout["next_states"] = np.concatenate((np.tile(next_Mits, (team_size, 1, 1)), np.tile(next_team_values, (team_size, 1, 1)), next_states, matching_indices[...,np.newaxis] ), axis=-1) #
                
                if possible_values is not None:
                    tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                    all_possible_states.append(tmp_states_values)
                del rollout["Mits"], rollout["next_Mits"], rollout["team_id"], rollout['matched_id']
        else:
            for rollout in rollouts:
                team_values = rollout["team_states"] # np.mean(rollout["states"], axis=0)
                if include_Mit:
                    Mit = rollout["Mits"]
                    if round_Mit:
                        Mit = round_to_nearest(Mit, possible_values)
                    rollout["states"] = np.concatenate((np.tile(Mit[np.newaxis,...], (team_size, 1, 1)), np.tile(team_values, (team_size, 1, 1)), rollout["states"]), axis=-1)
                else:
                    rollout["states"] = np.concatenate((np.tile(team_values, (team_size, 1, 1)), rollout["states"]), axis=-1)
  
                next_team_values = rollout["next_team_states"] #np.mean(rollout["next_states"], axis=0)
                if include_Mit:
                    next_Mit = rollout["next_Mits"]
                    if round_Mit:
                        next_Mit = round_to_nearest(next_Mit, possible_values)
                    rollout["next_states"] = np.concatenate((np.tile(next_Mit[np.newaxis,...], (team_size, 1, 1)), np.tile(next_team_values, (team_size, 1, 1)), rollout["next_states"]), axis=-1)
                else:
                    rollout["next_states"] = np.concatenate((np.tile(next_team_values, (team_size, 1, 1)), rollout["next_states"]), axis=-1)
                               
                del rollout["Mits"], rollout["next_Mits"], rollout["team_states"], rollout["next_team_states"], rollout["team_id"], rollout['matched_id']
                if possible_values is not None:
                    tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                    all_possible_states.append(tmp_states_values)
    else:
        if instt:
            for rollout in rollouts:
                states =  rollout["states"][:,:,:-1]
                instt_a =  rollout["states"][:,:,-1]
                instt_b = rollout["Mits"][:, -1]
                Mits = rollout["Mits"][:, :-1]
                if round_Mit:
                    Mits = round_to_nearest(Mits, possible_values)
                matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
                matching_indices = matching_mask.astype(int)
                rollout["states"] = np.concatenate((np.tile(Mits, (team_size, 1, 1)), states, matching_indices[...,np.newaxis]), axis=-1) 
                
                next_states =  rollout["next_states"][:,:,:-1]
                instt_a =  rollout["next_states"][:,:,-1]
                instt_b = rollout["next_Mits"][:, -1]
                next_Mits = rollout["next_Mits"][:, :-1]
                if round_Mit:
                    next_Mits = round_to_nearest(next_Mits, possible_values)
                matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
                matching_indices = matching_mask.astype(int)
                rollout["next_states"] = np.concatenate((np.tile(next_Mits, (team_size, 1, 1)), next_states, matching_indices[...,np.newaxis]), axis=-1) 
                if possible_values is not None:
                    tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                    all_possible_states.append(tmp_states_values)
                del rollout["Mits"], rollout["next_Mits"], rollout["team_id"], rollout['matched_id']
        else:
            for rollout in rollouts:
                Mit = rollout["Mits"]
                if round_Mit:
                    Mit = round_to_nearest(Mit, possible_values)
                rollout["states"] = np.concatenate((np.tile(Mit[np.newaxis,...], (team_size, 1, 1)), rollout["states"]), axis=-1)
                next_Mit = rollout["next_Mits"]
                if round_Mit:
                    next_Mit = round_to_nearest(next_Mit, possible_values)
                rollout["next_states"] = np.concatenate((np.tile(next_Mit[np.newaxis,...], (team_size, 1, 1)), rollout["next_states"]), axis=-1)
                del rollout["Mits"], rollout["next_Mits"], rollout["team_states"], rollout["next_team_states"], rollout["team_id"], rollout['matched_id']
                if possible_values is not None:
                    tmp_states_values = np.unique(rollout["states"].reshape(-1, rollout["states"].shape[-1]), axis=0)
                    all_possible_states.append(tmp_states_values)
    
    # After processing under each condition, apply history concatenation
    if history_length >1:
        for rollout in rollouts:
            rollout['states'] = concatenate_history(rollout['states'], history_length)
            rollout['next_states'] = concatenate_history(rollout['next_states'], history_length)
            if transition_state_type == "weekly" and not only_states:
                if delete_week_end:
                    rollout["states"] = np.delete(rollout["states"], list(range(horizon-1, num_weeks*horizon-1, horizon)), axis=1)
                    rollout["next_states"] = np.delete(rollout["next_states"], list(range(horizon-1, num_weeks*horizon-1, horizon)), axis=1)
                    rollout['actions'] =  np.delete(rollout["actions"], list(range(horizon-1, num_weeks*horizon-1, horizon)), axis=1)
                    rollout['rewards'] =  np.delete(rollout["rewards"], list(range(horizon-1, num_weeks*horizon-1, horizon)), axis=1)
                elif include_weekend_indicator:
                    array_length = num_weeks * horizon - 1
                    weekend_indicator = np.zeros(array_length)
                    weekend_indicator[list(range(horizon - 1, array_length, horizon))] = 1
                    weekend_indicator = np.tile(weekend_indicator.reshape(1, -1, 1),(rollout['states'].shape[0], 1, 1))
                    rollout["states"] = np.concatenate((rollout["states"], weekend_indicator), axis=2)
                    array_length = num_weeks * horizon - 1
                    weekend_indicator = np.zeros(array_length)
                    weekend_indicator[list(range(horizon - 2, array_length, horizon))] = 1
                    weekend_indicator = np.tile(weekend_indicator.reshape(1, -1, 1),(rollout['states'].shape[0], 1, 1))
                    rollout["next_states"] = np.concatenate((rollout["next_states"], weekend_indicator), axis=2)
  
    if possible_values is not None:
        all_possible_states = np.unique(np.vstack(all_possible_states), axis=0)
    else:
        all_possible_states=None
    return rollouts, all_possible_states



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
                                      # collect_for_training=0,
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
#%% if random policy
if method in ["random", "action=1", "action=0"]:
    if method == "random":
        def target_policy(Sijt):
            Sijt = np.atleast_1d(Sijt)
            return np.random.choice(action_space, len(Sijt))
    elif method == "action=1":
        def target_policy(Sijt):
            Sijt = np.atleast_1d(Sijt)
            return np.ones(len(Sijt))
    elif method == "action=0":
        def target_policy(Sijt):
            Sijt = np.atleast_1d(Sijt)
            return np.zeros(len(Sijt))
    av_r, dis_r = evaluate_policy(target_policy, nthread=nthread, corr_eva=corr_eva)
    sys.stdout.flush()
    with open(file_name, "wb") as f:
        pickle.dump({'av_r':av_r, 'dis_r':dis_r}, f)
    exit()
#%%
data_path = data_save_path()
whole_data_path = data_path + "/rollouts.pkl" if data_path is not None else None
if whole_data_path is not None and os.path.exists(whole_data_path):
    try:
        with open(whole_data_path, 'rb') as f:
            rollouts = pickle.load(f)
        rollouts = rollouts['rollouts']
        need_data=0
    except EOFError:
        print("Caught EOFError: The file may be empty or corrupted.")
        need_data=1
else:
    need_data=1
if need_data:
#%%
    teams = renew_teams(num_teams, num_weeks, team_size=team_size, horizon=horizon, is_training=True, is_record_all=0, transition_state_type=transition_state_type)
    # states shape: (i, t, p)
    # rewards shape: (i, t)
    # Mit shape: (t, p)
    # actions shape: (i, t). action dim=1
    # matched id shape: (t)
    rollouts, _ = simulate_teams(teams, num_weeks, save_path = data_path, individual_policy = my_target_policy,
                                 only_states=only_states,
                                 transition_state_type=transition_state_type, 
                                 delete_week_end=delete_week_end, burnin=burnin)
    # plt.plot(rollouts[0]["states"][:,:,0])
    # for i in range(num_teams):
    #     plt.plot(rollouts[i]["states"][0,:, 0])
#%%
# plot_state_trajectories(trajectories, num_teams, is_legend=0, instt= instt)
# plot_state_trajectori|es(trajectories, num_teams, 'reward', is_legend=0, instt=instt)
rollouts, all_possible_states = process_data(rollouts, only_states=only_states, include_Mit=include_Mit, 
                                             history_length=history_length)
    
#%% model settings
'''
Distinguishing Between `statesmodels_cov` and `cov_struct`:

1. `statesmodels_cov` (classes ending with "_sm" in functions.cov_struct):
   - Purpose: To learn the Q function using the GEE Bellman equation.
   - Implementation: Directly calls `statsmodel.api` for its faster implementation.

2. `cov_struct` (classes not ending with "_sm"):
   - Purpose: Even though each action has its own Q function (and hence its own `statesmodels_cov`), the parameters related to the working correlation structure are estimated by pooling residuals from different actions. This pooled estimation is achieved using `cov_struct`.
   - Usage: Comes into play when `separate_corr` is set to 0. It estimates common parameters for various `statesmodels_cov`.
   
3. if `separate_corr`, then each action has its own parameters in the working correlation structure.
    
'''
if method == 'exchangeable':
    cov_struct =cov_structs.Exchangeable()
    statsmodels_cov = cov_structs.Exchangeable_sm()
    separate_corr = 0
    rho=None
elif method =="smexchangeable":
    cov_struct =cov_structs.Exchangeable()
    statsmodels_cov = cov_structs.statsExchangeable()
    separate_corr = 1
    rho=None
elif method == 'independent':
    cov_struct = cov_structs.Independence()
    if optimal_GEE:
        statsmodels_cov = cov_structs.Independence_sm()
    else:
        statsmodels_cov = None
    separate_corr = 0
    rho=None
elif method == "autoex":
    if new_cov:
        cov_struct =cov_structs.Autoex(set_weight=None, var_td=1)
        statsmodels_cov = cov_structs.Autoex_sm()
    else:
        cov_struct =cov_structs.Autoex_old(set_weight=None, var_td=1)
        statsmodels_cov = cov_structs.Autoex_sm_old()
    separate_corr = 0
    rho =None
    # rho={'var_alpha': 4,
    #   'var_epsilon/1-phi^2': 1,
    #   'autoregressive_coef': 0.8,
    #   'var_td': 0}
elif method == "autoex_exsubject":
    ## r~autoex ,s~exsubject
    cov_struct = cov_structs.Autoex_exsubject(set_weight=None, var_td=1)
    statsmodels_cov = cov_structs.Autoex_exsubject_sm()
    separate_corr = 0
    rho =None
elif method == "autoexsubject":
    ## r~autoexsubject ,s~exsubject
    cov_struct =cov_structs.Autoexsubject(set_weight=None, var_td=1)
    # cov_struct.var_td=var_td 
    statsmodels_cov = cov_structs.Autoexsubject_sm()
    separate_corr = 0
    rho =None
    # rho={'var_alpha': 4,
    #   'var_epsilon/1-phi^2': 1,
    #   'autoregressive_coef': 0.8,
    #   'var_td': 0}
elif method == "exchangeable_subjects":
    cov_struct= cov_structs.Exchangeable_subjects()
    statsmodels_cov = cov_structs.Exchangeable_subjects_sm()
    separate_corr = 0
    rho=None
elif method == "nn":
    FQI_trainer = OfflineFQI(hidden_nodes = hidden_nodes, update_target_every=update_target_every)
    FQI_trainer.load_data(rollouts)
    FQI_trainer.fit(num_epochs=1000)
    av_r, dis_r = evaluate_policy(FQI_trainer.predict_action, nthread=nthread, corr_eva=corr_eva)
    sys.stdout.flush()
    with open(file_name, "wb") as f:
        pickle.dump({'av_r':av_r, 'dis_r':dis_r}, f)
    exit()
else:
    raise ValueError("Invalid method")
#%%
# if not os.path.exists(qmodel_file):
# CV select number of basis
# using one_hot some times have 0 observation on certain catagories and cause error in statsapi
if cv_criterion not in ["min", "1sd"]:
    if setting == "tab2":
        include_bias=0
    else:
        include_bias=1 
    cluster = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=int(cv_criterion),
                       include_bias=include_bias,
                       all_possible_states=all_possible_states) for rollout in rollouts]
#%%
elif not cv_in_training: #  cv_criterion == "min" and basis != "one_hot":
    if basis == "polynomial":
        num_basis_list = [1,2,3,4]
    elif basis == "rbf":
        num_basis_list = [2,3,4,5]
    cv_path = find_cv_file()
    cv_file = 'CV_basis_res.json' if cv_seed=="None" else 'CV_basis_res_cv_seed'+cv_seed+'.json'
    fitted = False
    if os.path.exists(cv_path+'/'+cv_file) and not refit:
        Basis_res = namedtuple("Basis_res", ["num_basis_min", 'num_basis_1se', "cv_loss", "cv_se", "basis",  
                                     "num_basis_list"])
        with open(cv_path+'/'+cv_file, 'r') as f:
            data = json.loads(f.read())
        CV_basis_res = Basis_res(**data)
        if cv_loss not in CV_basis_res.num_basis_min.keys():
            fitted = False
        else:
            fitted = True
    if not fitted:
        sys.stdout.flush()
        if new_uti:
            if cv_seed== "None":
                cv_seed=None
            else:
                cv_seed=int(cv_seed)
            CV_basis_res= uti.select_num_basis_cv(rollouts, cov_struct, statsmodels_cov, 
                                                   action_space,basis = basis, 
                                                   num_threads=nthread,
                                                   seed=cv_seed,gamma=gamma,
                                                   num_basis_list=num_basis_list,
                                                   num_batches=num_batches, new_GEE=new_GEE,
                                                   file_path=cv_path, optimal_GEE=optimal_GEE, combine_actions=combine_actions,
                                                   refit=refit)
        else:
            if cv_seed== "None":
                cv_seed=None
            else:
                cv_seed=int(cv_seed)
            CV_basis_res= uti_old.select_num_basis_cv(rollouts, cov_struct, statsmodels_cov, 
                                                   action_space,basis = basis, 
                                                   num_threads=nthread,
                                                   num_basis_list=num_basis_list,
                                                   seed=cv_seed,
                                                   num_batches=num_batches, new_GEE=new_GEE)
        def convert(o):
            if isinstance(o, np.float32):   
                return float(o)            
            raise TypeError
        with open(cv_path+'/'+cv_file, 'w') as f:
            f.write(json.dumps(CV_basis_res._asdict(), default=convert))
        if only_cv:
            exit()
    if cv_criterion == "min":
        selected_num_basis = CV_basis_res.num_basis_min[cv_loss]
    elif cv_criterion == "1se":
        selected_num_basis = CV_basis_res.num_basis_1se[cv_loss]
    cluster = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=selected_num_basis) for rollout in rollouts]
# else:
#%% 
#     cluster = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=int(cv_criterion),
#                        include_bias=1,
#                        all_possible_states=all_possible_states) for rollout in rollouts]
else:
    # GEE = GEE_Q.GEE_fittedQ(cov_struct, statsmodels_cov=statsmodels_cov, 
    #                   rho = rho,
    #                   gamma=gamma, action_space=action_space, separate_corr=separate_corr,
    #                   statsmodel_maxiter=1)
    qmodels=None 
    dep_params=None
    selected_num_basis =None
    cluster_old=None
    if basis == "polynomial":
        num_basis_list = [1,2,3,4]
    elif basis == "rbf":
        num_basis_list = [2,3,4,5]
    if cv_seed== "None":
        cv_seed=None
    else:
        cv_seed=int(cv_seed)
            
    for iter_num in range(100):
        if dep_params is not None:
            cov_struct.dep_params = dep_params
        CV_basis_res= uti.select_num_basis_cv(rollouts, cov_struct, statsmodels_cov, 
                                               action_space,basis = basis, 
                                               num_threads=nthread,
                                               seed=cv_seed,gamma=gamma,
                                               num_basis_list=num_basis_list,
                                               num_batches=num_batches, new_GEE=new_GEE,
                                               file_path=None, q_function_list=deepcopy(qmodels), 
                                               num_basis_old=selected_num_basis)
        if cv_criterion == "min":
            selected_num_basis = CV_basis_res.num_basis_min[cv_loss]
        elif cv_criterion == "1se":
            selected_num_basis = CV_basis_res.num_basis_1se[cv_loss]
        cluster = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=selected_num_basis) for rollout in rollouts]
        
        GEE = GEE_Q.GEE_fittedQ(cov_struct, statsmodels_cov=statsmodels_cov, 
                          rho = rho,
                          gamma=gamma, action_space=action_space, separate_corr=separate_corr,
                          statsmodel_maxiter=1)
        if qmodels is not None:
            GEE.q_function_list = qmodels
        GEE.cluster_old = cluster_old
        _ = GEE.fit(cluster, num_batches=num_batches, max_iter=1,batch_iter=1,
                    verbose=1, accelerate_method=accelerate_method)
        qmodels = deepcopy(GEE.q_function_list)
        dep_params = deepcopy(cov_struct.dep_params)
        cluster_old = cluster
        # selected_num_basis_old = selected_num_basis
        if GEE.inner_converge:
            save_data(GEE, qmodel_file)
            break
        
#%%
if new_GEE:
    GEE = GEE_Q.GEE_fittedQ(cov_struct, statsmodels_cov=statsmodels_cov, 
                      rho = rho,
                      gamma=gamma, action_space=action_space, separate_corr=separate_corr,
                      statsmodel_maxiter=1, 
                      combine_actions=combine_actions, optimal_GEE=optimal_GEE)
    # _ = GEE.fit(cluster, num_batches=num_batches, max_iter=1,batch_iter=1000,
    #             verbose=1, accelerate_method=accelerate_method)

else:
    GEE = GEE_Q_old.GEE_fittedQ(cluster, cov_struct, statsmodels_cov=statsmodels_cov, 
                      rho = rho,
                      gamma=gamma, action_space=action_space, separate_corr=separate_corr)

#%%
if not refit:
    sys.stdout = open("fitting_log.txt", "a")
else:
    sys.stdout = open("fitting_log.txt", "w")
if os.path.exists(qmodel_file) and not refit:
    if new_GEE:
        GEE.load_data(cluster)
    with open(qmodel_file , "rb") as f:
        qmodels = pickle.load(f)['q_function_list']
    GEE.q_function_list = qmodels
else:
    if new_GEE:
        _ = GEE.fit(cluster, num_batches=num_batches, max_iter=1, batch_iter=500,
                    verbose=1, accelerate_method=accelerate_method, global_TD=1)

        # _ = GEE.fit(cluster, num_batches=num_batches, max_iter=4, batch_iter=10,
        #             verbose=1, accelerate_method=accelerate_method, global_TD=1)
    # plt.plot(GEE.TDsqlist[2:])
    else:
        _ = GEE.fit(max_iter=500, verbose=1)
    GEE.summary()
    save_data(GEE, qmodel_file)
#%% evaluate the "optimal" policy
sys.stdout = open("eva_log"+'n_test'+str(num_teams_eva)+'n_weeks'+str(num_weeks_eva)+'horizon_eva'+str(horizon_eva)+'cv_seed'+str(cv_seed)+".txt", "w")
#%%
def target_policy(states):
    return GEE.predict_action(states)
av_r, dis_r = evaluate_policy(target_policy, nthread=nthread, corr_eva=corr_eva)
sys.stdout.flush()
#%%
with open(file_name, "wb") as f:
    pickle.dump({'av_r':av_r, 'dis_r':dis_r}, f)