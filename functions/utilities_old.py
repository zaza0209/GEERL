# -*- coding: utf-8 -*-
"""
Utilities

"""
import numpy as np
import sys
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import pickle
from collections import namedtuple
from functions.simulate_data_1d_flexible import *
# from functions.GEE_Q import *
import functions.GEE_Q as GEE_Q
import functions.GEE_Q_old as GEE_Q_old
from copy import deepcopy
from scipy.spatial.distance import pdist
#%% autoregressive exchangeable reward
class autoex_reward:
    
    def __init__(self,cluster_noise,reward_function, autoregressive_coef=0.8, mean=0, std_dev=1):
        
        self.autoregressive_coef=autoregressive_coef
        self.memo={}
        self.mean = mean
        self.std_dev=std_dev
        self.cluster_noise = cluster_noise
        self.reward_function= reward_function
        
    def generate_autoregressive_data(self, St, At, t, i, matched_team_current_states=None, team_current_states=None, recursive=0):
        """
        Generate autoregressive data at time t based on the autoregressive coefficient.
    
        Parameters:
            t (int): The time step at which to generate the data.
            autoregressive_coefficient (float): The autoregressive coefficient.
            mean (float, optional): Mean of the white noise (error term). Default is 0.
            std_dev (float, optional): Standard deviation of the white noise (error term). Default is 1.
            memo (dict, optional): A dictionary to store previously generated values. Default is None.
    
        Returns:
            float: The generated autoregressive data at time t.
        """
        if (i,t) in self.memo and recursive==1:
            # Return the memoized value if available
            return self.memo[i, t]
        
        if t == 0:
            # For the first time step, generate any arbitrary value (can be improved with actual data initialization)
            autoregressive_data=np.random.normal(loc=self.mean, scale=self.std_dev)
            self.memo[i, t] = autoregressive_data 
            # print(self.reward_function(St, At, matched_team), 'St', St, 'At', At, 'self.cluster_noise',self.cluster_noise)
            reward = self.reward_function(St, At, matched_team_current_states, team_current_states) + autoregressive_data + self.cluster_noise
            # print(self.memo)
            return reward
    
        # Generate white noise (error term) for the current time step
        white_noise = np.random.normal(loc=self.mean, scale=self.std_dev)
        autoregressive_data = self.autoregressive_coef * self.generate_autoregressive_data(St=None, At=None, t=t-1,i=i, recursive=1) + white_noise
        
        # Store the calculated value in the memo dictionary for future reference
        self.memo[i, t] = autoregressive_data 
        reward = self.reward_function(St, At, matched_team_current_states, team_current_states) + autoregressive_data + self.cluster_noise
    
        return reward
    
#%% TD ssq CV--select number of basis
def gaussian_rbf_kernel(x, y, bandwidth):
    distance_squared = np.linalg.norm(x - y)**2
    return np.exp(-distance_squared * bandwidth)

def gaussian_rbf_kernel_vectorized(X, Y, bandwidth):
    # Compute the squared distance matrix in a vectorized form
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distance_squared = np.einsum('ijk,ijk->ij', diff, diff)
    return np.exp(-distance_squared * bandwidth)

def compute_testing_error_in_chunks(X, gtde, bandwidth, chunk_size=100):
    n_samples = X.shape[0]
    testing_err = 0

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        gtde_i = gtde[i:end_i]

        for j in range(i, n_samples, chunk_size):
            end_j = min(j + chunk_size, n_samples)
            gtde_j = gtde[j:end_j]

            # Compute the RBF kernel for the chunk
            K_chunk = gaussian_rbf_kernel_vectorized(X[i:end_i], X[j:end_j], bandwidth)

            # Compute the gtde product for the chunk using vectorization
            gtde_chunk = np.outer(gtde_i, gtde_j)

            # Accumulate the testing error
            testing_err += np.sum(K_chunk * gtde_chunk)

    return testing_err / (n_samples * (n_samples - 1) / 2)

# def compute_testing_error_in_chunks(X, gtde, bandwidth, chunk_size=100):
#     def gtde_product(x1, x2):
#         return x1 * x2
#     n_samples = X.shape[0]
#     testing_err = 0

#     for i in range(0, n_samples, chunk_size):
#         end_i = min(i + chunk_size, n_samples)
#         for j in range(i, n_samples, chunk_size):
#             end_j = min(j + chunk_size, n_samples)

#             # Compute the RBF kernel for the chunk
#             K_chunk = gaussian_rbf_kernel_vectorized(X[i:end_i], X[j:end_j], bandwidth)

#             # Compute the gtde product for the chunk
#             gtde_chunk = np.array([gtde_product(gtde[k], gtde[l]) for k in range(i, end_i) for l in range(j, end_j)])

#             # Accumulate the testing error
#             testing_err += np.sum(K_chunk * gtde_chunk.reshape((end_i - i, end_j - j)))

#     return testing_err / (n_samples * (n_samples - 1) / 2)


def train_test(train_clusters, test_clusters,  cov_struct_copy, statscov_copy, 
               gamma, action_space, bandwidth, cv_loss="kerneldist", verbose=0,num_batches=1, new_GEE=1):
    if new_GEE:
        GEE = GEE_Q.GEE_fittedQ(cov_struct_copy,statsmodels_cov=statscov_copy,
                               gamma=gamma, action_space=action_space, verbose=verbose)
        _ = GEE.fit(train_clusters, num_batches=num_batches)
    else:
        GEE = GEE_Q_old.GEE_fittedQ(train_clusters, cov_struct_copy,statsmodels_cov=statscov_copy,
                               gamma=gamma, action_space=action_space, verbose=verbose)
        _ = GEE.fit()
    tde = np.concatenate(GEE.TD(test_clusters))
    if cv_loss == "TDssq":
        testing_err = np.mean(tde **2)
    elif cv_loss == "kerneldist":
        
        # import tracemalloc
        # tracemalloc.start()

        
        states = np.concatenate([cluster.states for cluster in test_clusters])
        actions = np.concatenate([cluster.actions for cluster in test_clusters])
        state_action =np.hstack([states.transpose(2, 0, 1).reshape(states.shape[2], -1).T, 
                                 actions.reshape(-1, np.prod(actions.shape)).T])
        dim = state_action.shape[1]
        g = gaussian_rbf_kernel_vectorized(state_action, np.zeros((1, dim)), bandwidth).flatten()
        gtde = g*tde
        
        testing_err = compute_testing_error_in_chunks(state_action, gtde, bandwidth)

        # tracemalloc.stop()
    else:
        raise ValueError("Invalid type of CV loss")
    sys.stdout.flush()
    return testing_err, np.mean(tde **2)

def select_num_basis_cv(rollouts, cov_struct, statsmodels_cov, action_space,
                        num_basis_list=[1,2,3], basis='rbf', cv_loss="kerneldist", gamma=0.9,
                        criterion = "min", bandwidth=None,
                        num_threads=1,
                        nfold = 5, seed=42, save_path =None, verbose=1,num_batches=1,
                        new_GEE=1):
    if seed is not None:
        np.random.seed(seed)
    min_test_error = np.inf
    test_err_list = []
    test_tdssq_list = []
    test_se_list = []
    selected_num_basis = num_basis_list[0]

    kf = KFold(n_splits=nfold, shuffle=False)#, random_state=seed)
    if (cv_loss == 'kerneldist') and (bandwidth is None):
        if type(rollouts[0]) == dict:
            states = np.concatenate([rollout["states"] for rollout in rollouts])
            actions = np.concatenate([rollout["actions"] for rollout in rollouts])
        else:
            states = np.concatenate([rollout.states for rollout in rollouts])
            actions = np.concatenate([rollout.actions for rollout in rollouts])

        if states.shape[0]*states.shape[1] > 1000:  # if sample size is too large
            sample_subject_index = np.random.choice(states.shape[0], min(100, states.shape[0]), replace=False)
            sample_time_index = np.random.choice(states.shape[1],  min(100, states.shape[1]), replace=False)
        else:
            sample_subject_index = np.arange(states.shape[0])
            sample_time_index =  np.arange(states.shape[1])
        ### compute bandwidth
        # compute pairwise distance between states for the first piece
        state_action =np.hstack([states[sample_subject_index,:,:][:, sample_time_index, :].transpose(2, 0, 1).reshape(states.shape[2], -1).T, 
                                 actions[sample_subject_index][:, sample_time_index].reshape(1, -1).T])
        pw_dist = pdist(state_action, metric='euclidean')
        
        # states = np.concatenate([rollout["states"] for rollout in rollouts])
        # actions = np.concatenate([rollout["actions"] for rollout in rollouts])
        # if states.shape[0] > 100:  # if sample size is too large
        #     sample_subject_index = np.random.choice(states.shape[0], 100, replace=False)
        # else:
        #     sample_subject_index = np.arange(states.shape[0])
        # ### compute bandwidth
        # # compute pairwise distance between states for the first piece
        # state_action =np.hstack([states[sample_subject_index, :, :].transpose(2, 0, 1).reshape(states.shape[2], -1).T, 
        #                          actions[sample_subject_index,:].reshape(1, -1).T])
        # pw_dist = pdist(state_action, metric='euclidean')
        bandwidth = 1.0 / (2*np.nanmedian(np.where(pw_dist > 0, pw_dist, np.nan)))
        # use the median of the minimum of distances as bandwidth
        if verbose:
            print("Bandwidth chosen: {:.5f}".format(bandwidth))
        del pw_dist
        
    Basis_res = namedtuple("Basis_res", ["num_basis", "test_error", "basis", "criterion",
                                 "num_basis_list", "test_errors_list", "test_se_list", "test_tdssq_list"])

    for num_basis in num_basis_list:
        print("num_basis", num_basis)
        sys.stdout.flush()
        clusters = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=num_basis) for rollout in rollouts]
        # N, T, p = clusters[0].N, clusters[0].T, clusters[0].p

        # num_threads = 1 if N*T*p > 100000 else 4
    
        def run_one(train_indices, test_indices):
            train_clusters = [clusters[i] for i in train_indices]
            test_clusters = [clusters[i] for i in test_indices]
            cov_struct_copy = deepcopy(cov_struct)
            statscov_copy = deepcopy(statsmodels_cov)
            return train_test(train_clusters, test_clusters, cov_struct_copy, statscov_copy, gamma, action_space, 
                              bandwidth, cv_loss, verbose,num_batches=num_batches, new_GEE=new_GEE)

        if num_threads>1:
            res = Parallel(n_jobs=num_threads, prefer="threads")(
                delayed(run_one)(train_indices, test_indices) for train_indices, test_indices in kf.split(clusters)
            )
        else:
            print('no paralle')
            sys.stdout.flush()
            res = []
            fold=0
            for train_indices, test_indices in kf.split(clusters):
                print('********** ============ fold', fold)
                sys.stdout.flush()
                fold+=1
                tmp = run_one(train_indices, test_indices)
                res.append(tmp)
                
        test_errors = []
        tdssq = []
        
        for tuple_item in res:
            test_errors.append(tuple_item[0])
            tdssq.append(tuple_item[1])
    
        test_error = np.mean(test_errors)
        test_tdssq = np.mean(tdssq)
        test_se = np.std(test_errors)/nfold
        if verbose:
            print(test_error)
            sys.stdout.flush()
        test_err_list.append(test_error)
        test_tdssq_list.append(test_tdssq)
        test_se_list.append(test_se)
        
    if verbose:
        print('test_err_list',test_err_list)
        sys.stdout.flush()
    min_test_error = np.nanmin(test_err_list)
    selected_num_basis = num_basis_list[test_err_list.index(min_test_error)]
    
    if criterion == "1se":
        min_test_se = test_se[test_err_list.index(min_test_error)]
        flag = 1
        while flag:
            err = test_err_list[selected_num_basis -1]
            if err < min_test_error + min_test_se:
                selected_num_basis -= 1
            else:
                flag = 0
    cv_basis_res=Basis_res(selected_num_basis, min_test_error, basis, criterion,num_basis_list, test_err_list,test_se_list, test_tdssq_list)
    return cv_basis_res



#%% MC estimate of the rho -- for tabular states
def MC_estimate_rho(state_space, cov_r,transition_function,reward_function, N=10, T = 1000, p=1, seed=0):
    def target_policy(St):
        return int(1)
    system_settings = {'N': N, 'T': T-1,
                       'changepoints': [T-1],
                       'state_functions': [transition_function, transition_function],
                       'reward_functions': [reward_function, reward_function],
                       'state_change_type':'pwconst2',
                       'reward_change_type': 'homogeneous',
                        }
    Q_list = [None for s in state_space]
    for s in state_space:
        # states = np.zeros([N, T, p])
        rewards = np.zeros([N, T-1])
        # actions = np.zeros([N, T-1])
        _, rewards, _ = simulate(system_settings, seed =seed**3+ s, 
                                            target_policy = target_policy,
                                            S0 =  s * np.ones([N,p]), cov=0)
        Q_list[state_space.index(s)] = np.mean(rewards)
    
    # estimate the variance of TD error
    states = np.zeros([N, T, p])
    rewards = np.zeros([N, T-1])
    actions = np.zeros([N, T-1])
    states, rewards, actions = simulate(system_settings, seed =seed**3 +100, 
                                        target_policy = target_policy,
                                        S0 = np.random.binomial(1, 0.5, N).reshape([N,p]), cov=0)
    # tmp = Cluster(states[:, :-1, :], actions, states[:, 1:, :], rewards, evaluate_action=1, basis = "one_hot",
    #               all_possible_categories=np.array([0,1]))
    
    def rho_one(state, action, reward, next_state):
        TD = reward + [Q_list[int(i)] for i in next_state] - [Q_list[int(i)] for i in state] - 2.83
        var_TD = np.var(TD)
        rho = cov_r/(cov_r+var_TD)
        return rho, cov_r+var_TD
    res_list = [rho_one(state, action, reward, next_state) for state, action, reward, next_state in zip(states[:, :-1, :], actions, rewards, states[: ,1:, :])]
    rho_list, margianl_var = zip(*res_list)
    return np.mean(rho_list), np.mean(margianl_var)
    