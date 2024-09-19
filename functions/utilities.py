# -*- coding: utf-8 -*-
"""
Utilities

"""
import numpy as np
import sys, os
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import namedtuple
from functions.team_matching_training_new import *
import functions.GEE_Q as GEE_Q
import functions.GEE_Q_old as GEE_Q_old
# from functions.GEE_Q import *
import json
import time
import torch
from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
#%% autoregressive exchangeable reward
class autoex_reward:
    
    def __init__(self,cluster_noise, reward_function, autoregressive_coef=0.8, mean=0, std_dev=1):
        '''
        generate rewards that is autoregressive temporally and exchangeable between subjects

        Parameters
        ----------
        cluster_noise : int or a list. If int, the team is exchangeblae; If list, exchangeble at a give time point

        '''
        self.autoregressive_coef=autoregressive_coef
        self.memo={}
        self.mean = mean
        self.std_dev=std_dev
        self.cluster_noise = cluster_noise
        self.reward_function = reward_function
    
    def get_cluster_noise(self, t):
        if isinstance(self.cluster_noise, (list, np.ndarray)):
            # print('list noise', self.cluster_noise[t])
            return self.cluster_noise[t]
        else:
            # print('self.cluster_noise',self.cluster_noise)
            return self.cluster_noise
        
    def generate_autoregressive_data(self, Sijt, Aijt, t, i, matched_team_current_states=None, team_current_states=None, recursive=0, week=None,
     competition_indicator = None,
     next_week = None):
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
            # print(self.reward_function(Sijt, Aijt, matched_team), 'Sijt', Sijt, 'Aijt', Aijt, 'self.cluster_noise',self.cluster_noise)
            reward = self.reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + autoregressive_data + self.get_cluster_noise(t)
            # print(self.memo)
            return reward
    
        # Generate white noise (error term) for the current time step
        white_noise = np.random.normal(loc=self.mean, scale=self.std_dev)
        autoregressive_data = self.autoregressive_coef * self.generate_autoregressive_data(Sijt=None, Aijt=None, t=t-1,i=i, recursive=1) + white_noise
        
        # Store the calculated value in the memo dictionary for future reference
        self.memo[i, t] = autoregressive_data 
        reward = self.reward_function(Sijt, Aijt, matched_team_current_states, team_current_states) + autoregressive_data + self.get_cluster_noise(t)
    
        return reward
    
#%% TD ssq CV--select number of basis
def gaussian_rbf_kernel(x, y, bandwidth):
    distance_squared = np.linalg.norm(x - y)**2
    return np.exp(-distance_squared * bandwidth)

def gaussian_rbf_kernel_vectorized(X, Y, bandwidth):
#     # Compute the squared distance matrix in a vectorized form
#     diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
#     distance_squared = np.einsum('ijk,ijk->ij', diff, diff)
#     return np.exp(-distance_squared * bandwidth)
# def optimized_gaussian_rbf_kernel(X, Y, bandwidth):
    # Using cdist to compute the squared Euclidean distance
    distance_squared = cdist(X, Y, 'sqeuclidean')
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
            # K_chunk = optimized_gaussian_rbf_kernel(X[i:end_i], X[j:end_j], bandwidth)

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

def bic(clusters, cov_struct, statsmodels_cov, action_space,
                        num_basis, basis='rbf', gamma=0.9,
                        nfold = 5, seed=None, save_path =None, verbose=1, num_batches=1,
                        new_GEE=1, q_function_list=None, clusters_old=None,
                        optimal_GEE=0, combine_actions=0):
    start = time.time()
    sys.stdout.flush()
    if new_GEE:
        GEE = GEE_Q.GEE_fittedQ(deepcopy(cov_struct),statsmodels_cov=deepcopy(statsmodels_cov),
                               gamma=gamma, action_space=action_space, verbose=verbose,
                               optimal_GEE=optimal_GEE, combine_actions = combine_actions)
        if q_function_list is not None:
            GEE.q_function_list = q_function_list
        GEE.cluster_old = clusters_old
        _ = GEE.fit(clusters, num_batches=num_batches, batch_iter=1000)
    else:
        GEE = GEE_Q_old.GEE_fittedQ(clusters, deepcopy(cov_struct),statsmodels_cov=deepcopy(statsmodels_cov),
                               gamma=gamma, action_space=action_space, verbose=verbose)
        if q_function_list is not None:
            GEE.q_function_list = q_function_list
        _ = GEE.fit()
        
    if verbose: 
        print('fitting time', time.time()-start)
        sys.stdout.flush()
    tde = np.concatenate(GEE.TD(clusters))
    weighted_tde = GEE.weightedTD(clusters)
    n = np.log(np.sum(np.array(GEE.cluster_T)*np.array(GEE.cluster_size))) 
    bic_value = n * num_basis + np.mean(tde **2)
    weighted_bic_value = n * num_basis + np.mean(weighted_tde ** 2)
    return bic_value, weighted_bic_value
    

def train_test(train_clusters, test_clusters, cov_struct_copy, statscov_copy, 
               gamma, action_space, bandwidth, verbose=0, num_batches=1,
               new_GEE=1, q_function_list=None, batch_iter=1000, max_iter=1, 
               train_clusters_old=None, optimal_GEE=0, combine_actions = 0):
    start = time.time()
    # Consider using a simpler model or different initial parameters if possible

    if new_GEE:
        GEE = GEE_Q.GEE_fittedQ(cov_struct_copy, statsmodels_cov=statscov_copy,
                               gamma=gamma, action_space=action_space, verbose=verbose,
                               optimal_GEE=optimal_GEE, combine_actions = combine_actions)
        
        if q_function_list is not None:
            GEE.q_function_list = q_function_list
        GEE.cluster_old = train_clusters_old
        _ = GEE.fit(train_clusters, num_batches=num_batches, batch_iter=batch_iter)
    else:
        GEE = GEE_Q_old.GEE_fittedQ(train_clusters, cov_struct_copy,statsmodels_cov=statscov_copy,
                               gamma=gamma, action_space=action_space, verbose=verbose)
        if q_function_list is not None:
            GEE.q_function_list = q_function_list
        _ = GEE.fit()
        
    if verbose: 
        print('fitting time', time.time()-start)
        sys.stdout.flush()
    tde = np.concatenate(GEE.TD(test_clusters))
    weighted_tde = GEE.weightedTD(test_clusters)
    # if cv_loss == "TDssq":
    testing_err = np.mean(tde **2)
    weighted_tdssq = np.mean(weighted_tde ** 2)
    
    
    # elif cv_loss == "kerneldist":
        
    states = np.concatenate([cluster.states for cluster in test_clusters])
    actions = np.concatenate([cluster.actions for cluster in test_clusters])
    state_action =np.hstack([states.transpose(2, 0, 1).reshape(states.shape[2], -1).T, 
                             actions.reshape(-1, np.prod(actions.shape)).T])
    dim = state_action.shape[1]
    g = gaussian_rbf_kernel_vectorized(state_action, np.zeros((1, dim)), bandwidth).flatten()
    gtde = g*tde
    start = time.time()
    testing_err = compute_testing_error_in_chunks(state_action, gtde, bandwidth)
    if verbose: 
        print('loss time', time.time() - start, 'testing_Err', testing_err)
        sys.stdout.flush()
    
    
    # elif cv_loss == "init_Q":
    states = np.concatenate([cluster.states[:,0,:] for cluster in test_clusters])
    actions = np.concatenate([cluster.actions[:,0] for cluster in test_clusters])
    q_values_init = np.sum(np.max(np.concatenate(GEE.predict_Q(states[:, np.newaxis, :]), axis=1), axis=1))
    
    
    # elif cv_loss == "all_Q":
    states = np.concatenate([cluster.states for cluster in test_clusters])
    actions = np.concatenate([cluster.actions for cluster in test_clusters])
    q_values_all = np.sum(np.max(np.concatenate(GEE.predict_Q(states.reshape(-1,1,states.shape[-1])), axis=1), axis=1))

    # else:
    #     raise ValueError("Invalid type of CV loss")
    return {'tdssq': testing_err,
            'kerneldist': np.mean(tde **2), 
            'init_Q': -1*q_values_init, 
            'all_Q': -1*q_values_all,
            'weighted_tdssq':weighted_tdssq}

    
    
def select_num_basis_cv(rollouts, cov_struct, statsmodels_cov, action_space,
                        num_basis_list=[1,2,3], basis='polynomial', gamma=0.9,
                        bandwidth=None,
                        num_threads=1,
                        nfold = 5, seed=None, save_path =None, verbose=1, num_batches=1,
                        new_GEE=1, file_path=None, q_function_list=None, batch_iter=1000,
                        max_iter=1, num_basis_old=None, combine_actions=0, optimal_GEE=0,
                        refit=0):
    print('num_batches', num_batches, 'num_threads', num_threads)
    sys.stdout.flush()
    if seed is not None:
        np.random.seed(seed)
    
    loss_name_list = ["tdssq", "kerneldist", "init_Q", "all_Q", "weighted_tdssq", 'bic', 'weighted_bic']
    cv_loss_all_basis = {loss_name:[] for loss_name in loss_name_list} #"tdssq":[], 'kerneldist':[], "init_Q":[], "all_Q":[]}
    cv_se_all_basis = {loss_name:[] for loss_name in loss_name_list} #"tdssq":[], 'kerneldist':[], "init_Q":[], "all_Q":[]}


    kf = KFold(n_splits=nfold, shuffle=False)#, random_state=seed)
    # if (cv_loss == 'kerneldist') and (bandwidth is None):
    if bandwidth is None:
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
        state_action=np.hstack([states[sample_subject_index,:,:][:, sample_time_index, :].transpose(2, 0, 1).reshape(states.shape[2], -1).T, 
                                 actions[sample_subject_index][:, sample_time_index].reshape(1, -1).T])
        pw_dist = pdist(state_action, metric='euclidean')
        bandwidth = 1.0 / (2*np.nanmedian(np.where(pw_dist > 0, pw_dist, np.nan)))
        # use the median of the minimum of distances as bandwidth
        if verbose:
            print("Bandwidth chosen: {:.5f}".format(bandwidth))
        del pw_dist, state_action, states, actions, sample_subject_index,  sample_time_index
        
    
    Basis_res = namedtuple("Basis_res", ["num_basis_min", 'num_basis_1se', "cv_loss", "cv_se", "basis",  
                                 "num_basis_list"])

#%%
    for num_basis in num_basis_list:
        fitted=False
        only_bic=False
        ## if we have res for this num_basis
        if file_path is not None:
            cv_tmp_path = file_path + "/cv_num_basis"+str(num_basis)+ ".json"
        else:
            cv_tmp_path = None # "cv_num_basis"+str(num_basis)+ ".json"
        if cv_tmp_path is not None:
            if os.path.exists(cv_tmp_path) and not refit:
                with open(cv_tmp_path, 'r') as f:
                    data = json.loads(f.read())
                cv_tmp = Basis_res(**data)
                set_diff = set(loss_name_list).difference(set(cv_tmp.cv_loss.keys()))
                if not set_diff:
                    fitted = True 
                    cv_loss_dict= cv_tmp.cv_loss 
                    cv_se_dict = cv_tmp.cv_se 
                elif set_diff == {"bic", "weighted_bic"}:
                    fitted=False
                    cv_loss_dict= cv_tmp.cv_loss 
                    cv_se_dict = cv_tmp.cv_se 
                    only_bic=True
                else:
                    fitted=False
                    only_bic=False
          
        ## if not
        if not fitted:
            print("num_basis", num_basis)
            sys.stdout.flush()
            clusters_old=None
            if type(rollouts[0]) == dict:
                clusters = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=num_basis) for rollout in rollouts]
                if num_basis_old is not None:
                    clusters_old = [GEE_Q.Cluster(**rollout, basis=basis, num_basis=num_basis_old) for rollout in rollouts]
            else:
                clusters = [GEE_Q.Cluster(*rollout, basis=basis, num_basis=num_basis) for rollout in rollouts]
                if num_basis_old is not None:
                    clusters_old = [GEE_Q.Cluster(*rollout, basis=basis, num_basis=num_basis_old) for rollout in rollouts]

            if not only_bic:
                def run_one(train_indices, test_indices):
                    train_clusters = [clusters[i] for i in train_indices]
                    test_clusters = [clusters[i] for i in test_indices]
                    if num_basis_old is not None:
                        train_clusters_old = [clusters_old[i] for i in train_indices]
                    else:
                        train_clusters_old = None
                    cov_struct_copy = deepcopy(cov_struct)
                    statscov_copy = deepcopy(statsmodels_cov)
                    if num_threads>1:
                        verbose_run=0
                    else:
                        verbose_run=verbose
                    return train_test(train_clusters, test_clusters, cov_struct_copy, statscov_copy, gamma, action_space, 
                                      bandwidth, verbose_run, num_batches = num_batches,
                                      new_GEE=new_GEE, q_function_list=deepcopy(q_function_list),
                                      batch_iter=batch_iter, max_iter=max_iter, train_clusters_old=train_clusters_old,
                                      optimal_GEE=optimal_GEE, combine_actions = combine_actions)
        
                if num_threads>1:
                    start= time.time()
                    print('num_threads', num_threads)
                    res = Parallel(n_jobs=num_threads)(
                        delayed(run_one)(train_indices, test_indices) for train_indices, test_indices in kf.split(clusters)
                    )
                    print('elapse', time.time() - start)
                    print('finish parallel')
                    sys.stdout.flush()
                else:
                    print('no parallel')
                    sys.stdout.flush()
                    res = []
                    fold=0
                    for train_indices, test_indices in kf.split(clusters):
                        print('********** ============ fold', fold)
                        start= time.time()
                        sys.stdout.flush()
                        fold+=1
                        tmp = run_one(train_indices, test_indices)
                        print('elapse', time.time() - start)
                        sys.stdout.flush()
                        res.append(tmp)
                        
                cv_loss_dict = {loss_name:[] for loss_name in loss_name_list} #"tdssq":[], 'kerneldist':[], "init_Q":[], "all_Q":[]}
                cv_se_dict = {loss_name:[] for loss_name in loss_name_list} #"tdssq":[], 'kerneldist':[], "init_Q":[], "all_Q":[]}
        
                
                for dict_item in res:
                    for key, val in dict_item.items():
                        cv_loss_dict[key].append(val)
                        
                for key, val in cv_loss_dict.items():
                    cv_se_dict[key] = np.std(val)/np.sqrt(nfold)
                    cv_loss_dict[key] = np.mean(val)
                    

            ## finish cv for one num_basis
            bic_value, weighted_bic_value = bic(clusters=clusters, cov_struct=deepcopy(cov_struct), statsmodels_cov=deepcopy(statsmodels_cov), action_space=action_space,
                                    num_basis=num_basis, gamma=gamma, 
                                   nfold = nfold, verbose=0, num_batches=num_batches,
                                    new_GEE=new_GEE, q_function_list=deepcopy(q_function_list),
                                    clusters_old=clusters_old, optimal_GEE=optimal_GEE, combine_actions = combine_actions)
            cv_loss_dict["bic"] = bic_value
            cv_se_dict['bic'] = 0
            
            cv_loss_dict["weighted_bic"] = weighted_bic_value
            cv_se_dict["weighted_bic"] = 0
            if verbose:
                print('loss', cv_loss_dict)
                print('se', cv_se_dict)
                sys.stdout.flush()
            # save result for this num_basis
            if cv_tmp_path is not None:
                cv_tmp=Basis_res(num_basis, num_basis, cv_loss_dict, cv_se_dict, basis, num_basis)
                def convert(o):
                    if isinstance(o, np.float32):   
                        return float(o)            
                    raise TypeError
                with open(cv_tmp_path, 'w') as f:
                    f.write(json.dumps(cv_tmp._asdict(), default=convert))
    
        for key, val in cv_loss_dict.items():
            cv_loss_all_basis[key].append(val)
            cv_se_all_basis[key].append(cv_se_dict[key])


 #%%   
    if verbose:
        print('test_err_list',cv_loss_all_basis)
        sys.stdout.flush()
        
    ## collecting the selected num of basis according to cv min
    selected_num_basis_dict_min = {loss_name:None for loss_name in loss_name_list} #"tdssq":None, 'kerneldist':None, "init_Q":None, "all_Q":None}
    for key, val in cv_loss_all_basis.items():
        try:
            selected_num_basis_dict_min[key] = num_basis_list[np.nanargmin(val)]
        except ValueError as e:
            if str(e) == "All-NaN slice encountered":
                selected_num_basis_dict_min[key] = num_basis_list[0]
            else:
                raise
        # selected_num_basis_dict_min[key] = num_basis_list[np.nanargmin(val)]
        
    ## collecting the selected num of basis according to cv 1se
    selected_num_basis_dict_1se = {loss_name:None for loss_name in loss_name_list} #"tdssq":None, 'kerneldist':None, "init_Q":None, "all_Q":None}
   
    for key, val in cv_loss_all_basis.items():
        min_err = np.nanmin(val)
        if np.isnan(val).any() or np.isnan(cv_se_all_basis[key]).any() or np.isinf(val).any() or np.isinf(cv_se_all_basis[key]).any():
            selected_num_basis_dict_1se[key] = selected_num_basis_dict_min[key]
            continue
        min_index = val.index(min_err)
        se = cv_se_all_basis[key][min_index]
        ub = min_err + se
        i=min_index 
        while i >=0:
            print('i', i, val[i] ,'<= ',ub, (val[i] <= ub))
            if np.isnan(val[i]):
                i -= 1
                continue
            if val[i] <= ub:
                i -= 1 
            else:
                break 
        i += 1 
        print('num_basis_list', num_basis_list, 'i', i, 'min_index', min_index, 'val', val, 'se',se)
        selected_num_basis_dict_1se[key] = num_basis_list[i]
    cv_basis_res=Basis_res(selected_num_basis_dict_min, selected_num_basis_dict_1se, cv_loss_all_basis, cv_se_all_basis, basis, num_basis_list)
    return cv_basis_res



#%% cv team
def train_test_team(train_data, test_data, trainer_copy, approximator_copy, q_degree, matching_state_dim=None, 
                    basis_type="polynomial", cv_loss='kerneldist', bandwidth=None,
                verbose=0):
    
    approximator_copy.assign_q_basis(q_degree=q_degree, q_n_component=q_degree, q_basis_type=basis_type)
    trainer_copy.assign_approximator(approximator_copy)
    _ = trainer_copy.fit(train_data, batch_size=len(train_data),
                               num_epochs=1000 , verbose=verbose)
    
    tde, states_valid, matching_state_valid = trainer_copy.approximator.TD(test_data)
    if isinstance(tde, torch.Tensor):
        tde = tde.numpy()
        states_valid=states_valid.numpy()
        matching_state_valid=matching_state_valid.numpy()
    if cv_loss == "TDssq":
        testing_err = np.mean(tde **2)
    elif cv_loss == "kerneldist":
        # states = np.concatenate([entry["state"].reshape(1,-1) for entry in test_data], axis=0)
        # matching_state = np.concatenate([entry["matching_state"] for entry in test_data]).reshape(-1, 1)

        state_matching_state_valid =np.hstack([states_valid, matching_state_valid])
        dim = state_matching_state_valid.shape[1]
        # g = optimized_gaussian_rbf_kernel(state_action, np.zeros((1, dim)), bandwidth).flatten()
        g = gaussian_rbf_kernel_vectorized(state_matching_state_valid, np.zeros((1, dim)), bandwidth).flatten()
        gtde = g*tde
        start = time.time()
        testing_err = compute_testing_error_in_chunks(state_matching_state_valid, gtde, bandwidth)
        if verbose: 
            print('loss time', time.time() - start, 'testing_Err', testing_err)
            sys.stdout.flush()

    else:
        raise ValueError("Invalid type of CV loss")
    return testing_err, np.mean(tde **2)
    # testing_err = np.mean(tde **2)
    # return testing_err

def select_num_basis_cv_team(data, trainer, approximator, num_weeks, matching_state_dim=None, 
                    basis_type="polynomial",  cv_loss="kerneldist", bandwidth=None,
                        num_basis_list=[1,2,3], criterion='min',
                        num_threads=1,
                        nfold = 5, seed=None, verbose=1, 
                        ):
    '''
    cv for BasisFunctionApproximator

    '''
    sys.stdout.flush()
    if seed is not None:
        np.random.seed(seed)
    min_test_error = np.inf
    test_err_list = []
    test_se_list = []
    test_tdssq_list=[]
    test_tdssq_se_list =[]
    selected_num_basis = num_basis_list[0]

    kf = KFold(n_splits=nfold, shuffle=False)#, random_state=seed)
        
    Basis_res = namedtuple("Basis_res", ["num_basis",'num_basis_tdssq', "test_error", "basis", "criterion",
                                 "num_basis_list", "test_errors_list", "test_se_list", "test_tdssq_list", "test_tdssq_se_list"])
    if (cv_loss == 'kerneldist') and (bandwidth is None):
        states = np.concatenate([entry["state"].reshape(1,-1) for entry in data], axis=0)
        matching_state = np.concatenate([entry["matching_state"] for entry in data]).reshape(-1, 1)

        if states.shape[0] > 1000:  # if sample size is too large
            sample_subject_index = np.random.choice(states.shape[0], min(1000, states.shape[0]), replace=False)
        else:
            sample_subject_index = np.arange(states.shape[0])
        ### compute bandwidth
        # compute pairwise distance between states for the first piece
        state_matching_state =np.hstack([states[sample_subject_index,:], 
                                 matching_state[sample_subject_index,:]])
        pw_dist = pdist(state_matching_state, metric='euclidean')
        bandwidth = 1.0 / (2*np.nanmedian(np.where(pw_dist > 0, pw_dist, np.nan)))
        # use the median of the minimum of distances as bandwidth
        if verbose:
            print("Bandwidth chosen: {:.5f}".format(bandwidth))
        del pw_dist, state_matching_state, states, matching_state, sample_subject_index
    
    ## cv begin
    for num_basis in num_basis_list:
        print("num_basis", num_basis)
        sys.stdout.flush()
        
        ## if we have res for this num_basis
        if os.path.exists("cv_num_basis"+str(num_basis)+ ".json"):
            with open("cv_num_basis"+str(num_basis)+ ".json", 'r') as f:
                data = json.loads(f.read())
            cv_tmp = Basis_res(**data)
            test_error = cv_tmp.test_error
            test_tdssq = cv_tmp.test_tdssq_list
            test_se = cv_tmp.test_se_list
            test_tdssq_se = cv_tmp.test_tdssq_se_list
            
        ## do cv if not
        else:
            def run_one(train_indices, test_indices):
                train_data = [entry for entry in data if entry['week'] in train_indices]
                test_data = [entry for entry in data if entry['week'] in test_indices]
                trainer_copy = deepcopy(trainer)
                approximator_copy = deepcopy(approximator)
                if num_threads>1:
                    verbose_run=0
                else:
                    verbose_run=verbose
                return train_test_team(train_data, test_data, trainer_copy, 
                                       approximator_copy,
                                       q_degree=num_basis, matching_state_dim = matching_state_dim, 
                                    cv_loss=cv_loss, bandwidth=bandwidth,
                                    basis_type=basis_type, verbose=verbose_run)
    
            if num_threads>1:
                start= time.time()
                print('num_threads', num_threads)
                res = Parallel(n_jobs=num_threads)(
                    delayed(run_one)(train_indices, test_indices) for train_indices, test_indices in kf.split(range(num_weeks))
                )
                print('elapse', time.time() - start)
                print('finish parallel')
                sys.stdout.flush()
            else:
                print('no parallel')
                sys.stdout.flush()
                res = []
                fold=0
                for train_indices, test_indices in kf.split(range(num_weeks)):
                    print('********** ============ fold', fold)
                    start= time.time()
                    sys.stdout.flush()
                    fold+=1
                    tmp = run_one(train_indices, test_indices)
                    print('elapse', time.time() - start)
                    sys.stdout.flush()
                    res.append(tmp)
                    
            test_errors = []
            tdssq = []
            for tuple_item in res:
                test_errors.append(tuple_item[0])
                tdssq.append(tuple_item[1])
        
            test_error = np.mean(test_errors)
            test_tdssq = np.mean(tdssq)
            test_se = np.std(test_errors)/np.sqrt(nfold)
            test_tdssq_se = np.std(tdssq)/np.sqrt(nfold)
            if verbose:
                print(test_error)
                print('tdssq', test_tdssq)
                sys.stdout.flush()
            
            # save result for this num_basis
            cv_tmp=Basis_res(num_basis, num_basis, test_error, basis_type, criterion, num_basis, test_error, test_se, test_tdssq, test_tdssq_se)
            def convert(o):
                if isinstance(o, np.float32):   
                    return float(o)            
                raise TypeError
            with open("cv_num_basis"+str(num_basis)+ ".json", 'w') as f:
                f.write(json.dumps(cv_tmp._asdict(), default=convert))
            print('save to ', os.getcwd()+"/cv_num_basis"+str(num_basis)+ ".json")
            sys.stdout.flush()
            
        test_err_list.append(test_error)
        test_tdssq_list.append(test_tdssq)
        test_se_list.append(test_se)
        test_tdssq_se_list.append(test_tdssq_se)
        
    if verbose:
        print('test_err_list',test_err_list)
        sys.stdout.flush()
    min_test_error = np.nanmin(test_err_list)
    selected_num_basis = num_basis_list[test_err_list.index(min_test_error)]
    selected_num_basis_tdssq = num_basis_list[test_tdssq_list.index(np.nanmin(test_tdssq_list))]
    if criterion == "1se":
        min_test_se = test_se[test_err_list.index(min_test_error)]
        flag = 1
        while flag:
            err = test_err_list[selected_num_basis -1]
            if err < min_test_error + min_test_se:
                selected_num_basis -= 1
            else:
                flag = 0
    cv_basis_res=Basis_res(selected_num_basis, selected_num_basis_tdssq, min_test_error, basis_type, criterion, num_basis_list, test_err_list,test_se_list, test_tdssq_list, test_tdssq_se_list)
    return cv_basis_res



#%% MC estimate of the rho -- for tabular states
def MC_estimate_rho(state_space, cov_r,transition_function,reward_function, N=10, T = 1000, p=1, seed=0):
    def target_policy(Sijt):
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
    
