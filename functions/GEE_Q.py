# -*- coding: utf-8 -*-
"""
Fitted Q iteration
"""
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from collections import namedtuple
from sklearn.linear_model import LinearRegression
from sklearn.kernel_approximation import RBFSampler
# from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from functions.learn_env import NeuralNetworkStateModel
from copy import copy, deepcopy
# import statsmodels.api as sm
from functions.CustomGEE import CustomGEE
from sklearn.model_selection import train_test_split
import statsmodels
import scipy, sys
import functions.cov_struct as cov_structs
import prettytable  
# import datetime
import warnings
import gc
import time
# import itertools
warnings.filterwarnings("ignore")
#%%
class Cluster():
    def __init__(self, states, actions, next_states, rewards,
                 basis='rbf', num_basis=2, rbf_bw=1.0, include_bias=True,
                 RBFSampler_random_state=1,  
                 all_possible_states=None):
        '''
        one cluster
        
        # states shape: (i, t, p)
        # rewards shape: (i, t)
        # actions shape: (i, t)
        # all_possible_states: a list of possible values of states. For one hot vector transformation.
        '''
        self.states = states
        self.next_states = next_states
        self.rewards = rewards
        self.actions = actions
        self.nactions = np.unique(self.actions).shape[0]
        self.T = states.shape[1]
        self.N = states.shape[0]
        self.p_original = states.shape[2]
        self.basis= basis
        self.num_basis = num_basis
        self.include_bias = include_bias
        
        # # if no rbf basis, then just a linear term
        # if num_basis == 0:
        #     self.basis = 'polynomial'
        #     self.num_basis = 1
        # print('self.num_basis',self.num_basis)
        if self.basis == "rbf":
            self.featurize = RBFSampler(gamma=rbf_bw, random_state=RBFSampler_random_state, n_components=num_basis)

        elif self.basis == "polynomial":
            self.featurize = PolynomialFeatures(degree=self.num_basis, include_bias=include_bias)#(centered==False)

        elif self.basis == "one_hot":
            self.featurize = OneHotEncoder(categories='auto', sparse=False)

        else:
            raise ValueError('Invalid basis')
        
        # self.dim_values = dim_values
        self.all_possible_states = all_possible_states
        self.featurize_state()        
        self.rewards_stacked = rewards.flatten()
        self.actions_stacked = actions.flatten()
        self.p = self.transformed_states.shape[2]
        self.expected_next_states=None
        
        
    def featurize_state(self):
        """
        Returns the transformed representation for a state.
        
        Returns
        -------
        self.states_stacked: stacked states
        transformed_states_1: has the same number of dimensions as the input states
        transformed_states_2: has the same number of dimensions as the input next_states
        """
        states_stack = self.states.transpose(2, 0, 1).reshape(self.states.shape[2], -1).T
        next_states_stack = self.next_states.transpose(2, 0, 1).reshape(self.next_states.shape[2], -1).T
    
        combined_state = np.vstack([states_stack, next_states_stack])
        # print(combined_state.shape)
        if self.basis == "polynomial" or self.basis == "rbf":
            # print(self.basis)
            transformed_state = self.featurize.fit_transform(combined_state)
            # identification_metric = self.featurize.fit_transform(np.zeros([1,self.states.shape[2]]))
            state_shape = (self.states.shape[0], self.states.shape[1])
            self.states_stacked = transformed_state[:states_stack.shape[0]]
            self.next_states_stacked = transformed_state[states_stack.shape[0]:]
            self.transformed_states = self.states_stacked.reshape((*state_shape, -1))  
            self.transformed_next_states = self.next_states_stacked.reshape((*state_shape, -1))  
            if self.basis == "rbf":
                self.feature_names = ["rbf_" + str(i) for i in range(self.featurize.n_components)]
            else:
                self.feature_names = self.featurize.get_feature_names_out()

            
        elif self.basis == "one_hot":
            # print("one_hot")
            # Generate all possible states as tuples
            # Create a mapping of each state to a unique identifier
            self.state_to_id = {state: i for i, state in enumerate(self.all_possible_states)}

            # Function to convert states to their unique identifiers
            def states_to_ids(states):
                ids = [self.state_to_id[tuple(state)] for state in states]
                return np.array(ids)

            # Convert states and next_states to their unique identifiers
            states_ids = states_to_ids(states_stack)
            next_states_ids = states_to_ids(next_states_stack)

            # Initialize LabelEncoder and OneHotEncoder
            self.label_encoder = LabelEncoder()
            # self.one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

            # Fit and transform the unique identifiers
            self.label_encoder.fit(np.arange(len(self.all_possible_states)))
            self.featurize.fit(self.label_encoder.transform(np.arange(len(self.all_possible_states))).reshape(-1, 1))

            # Transform the states and next_states
            self.states_stacked = self.featurize.transform(self.label_encoder.transform(states_ids).reshape(-1, 1))
            self.next_states_stacked = self.featurize.transform(self.label_encoder.transform(next_states_ids).reshape(-1, 1))

            # Reshape the transformed states back to the original structure
            state_shape = self.states.shape[:2]
            self.transformed_states = self.states_stacked.reshape((*state_shape, -1))
            self.transformed_next_states = self.next_states_stacked.reshape((*state_shape, -1))
            self.feature_names = ['state_' + str(i) for i in range(self.transformed_states.shape[-1])]

        else:  # do nothing
            pass

#%%
class GEE_fittedQ():
    def __init__(self, cov_struct,
                 action_space =[0,1], 
                 rho = None,estimate_covariance_ornot=None,
                 gamma = 0.99, decor_combined_clusters = 0, verbose=1,
                 statsmodels_cov = None, statsmodel_maxiter=1,separate_corr=0,
                 Qmax = None, ridge_lambda=0,
                 evaluate_policy=None, combine_actions=0, optimal_GEE=0
                ):
        '''
        Initialization for the Fitted Q iteration with GEE Bellman Equation.
        Each action is associated with a Q model and a statsmodels_cov (if not statsmodels_cov is not None)

        Parameters
        ------------
        cluster: a list of instances from the Cluster object;
        gamma: discount rate of rewards
        decor_combined_clusters: whether or not to combine the data for 
        different actions to decorrelate. not recommand.
        rho: the parameters for the working correlation matrix. cov_struct.dep_params
        Distinguishing Between `statesmodels_cov` and `cov_struct`:
        
        1. `statesmodels_cov` (classes ending with "_sm" in functions.cov_struct):
           - Purpose: To learn the Q function using the GEE Bellman equation.
           - Implementation: Directly calls `statsmodel.api` for its faster implementation.
        
        2. `cov_struct` (classes not ending with "_sm"):
           - Purpose: Even though each action has its own Q function (and hence its own `statesmodels_cov`), the parameters related to the working correlation structure are estimated by pooling residuals from different actions. This pooled estimation is achieved using `cov_struct`.
           - Usage: Comes into play when `separate_corr` is set to 0. It estimates common parameters for various `statesmodels_cov`.

        3. if `separate_corr`, then each action has its own parameters in the working correlation structure.
        
        '''
        # common attributes for all clusters
        # self.T = cluster[0].T
        # self.N = cluster[0].N # not used
        # self.p = cluster[0].p
        self.decor_combined_clusters=decor_combined_clusters
        self.verbose = verbose
        self.statsmodels_api = 0 if statsmodels_cov is None else 1
        self.statsmodel_maxiter = statsmodel_maxiter
        self.separate_corr = separate_corr
        self.Qmax = Qmax
        # self.m = len(cluster)
        self.beta = None
        self.beta_old = None
        # self.cluster = cluster
        
        self.MC_rho = rho
        self.gamma = gamma
        self.action_space = action_space
        # self.cluster_size = [cluster[i].N for i in range(self.m)]
        self.evaluate_policy = evaluate_policy
        self.combine_actions = combine_actions
        self.optimal_GEE=optimal_GEE
        # transformed states and reward for each action
        # stacked rewards and stacked states
        # self.X_list = [None for x in range(self.m)] 
        # self.Y_list = [None for x in range(self.m)] 
        # self.Y_action_list = [[None for i in range(self.m)] for x in action_space]
        # self.X_action_list = [[None for i in range(self.m)] for x in action_space]

        # # for each action, the corresponding indices of the data in each cluster that the aciton is taken
        # self.action_indices = [[None for i in range(self.m)] for x in action_space]
        # # TD error for each cluster
        # self.TD_list = [np.empty(self.cluster_size[i]*self.T) for i in range(self.m)]
        
        
        # create a list of q function models, one for each action
        self.model = LinearRegression(fit_intercept=False)
        self.q_function_list = [deepcopy(self.model) for x in action_space] # a list of statsmodels.api.GEE objects later in the solve_GEE function if statsmodel.api is true
        # store the in-sample predicted q function; for stopping criterion
        self.q_func_preds = [None for x in action_space]
        self.q_func_preds_old = [None for x in action_space]
        # store the difference in the predicted Q function in all adjacent iterations; for stopping criterion
        self.Q_diff_list = []
        # the covariance of the fitted coefficient of q_function
        self.coef_sd =[None for x in action_space]
        self.infnan_indicator=0
    
        # for the `fit` function. allowing intermittent fitting
        self.converge = 0
        self.inner_converge=0
        self.iternum = 0
        self.run_time = 0
        self.cluster_old=None
        
        # the working correlation matrix
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        else:
            if not issubclass(cov_struct.__class__, cov_structs.CovStruct):
                raise ValueError("GEE: `cov_struct` must be a genmod "
                                 "cov_struct instance")
                
        self.cov_struct = cov_struct
        self.cov_struct.initialize(self)
        self.statsmodels_cov = statsmodels_cov
        
            
        if estimate_covariance_ornot == None:
            if isinstance(cov_struct, cov_structs.Independence) or self.statsmodels_api:
                self.estimate_covariance_ornot = 0
            else:
                self.estimate_covariance_ornot=1
        else:
            self.estimate_covariance_ornot = estimate_covariance_ornot
    
    def load_data(self, cluster, marginal_variance=1):
        # common attributes for all clusters
        self.T = cluster[0].T
        self.N = cluster[0].N # not used
        self.p = cluster[0].p
        self.p_original = cluster[0].p_original
        self.m = len(cluster)
        self.cluster = cluster
        self.cluster_T = [cluster[i].T for i in range(self.m)]
        self.cluster_size = [cluster[i].N for i in range(self.m)]
        # transformed states and reward for each action
        # stacked rewards and stacked states
        self.X_list = [None for x in range(self.m)] 
        self.Y_list = [None for x in range(self.m)] 
        self.expected_next_states_list = [None for x in range(self.m)] 
        self.Y_action_list = [[None for i in range(self.m)] for x in self.action_space]
        self.X_action_list = [[None for i in range(self.m)] for x in self.action_space]
        self.expected_next_states_action_list = [[None for i in range(self.m)] for x in self.action_space]
        
        # # for each action, the corresponding indices of the data in each cluster that the aciton is taken
        self.action_indices = [[None for i in range(self.m)] for x in self.action_space]
        # TD error for each cluster
        self.TD_list = [np.empty(self.cluster_size[i]*self.cluster_T[i]) for i in range(self.m)]
        
        
        # To predict Q function on a new state, we need to store the transformation of state variable
        self.basis = self.cluster[0].basis
        self.num_basis = self.cluster[0].num_basis
        self.featurize = self.cluster[0].featurize
        if self.basis == 'one_hot':
            self.label_encoder = self.cluster[0].label_encoder
            self.state_to_id = self.cluster[0].state_to_id
        self.feature_names =self.cluster[0].feature_names
        
    
        self.get_action_indices()
        self.marginal_variance = marginal_variance
        self.estimate_marginal_variance()
        
        if self.statsmodels_api:
            self.statsmodels_cov.setparams(self)
    
    def load_global_data(self, cluster, marginal_variance=1):
        if not isinstance(self.cov_struct, (cov_structs.Independence, cov_structs.Autoex_exsubject)):
            raise NotImplementedError("global TD update of the cov structure hasn't been implemented.")
        # common attributes for all clusters
        self.T = cluster[0].T
        self.N = cluster[0].N # not used
        self.p = cluster[0].p
        self.p_original = cluster[0].p_original
        self.global_m = len(cluster)
        self.global_cluster = cluster
        self.global_cluster_T = [cluster[i].T for i in range(self.global_m)]
        self.global_cluster_size = [cluster[i].N for i in range(self.global_m)]
        # transformed states and reward for each action
        # stacked rewards and stacked states
        # self.X_list = [None for x in range(self.m)] 
        # self.Y_list = [None for x in range(self.m)] 
        # self.expected_next_states_list = [None for x in range(self.m)] 
        # self.Y_action_list = [[None for i in range(self.m)] for x in self.action_space]
        # self.X_action_list = [[None for i in range(self.m)] for x in self.action_space]
        # self.expected_next_states_action_list = [[None for i in range(self.m)] for x in self.action_space]
        
        # # for each action, the corresponding indices of the data in each cluster that the aciton is taken
        self.global_action_indices = [[None for i in range(self.global_m)] for x in self.action_space]
        # TD error for each cluster
        self.global_TD_list = [np.empty(self.global_cluster_size[i]*self.global_cluster_T[i]) for i in range(self.global_m)]
        
        
        # To predict Q function on a new state, we need to store the transformation of state variable
        # self.basis = self.cluster[0].basis
        # self.num_basis = self.cluster[0].num_basis
        # self.featurize = self.cluster[0].featurize
        # if self.basis == 'one_hot':
        #     self.label_encoder = self.cluster[0].label_encoder
        #     self.state_to_id = self.cluster[0].state_to_id
        # self.feature_names =self.cluster[0].feature_names
        
    
        self.get_action_indices(is_global=True)
        self.global_marginal_variance = marginal_variance
        self.estimate_marginal_variance(is_global=True)
            

    def estimate_marginal_variance(self, is_global=False):
        '''
        Estimate the marginal variance: sufficient to know the marginal variance in proportion
        which is in proportion to the Phi(S)^2
        '''
        if is_global:
            if self.global_marginal_variance is None:                    
                self.global_marginal_variance = [np.linalg.norm(self.global_cluster[m].transformed_state, axis=1) ** 2 + self.ridge_lambda for m in range(self.global_m)]
            elif isinstance(self.global_marginal_variance, (int, float)):
                self.global_marginal_variance = [self.global_marginal_variance*np.ones(self.global_cluster[m].T *self.global_cluster[m].N) for m in range(self.global_m)]
        
        else:
            if self.marginal_variance is None:                    
                self.marginal_variance = [np.linalg.norm(self.cluster[m].transformed_state, axis=1) ** 2 + self.ridge_lambda for m in range(self.m)]
            elif isinstance(self.marginal_variance, (int, float)):
                self.marginal_variance = [self.marginal_variance*np.ones(self.cluster[m].T *self.cluster[m].N) for m in range(self.m)]
        
            
    def V_generator(self, m):
        return self.cov_struct.covariance_matrix(m)


    def V_invsqrt_generator(self, action_index=None):
        for m in range(self.m):
            yield self.cov_struct.V_invsqrt(m, action_index)


    def get_action_indices(self, is_global=False):
        if is_global:
            if self.combine_actions:
                self.global_action_indices[0] = [list(range(self.global_cluster[m].N*self.global_cluster[m].T)) for m in range(self.global_m)]
                return
            
            for action_index, a in enumerate(self.action_space):
                self.global_action_indices[action_index] = [np.where(self.global_cluster[m].actions_stacked == a)[0] for m in range(self.global_m)]
                
        else:
            if self.combine_actions:
                self.action_indices[0] = [list(range(self.cluster[m].N*self.cluster[m].T)) for m in range(self.m)]
                return
            
            for action_index, a in enumerate(self.action_space):
                self.action_indices[action_index] = [np.where(self.cluster[m].actions_stacked == a)[0] for m in range(self.m)]
    
        

    def transform_state_action_long_vec(self, S, A):
        # Assuming S and A are your states and actions numpy arrays
        N, p = S.shape  # Get the shape of your states array
        
        # Initialize an array of zeros for the transformed data
        transformed_data = np.zeros((N, len(self.action_space) * self.p))
        A = np.atleast_1d(A).flatten()
        # Iterate over each state-action pair
        for i in range(N):
            action = A[i]  # Assuming actions are stored in a 2D array with a single column
            action_index = np.where(self.action_space == action)[0][0]  # Find the index of the action in the unique list of actions
        
            # Fill the corresponding segment for the action with the state, all other segments remain zero
            transformed_data[i, action_index*p : (action_index+1)*p] = S[i]
        return transformed_data

        
    
    def create_design_matrix_cluster(self):
        if self.combine_actions:
            for m in range(self.m):
                self.Y_action_list[0][m] = self.cluster[m].rewards_stacked 
                self.X_action_list[0][m] = self.transform_state_action_long_vec(self.cluster[m].states_stacked,
                                                                                self.cluster[m].actions_stacked)
            return 
        
        for action_index, _ in enumerate(self.action_space):
            for m in range(self.m):
                rewards = self.cluster[m].rewards_stacked[self.action_indices[action_index][m]]
                x_onestep = self.cluster[m].states_stacked[self.action_indices[action_index][m]]
    
                self.Y_action_list[action_index][m] = rewards
                self.X_action_list[action_index][m] = x_onestep



    def decorrelated_XY_notused(self):
        if self.verbose:
            print('decor')
        if self.check_fitted:
            if self.estimate_covariance_ornot:
                # combined decorrelation for different actions
                if self.decor_combined_clusters:
                    for m, (V, _) in enumerate(self.V_generator()):
                        # Eigenvalue decomposition
                        eigvals, eigvecs = np.linalg.eigh(V)
                        V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        
                        self.Y_list[m] = V_invsqrt @ self.cluster[m].rewards_stacked + self.gamma * V_invsqrt @ self.Qmax[m]
                        self.X_list[m] = V_invsqrt @ self.cluster[m].states_stacked
                        
                    for action_index, _ in enumerate(self.action_space):
                        self.X_action_list[action_index] = [self.X_list[m][self.action_indices[action_index][m]] for m in range(self.m)]
                        self.Y_action_list[action_index] = [self.Y_list[m][self.action_indices[action_index][m]] for m in range(self.m)]
                
                
                # separate decorrelation for different actions
                else:
                    self.Y_list = [self.cluster[m].rewards_stacked + self.gamma * self.Qmax[m] for m in range(self.m)]
                    self.X_list = [self.cluster[m].states_stacked for m in range(self.m)]
                    
                    for action_index, _ in enumerate(self.action_space):
                        for m in range(self.m):
                            selected_indices = self.action_indices[action_index][m]
                            
                            b = np.hstack([self.X_list[m][selected_indices], self.Y_list[m][selected_indices].reshape(-1,1)])
                            x = self.cov_struct.decorrelate(b, m, action_index)
                            self.X_action_list[action_index][m] = x[:, :-1]
                            self.Y_action_list[action_index][m] = x[:, -1]  
            # no decorrelation
            else:
                for m in range(self.m):
                    self.Y_list[m] = self.cluster[m].rewards_stacked + self.gamma * self.Qmax[m]
                    self.X_list[m] = self.cluster[m].states_stacked
                for action_index, _ in enumerate(self.action_space):
                    self.X_action_list[action_index] = [self.X_list[m][self.action_indices[action_index][m]] for m in range(self.m)]
                    self.Y_action_list[action_index] = [self.Y_list[m][self.action_indices[action_index][m]] for m in range(self.m)]
            
        else:
            self.create_design_matrix_cluster()


    def calculate_decorrelation(self, m):
        if self.decor_combined_clusters and (not self.statsmodels_api):
            V, _ = next(self.V_generator(m))
            eigvals, eigvecs = np.linalg.eigh(V)
            V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
    
            self.Y_list[m] = V_invsqrt @ self.cluster[m].rewards_stacked + self.gamma * V_invsqrt @ self.Qmax[m]
            self.X_list[m] = V_invsqrt @ self.cluster[m].states_stacked
        else:
            # print('not combined decor')
            self.Y_list[m] = self.cluster[m].rewards_stacked + self.gamma * self.Qmax[m]
            self.X_list[m] = self.cluster[m].states_stacked


    def calculate_decor_XY(self, action_index):
        for m in range(self.m):
            selected_indices = self.action_indices[action_index][m]
            if self.estimate_covariance_ornot and not self.decor_combined_clusters and not self.statsmodels_api:
                b = np.hstack([self.X_list[m][selected_indices], self.Y_list[m][selected_indices].reshape(-1,1)])
                x = self.cov_struct.decorrelate(b, m, action_index)
                self.X_action_list[action_index][m] = x[:, :-1]
                self.Y_action_list[action_index][m] = x[:, -1]  
            else:
                # print('not decor')
                self.X_action_list[action_index][m] = self.X_list[m][selected_indices]
                self.Y_action_list[action_index][m] = self.Y_list[m][selected_indices]
                if self.expected_next_states_list[m] is not None:
                    self.expected_next_states_action_list[action_index][m] = self.expected_next_states_list[m][selected_indices]
                else:
                    self.expected_next_states_action_list[action_index][m] = None
                    
                    
    def decorrelated_XY(self):
        if self.check_fitted:
            if self.estimate_covariance_ornot and not self.statsmodels_api:
                for m in range(self.m):
                    self.calculate_decorrelation(m)
    
            else:
                # print('not combined decor')
                for m in range(self.m):
                    self.Y_list[m] = self.cluster[m].rewards_stacked + self.gamma * self.Qmax[m]
                    if self.combine_actions:
                        self.X_list[m] = self.transform_state_action_long_vec(self.cluster[m].states_stacked, self.cluster[m].actions_stacked)
                    else:
                        self.X_list[m] = self.cluster[m].states_stacked
                    self.expected_next_states_list[m] = self.cluster[m].expected_next_states
            for action_index, _ in enumerate(self.action_space):
                self.calculate_decor_XY(action_index)
                if self.combine_actions:
                    return
                
    
        else:
            self.create_design_matrix_cluster()


    # Function to check if the model has been fitted
    # def is_model_fitted(self):
    #     try:
    #         # If coef_ attribute exists, the model is considered fitted
    #         for model in self.q_function_list:
    #             _ = model.coef_
    #         return True
    #     except AttributeError:
    #         # If coef_ attribute does not exist, the model is not fitted
    #         return False
    def is_model_fitted(self):
        if isinstance(self.q_function_list[0], LinearRegression):
            try:
                # Check for the existence of coefficients
                for model in self.q_function_list:
                    _ = model.coef_
                return True
            except AttributeError:
                return False
        elif isinstance(self.q_function_list[0], statsmodels.genmod.generalized_estimating_equations.GEEResultsWrapper):
            try:
                # Check for the existence of fitted parameters
                for model in self.q_function_list:
                    _ = model.params
                    if self.combine_actions:
                        return
                return True
            except AttributeError:
                return False
        else:
            return False  # If the model is neither, it's not recognized

        
    def solve_GEE(self, calculate_TD=True):
        # calculate the max_a Q(s,a)
        if self.check_fitted:
            self.q_function_list_old = [copy(self.q_function_list[i]) for i in range(len(self.action_space))]
            if self.iternum ==0 and self.cluster_old is not None:
                cluster_data = self.cluster_old 
            else:
                cluster_data = self.cluster
            
            if self.evaluate_policy is None:
                # learning
                if not self.combine_actions:
                    self.Qmax = [np.max(np.vstack([self.q_function_list_old[a].predict(cluster_data[m].next_states_stacked)
                                                         for a in range(len(self.action_space))]), axis=0)
                                      for m in range(self.m)]
                    if self.global_TD:
                        self.global_Qmax = [np.max(np.vstack([self.q_function_list_old[a].predict(self.global_cluster[m].next_states_stacked)
                                                             for a in range(len(self.action_space))]), axis=0)
                                          for m in range(self.global_m)] 
                else:
                    self.Qmax = [np.max(np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(cluster_data[m].next_states_stacked, np.repeat(a, cluster_data[m].next_states_stacked.shape[0])))
                                                         for a in range(len(self.action_space))]), axis=0)
                                      for m in range(self.m)]
                    if self.global_TD:
                        self.global_Qmax =  [np.max(np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(self.global_cluster[m].next_states_stacked, np.repeat(a, self.global_cluster[m].next_states_stacked.shape[0])))
                                                             for a in range(len(self.action_space))]), axis=0)
                                          for m in range(self.global_m)]

            else:
                # evaluation
                if not self.combine_actions:
                    self.Qmax = [np.vstack([self.q_function_list_old[self.action_space.index(int(self.evaluate_policy(cluster_data.states_stacked[i].reshape(1, -1))))].predict(cluster_data.next_states_stacked[i].reshape(1, -1))
                                                         for i in range(cluster_data[m].states_stacked.shape[0])]).flatten()
                                      for m in range(self.m)]
                    if self.global_TD:
                        self.global_Qmax = [np.vstack([self.q_function_list_old[self.action_space.index(int(self.evaluate_policy(self.global_cluster.states_stacked[i].reshape(1, -1))))].predict(self.global_cluster.next_states_stacked[i].reshape(1, -1))
                                                             for i in range(self.global_cluster[m].states_stacked.shape[0])]).flatten()
                                          for m in range(self.global_m)]

                else:
                    self.Qmax = [np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(cluster_data.next_states_stacked[i].reshape(1, -1)), self.action_space.index(int(self.evaluate_policy(cluster_data.states_stacked[i].reshape(1, -1)))))
                                                         for i in range(cluster_data[m].states_stacked.shape[0])]).flatten()
                                      for m in range(self.m)]
                    if self.global_TD:
                        self.global_Qmax = [np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(self.global_cluster.next_states_stacked[i].reshape(1, -1)), self.action_space.index(int(self.evaluate_policy(self.global_cluster.states_stacked[i].reshape(1, -1)))))
                                                             for i in range(self.global_cluster[m].states_stacked.shape[0])]).flatten()
                                          for m in range(self.global_m)]

        else:
            if self.Qmax is None:
                self.Qmax = [np.zeros(self.cluster_size[m] *self.cluster_T[m]) for m in range(self.m)] 
            if self.global_TD:
                self.global_Qmax = [np.zeros(self.global_cluster_size[m] *self.global_cluster_T[m]) for m in range(self.global_m)] 

        self.decorrelated_XY()
        
        if self.statsmodels_api and self.check_fitted:
            if not self.separate_corr:
                self.statsmodels_cov.assign_params(self.cov_struct.dep_params)
            for action_index, _ in enumerate(self.action_space):
                self.statsmodels_cov.action_index = action_index
                groups = np.repeat(np.arange(self.m),[len(self.action_indices[action_index][i]) for i in range(self.m)])
                self.q_function_list[action_index] = CustomGEE(np.hstack(self.Y_action_list[action_index]),
                                                            np.vstack(self.X_action_list[action_index]), groups = groups,
                                                            cov_struct=self.statsmodels_cov)
                try:
                    self.q_function_list[action_index] = self.q_function_list[action_index].fit(maxiter = self.statsmodel_maxiter)
                
                except ValueError as e:
                    if str(e) == "The first guess on the deviance function returned a nan.  This could be a boundary  problem and should be reported.":
                        # Save the data that was used for fitting the GEE model
                        np.save('X_action_data.npy', np.vstack(self.X_action_list[action_index]))  # Save X data
                        np.save('Y_action_data.npy', np.hstack(self.Y_action_list[action_index]))  # Save Y data
                
                        # Assuming self.statsmodels_cov.dep_params is a structure that can be directly saved
                        # If it's more complex, consider converting it to a suitable format before saving
                        np.save('dep_params.npy', self.statsmodels_cov.dep_params)  # Save dep_params
                
                        # Handling the error or taking any fallback action
                        print("Error: The first guess on the deviance function returned a nan. Data has been saved for debugging.")
                        self.q_function_list = self.q_function_list_old
                        self.infnan_indicator = 1
                        break
                    elif str(e) == "array must not contain infs or NaNs" or str(e)== "Input y contains infinity or a value too large for dtype('float64').":
                        print("Error: The array contains infs or NaNs")
                        self.q_function_list = self.q_function_list_old
                        self.infnan_indicator=1
                        break
                    else:
                        # If it's a different ValueError, you might want to re-raise it
                        raise
                if self.combine_actions:
                    break

                
                # if self.verbose:
                #     print('statsmodel cov', self.q_function_list[action_index].cov_struct.dep_params)
        else:
            for action_index, _ in enumerate(self.action_space):
                X = np.vstack(self.X_action_list[action_index])
                Y = np.hstack(self.Y_action_list[action_index])
                try:
                    self.q_function_list[action_index].fit(X, Y)
                except ValueError as e:
                    if str(e) == "The first guess on the deviance function returned a nan.  This could be a boundary  problem and should be reported.":
                        # Save the data that was used for fitting the GEE model
                        np.save('X_action_data.npy', np.vstack(self.X_action_list[action_index]))  # Save X data
                        np.save('Y_action_data.npy', np.hstack(self.Y_action_list[action_index]))  # Save Y data
                
                        # Assuming self.statsmodels_cov.dep_params is a structure that can be directly saved
                        # If it's more complex, consider converting it to a suitable format before saving
                        np.save('dep_params.npy', self.statsmodels_cov.dep_params)  # Save dep_params
                
                        # Handling the error or taking any fallback action
                        print("Error: The first guess on the deviance function returned a nan. Data has been saved for debugging.")
                        self.q_function_list = self.q_function_list_old
                        self.infnan_indicator = 1
                        break
                    elif str(e) == "array must not contain infs or NaNs" or str(e)== "Input y contains infinity or a value too large for dtype('float64').":
                        print("Error: The array contains infs or NaNs")
                        self.q_function_list = self.q_function_list_old
                        self.infnan_indicator=1
                        break
                    elif "NaN" in str(e) or "infs" in str(e) or "infinity" in str(e):
                        print("Error: The array contains infs or NaNs")
                        self.q_function_list = self.q_function_list_old
                        self.infnan_indicator=1
                        break
                    else:
                        # If it's a different ValueError, you might want to re-raise it
                        raise
                
                if self.combine_actions:
                    break
                # self.q_function_list[action_index].fit(X, Y)
            
        # calculate the TD error to estimate the correlation
        if calculate_TD:
            # print('begin calculate td')
            if self.check_fitted:
                if self.cluster_old is None:
                    self.q_func_preds_old = deepcopy(self.q_func_preds)
                err = 0
                
            for action_index, a in enumerate(self.action_space):
                
                self.q_func_preds[action_index] = []
                if self.cluster_old is not None:
                    self.q_func_preds_old[action_index] = []
                
                if self.global_TD:
                    ## if calculate TD for all batches
                    selected_indices = self.global_action_indices[action_index]
                    cluster = self.global_cluster
                    TD_list = self.global_TD_list
                    Qmax = self.global_Qmax
                else:
                    ## if TD is calculated only for current batch
                    selected_indices = self.action_indices[action_index]
                    cluster = self.cluster
                    TD_list = self.TD_list
                    Qmax = self.Qmax
                    
                for m in range(len(cluster)):
                    if selected_indices[m] is not None:
                        if self.combine_actions:
                            states_stacked = self.transform_state_action_long_vec(cluster[m].states_stacked, cluster[m].actions_stacked)
                        else:
                            states_stacked = cluster[m].states_stacked[selected_indices[m]]

                        prediction = self.q_function_list[action_index].predict(states_stacked)
                        if self.check_fitted and self.cluster_old is not None: 
                            if self.iternum==0 and self.cluster_old is not None:
                                states_old =  self.cluster_old[m].states_stacked[selected_indices[m]]
                            else:
                                states_old = states_stacked
                            prediction_old = self.q_function_list_old[action_index].predict(states_old)
                    else:
                        prediction = None  # Replace None with the appropriate default value or handling logic.
                        if self.check_fitted: prediction_old = None
                    self.q_func_preds[action_index].append(prediction)
                    if self.check_fitted and self.cluster_old is not None: 
                        self.q_func_preds_old[action_index].append(prediction_old)

                if self.check_fitted:
                    err += np.sum((np.hstack(self.q_func_preds[action_index]) - np.hstack(self.q_func_preds_old[action_index]))**2)
                
                for m in range(len(cluster)):
                    if selected_indices[m] is not None:
                        TD_list[m][selected_indices[m]] = cluster[m].rewards_stacked[selected_indices[m]] + \
                            self.gamma * Qmax[m][selected_indices[m]] - \
                            self.q_func_preds[action_index][m]
                if self.combine_actions:
                        break
                # ## if calculate TD for all batches
                # if self.global_TD:
                    
                #     selected_indices = self.global_action_indices[action_index]
                #     for m in range(self.global_m):
                #         if selected_indices[m] is not None:
                #             if self.combine_actions:
                #                 states_stacked = self.transform_state_action_long_vec(self.global_cluster[m].states_stacked, self.global_cluster[m].actions_stacked)
                #             else:
                #                 states_stacked = self.global_cluster[m].states_stacked[selected_indices[m]]
    
                #             prediction = self.q_function_list[action_index].predict(states_stacked)
                #             if self.check_fitted and self.cluster_old is not None: 
                #                 if self.iternum==0 and self.cluster_old is not None:
                #                     states_old =  self.cluster_old[m].states_stacked[selected_indices[m]]
                #                 else:
                #                     states_old = states_stacked
                #                 prediction_old = self.q_function_list_old[action_index].predict(states_old)
                #         else:
                #             prediction = None  # Replace None with the appropriate default value or handling logic.
                #             if self.check_fitted: prediction_old = None
                #         self.q_func_preds[action_index].append(prediction)
                #         if self.check_fitted and self.cluster_old is not None: 
                #             self.q_func_preds_old[action_index].append(prediction_old)
    
                #     if self.check_fitted:
                #         err += np.sum((np.hstack(self.q_func_preds[action_index]) - np.hstack(self.q_func_preds_old[action_index]))**2)
                    
                #     for m in range(self.global_m):
                #         if selected_indices[m] is not None:
                #             self.global_TD_list[m][selected_indices[m]] = self.global_cluster[m].rewards_stacked[selected_indices[m]] + \
                #                 self.gamma * self.global_Qmax[m][selected_indices[m]] - \
                #                 self.q_func_preds[action_index][m]
                
                #     if self.combine_actions:
                #         break
                    
                # else:
                #     ## if TD is calculated only for current batch
                #     selected_indices = self.action_indices[action_index]
                #     for m in range(self.m):
                #         if selected_indices[m] is not None:
                #             if self.combine_actions:
                #                 states_stacked = self.transform_state_action_long_vec(self.cluster[m].states_stacked, self.cluster[m].actions_stacked)
                #             else:
                #                 states_stacked = self.cluster[m].states_stacked[selected_indices[m]]
    
                #             prediction = self.q_function_list[action_index].predict(states_stacked)
                #             if self.check_fitted and self.cluster_old is not None: 
                #                 if self.iternum==0 and self.cluster_old is not None:
                #                     states_old =  self.cluster_old[m].states_stacked[selected_indices[m]]
                #                 else:
                #                     states_old = states_stacked
                #                 prediction_old = self.q_function_list_old[action_index].predict(states_old)
                #         else:
                #             prediction = None  # Replace None with the appropriate default value or handling logic.
                #             if self.check_fitted: prediction_old = None
                #         self.q_func_preds[action_index].append(prediction)
                #         if self.check_fitted and self.cluster_old is not None: 
                #             self.q_func_preds_old[action_index].append(prediction_old)
    
                #     if self.check_fitted:
                #         err += np.sum((np.hstack(self.q_func_preds[action_index]) - np.hstack(self.q_func_preds_old[action_index]))**2)
                    
                #     for m in range(self.m):
                #         if selected_indices[m] is not None:
                #             self.TD_list[m][selected_indices[m]] = self.cluster[m].rewards_stacked[selected_indices[m]] + \
                #                 self.gamma * self.Qmax[m][selected_indices[m]] - \
                #                 self.q_func_preds[action_index][m]
                
                #     if self.combine_actions:
                #         break
            if self.global_TD:
                self.TDsqlist.append(np.mean([x**2 for sublist in self.global_TD_list for x in sublist]))
            else:
                self.TDsqlist.append(np.mean([x**2 for sublist in self.TD_list for x in sublist]))

            if self.verbose:
                print("TD^2", self.TDsqlist[-1])
            if self.check_fitted:
                self.Q_diff_list.append(err)

            
    def fit(self,cluster,num_batches=1, max_iter=100, batch_iter=100, stopping_threshold=1e-5, 
            verbose=None, convergence_criterion="response", accelerate_method = "batch_processing",
            estimate_beta_variance=True, global_TD=False):
        """
        solve GEE bellman equation

        Parameters
        ----------
        cluster : a list of cluster objects.
        num_batches : how many batches to split the cluster data along temporal axis. The default is 1.
        max_iter : global maximal iteration numbers. The default is 100.
        batch_iter : TYPE, optional
            DESCRIPTION. The default is 100.
        stopping_threshold : TYPE, optional
            DESCRIPTION. The default is 1e-5.
        verbose : TYPE, optional
            DESCRIPTION. The default is None.
        convergence_criterion : TYPE, optional
            DESCRIPTION. The default is "response".
        accelerate_method : "batch_processing" or "split_clusters"; suppose we have 5 clusters, i.e., the `cluster` would be a list of len 5, and the num_batches=3, then
            1). "batch_processing": the output from `self.split_cluster_data`, i.e., `split_clusters` would be like [[c11, c21, ..., c51], [c12, c22, ..., c52], [c13, ..., c53]]. 
                The training process would be firstly loop over [c11, c21, ..., c51] (for l in range(batch_iter)), then [c12, c22, ..., c52], then the outermost [] (for global_iter in range(max_iter)).
            2). "split_clusters": `split_clusters` would be like [[c11, c21, ..., c51, c12, c22, ..., c52, c13, ..., c53]]. There is no need to set global_iter > 1.
        """
        start_time = time.time()
        self.check_fitted = self.is_model_fitted()
        # Splitting the dataset into batches
        if accelerate_method == "batch_processing" and num_batches > 1:
            self.global_TD = global_TD
        ## TODO: is this correct?
        else:
            self.global_TD=0
            
        if self.global_TD:
            self.load_global_data(cluster)
            
        split_clusters = self.split_cluster_data(cluster, num_batches, accelerate_method = accelerate_method)
        
        previous_global_Q = None
        global_iter = 0
        self.TDsqlist = []
        
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print('num_batches', num_batches)
            sys.stdout.flush()
        
        if accelerate_method == "split_clusters" or num_batches == 1:
            max_iter=1
            
        for global_iter in range(max_iter):
            if verbose: 
                print('===== global_iter', global_iter, ' ======')
                sys.stdout.flush()
            for split_cluster in split_clusters:
                if verbose:
                    print('split index', split_clusters.index(split_cluster))
                self.load_data(split_cluster)
    
                for l in range(batch_iter):
                    if verbose: 
                        print('===== l', l ,'self.iternum',self.iternum,' ======')
                        sys.stdout.flush()
                    self.solve_GEE()
                    self.cov_struct.update()
                    if self.optimal_GEE:
                        self.update_expected_state_action()
                    self.iternum += 1
                    if verbose:
                        print('cov_struct.dep_params', self.cov_struct.dep_params)
                    # convergence or not
                    if self.check_fitted:
                        if convergence_criterion == "response":
                            if not self.combine_actions:
                                flat_array = np.array([element for sublist1 in self.q_func_preds for sublist2 in sublist1 for element in sublist2])
                            else:
                                flat_array = np.array([element for sublist1 in self.q_func_preds[0] for element in sublist1])
                  
                            sum_of_squares = np.sum(flat_array**2)
                            norm_ratio = self.Q_diff_list[-1] / sum_of_squares
                            if norm_ratio < stopping_threshold:
                                if verbose:
                                    print('inner converge')
                                    self.inner_converge=1
                                break
                        elif convergence_criterion == "coef":
                            diff = 0
                            coef_sumsq = 0
                            for action_index, _ in enumerate(self.action_space):
                                coef_old = self.q_function_list_old[action_index].params if self.statsmodels_api else self.q_function_list_old[action_index].coef_
                                coef_new = self.q_function_list[action_index].params if self.statsmodels_api else self.q_function_list[action_index].coef_
                                diff += np.sum((coef_old - coef_new)**2)
                                coef_sumsq += np.sum(coef_old**2)
                            if diff/coef_sumsq < stopping_threshold:
                                if verbose:
                                    print('inner converge')
                                    self.inner_converge=1
                                break
                    
                    self.check_fitted=True
            self.global_iter = global_iter
            # Check global convergence
            if previous_global_Q is not None:
                err = self.check_global_convergence(previous_global_Q)
                if verbose:
                    print('global loss', err)
                    sys.stdout.flush()
                if err < stopping_threshold:
                    self.converge = 1
                    if verbose: print('converged')
                    break
                
            previous_global_Q = deepcopy(self.q_func_preds)
        
        # Finalize and return results
        end_time = time.time()
        running_time = end_time - start_time
        if verbose: print(f"The running time is {running_time} seconds.")
        self.run_time += running_time
        if self.inner_converge==1 and max_iter==1:
            self.converge=1
        if self.converge == 0:
            warnings.warn("GEE not converge")
        if estimate_beta_variance:
            self.estimate_beta_variance()
        self.dep_params = self.cov_struct.dep_params
        GEE_fittedQ_res = namedtuple("GEE_fittedQ_res", ["beta", "cov", 'converge', 'iter_num', 'running_time'])
        return GEE_fittedQ_res(self.q_function_list, self.coef_sd, self.converge, self.iternum, running_time)


    def check_global_convergence(self, previous_global_Q):
        err = 0
        for action_index, _ in enumerate(self.action_space):
            err += np.sum((np.hstack(self.q_func_preds[action_index]) - np.hstack(previous_global_Q[action_index]))**2)
            if self.combine_actions:
                break
            
        if not self.combine_actions:
            flat_array = np.array([element for sublist1 in self.q_func_preds for sublist2 in sublist1 for element in sublist2])
        else:
            flat_array = np.array([element for sublist1 in self.q_func_preds[0] for element in sublist1])

        sum_of_squares = np.sum(flat_array**2)
        return err/sum_of_squares 
    
    
    def update_expected_state_action(self):
        def transform_cluster_data(s):
            states = s.transpose(2, 0, 1).reshape(s.shape[2], -1).T
            return states
        state_model = NeuralNetworkStateModel(state_dim=self.p_original, action_dim=1, state_type ="continous",
                                              action_type ="continous",
                                              layer_sizes =[self.p_original+1,self.p*len(self.action_space)])
            
        states = np.concatenate([transform_cluster_data(c.states) for c in self.cluster], axis=0)
        actions = np.concatenate([c.actions_stacked for c in self.cluster])
        transformed_next_states = np.concatenate([self.transform_state_action_long_vec(c.next_states_stacked, transform_cluster_data(self.predict_action(c.next_states))) for c in self.cluster], axis=0)

        # Combine states and actions into features X
        X = np.concatenate((states, actions.reshape(-1,1)), axis=1)
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, transformed_next_states, test_size=0.2)
        
        states_train = X_train[:, :states.shape[1]]  # Extract the first columns corresponding to states
        actions_train = X_train[:, states.shape[1]:]  # Ext
        states_val = X_val[:, :states.shape[1]]  # Extract the first columns corresponding to states
        actions_val = X_val[:, states.shape[1]:]  # Ext
        validation_data = [states_val, actions_val, y_val, y_val]

        state_model.fit(states_train, actions_train, y_train, validation_data, num_epochs=10000)
    
        for c in self.cluster:
            c.expected_next_states = state_model.predict(transform_cluster_data(c.states), c.actions_stacked.reshape(-1,1))
            
                  



    def split_cluster_data(self, cluster, num_batches, accelerate_method):
        if num_batches>1:
            split_clusters = [[] for i in range(num_batches)]
            for clust in cluster:
                T = clust.states.shape[1]  # Total number of time points
                batch_size = T // num_batches
    
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size if i < num_batches - 1 else T
    
                    # Create a new Cluster object for each batch
                    batch_data = {
                        "states": clust.states[:, start_idx:end_idx, :],
                        "actions": clust.actions[:, start_idx:end_idx],
                        "rewards": clust.rewards[:, start_idx:end_idx],
                        "next_states": clust.next_states[:, start_idx:end_idx, :]
                    }
                    split_clusters[i].append(Cluster(**batch_data, basis=clust.basis, 
                                                  num_basis=clust.num_basis, include_bias=clust.include_bias,
                                                  all_possible_states=clust.all_possible_states))
            if accelerate_method == "split_clusters":
                split_clusters = [[item for sub_list in split_clusters for item in sub_list]]
        else:
            split_clusters = [cluster]
        return split_clusters
    
    def estimate_beta_variance(self):
        for action_index, _ in enumerate(self.action_space):
            # if self.statsmodels_api and self.infnan_indicator==0 and self.iternum>0:
            #     self.coef_sd[action_index] = self.q_function_list[action_index].bse
            if type(self.q_function_list[action_index]) == LinearRegression:
            # if (self.statsmodels_api and self.infnan_indicator==1 and self.iternum<2) or not self.statsmodels_api:
                I1 = 0
                I0 = 0
                
                for m in range(self.m):
                    decor_X = self.X_action_list[action_index][m]
                    decor_Y = self.Y_action_list[action_index][m] - self.q_function_list[action_index].predict(decor_X)
                    # cov_Y =  decor_Y.reshape([-1,1]) @  decor_Y.reshape([1,-1])
                    X_T_cov_Y_X = decor_X.T @ decor_Y.reshape([-1,1]) @  decor_Y.reshape([1,-1]) @ decor_X
                        
                    I1 += X_T_cov_Y_X
                    I0 += decor_X.T @ decor_X
                    # Your code that might raise a LinAlgError
                    try:
                        I0_inv = np.linalg.inv(I0)
                    except np.linalg.LinAlgError as e:
                        print("Singular matrix error:", e)
                        # Add 0.001 to the diagonal of I0 to make it non-singular
                        I0 = I0 + 0.001 * np.eye(I0.shape[0])  # Add 0.001 to the diagonal elements
                        I0_inv = np.linalg.inv(I0)
                    # I0_inv = np.linalg.inv(I0)
                    
                I0_inv = np.linalg.inv(I0)
                var = np.diag(I0_inv @ I1 @ I0_inv)
                var = np.where(np.abs(var) < 1e-15, 0, var)
                self.coef_sd[action_index] = np.sqrt(var)
            else:
                self.coef_sd[action_index] = self.q_function_list[action_index].bse
            if self.combine_actions:
                return
    

    def significance(self, p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        elif p_val < 0.1:
            return '.'
        else:
            return ''

    def summary(self):
        t_model = prettytable.PrettyTable()
        t_model.field_names = ["Model Details", "Value"]
        t_model.add_row(["Num of clusters", self.m])
        t_model.add_row(["Max Num of subjects in each cluster", np.max(self.cluster_size)])
        t_model.add_row(["Min Num of subjects in each cluster", np.min(self.cluster_size)])
        t_model.add_row(["Ave Num of subjects in each cluster", np.mean(self.cluster_size)])
        t_model.add_row(["Ave Num of observations for each subject", np.mean(self.cluster_T)])
        t_model.add_row(["Gamma discounted factor", self.gamma])
        t_model.add_row(["Correlation structure", self.cov_struct.__class__.__name__])
        t_model.add_row(["Basis", self.basis])
        if self.basis == "polynomial":
            t_model.add_row(["Degree", self.num_basis])
        elif self.basis == 'rbf':
            t_model.add_row(["Num of component, Band width", f"{self.featurize.n_components}, {self.featurize.gamma}"])
        print(t_model)

    
        print(f'\nConverge: {self.converge}')
        print(f'Num of iterations: {self.iternum}')
        print(f'Running time: {self.run_time}\n')
    
        for action_index, a in enumerate(self.action_space):
            if type(self.q_function_list[action_index]) == LinearRegression:
            # if (self.statsmodels_api and self.infnan_indicator==1 and self.iternum<2) or not self.statsmodels_api:
                coef = self.q_function_list[action_index].coef_
            else:
                coef = self.q_function_list[action_index].params #if self.statsmodels_api else self.q_function_list[action_index].coef_
            sd = self.coef_sd[action_index]
            print(f'Action {a} coefficients:')
            t = prettytable.PrettyTable(border=False, hrules=prettytable.NONE)
            t.field_names = ['Coef', 'Value', 'Std Err', 'T', 'P>|t|', '[0.025', '0.975]']
            t.vertical_char = ' '
            t.junction_char = ' '
    
            if self.combine_actions:
                feature_name = [name + "_"+ str(a) for a in self.action_space for name in self.feature_names]
            else:
                feature_name = self.feature_names
            for i in range(len(coef)):
                t_val = coef[i] / sd[i]
                p_val = scipy.stats.t.sf(np.abs(t_val),  np.mean(self.cluster_T)*np.sum(self.cluster_size)-1)*2  # two-tailed pvalue = Prob(abs(t)>tt)
                ci_low = coef[i] - scipy.stats.t.ppf(0.975,  np.mean(self.cluster_T)*np.sum(self.cluster_size)-1)*sd[i]
                ci_high = coef[i] + scipy.stats.t.ppf(0.975,  np.mean(self.cluster_T)*np.sum(self.cluster_size)-1)*sd[i]
    
                signif = self.significance(p_val)
                
                predictor_name = f'{signif}{feature_name[i]}'.rjust(15)  # right-align with padding
                t.add_row([predictor_name, f'{coef[i]:.4f}', f'{sd[i]:.4f}', f'{t_val:.4f}', f'{p_val:.4f}', f'{ci_low:.4f}', f'{ci_high:.4f}'])
  
            print(t)
            if self.combine_actions:
                break
    
        print('\nEstimated Correlation:')
        print(self.cov_struct.dep_params)
        print('===============================================')


    def featurize_state(self, state):
        '''
        Transform state for prediction

        Parameters
        ----------
        state : one single state or a states for multiple stage. should be of shape [N, T, p]

        '''
        
        states_stack = state.transpose(2, 0, 1).reshape(state.shape[2], -1).T
        state_shape = (state.shape[0], state.shape[1])
        if self.basis == "polynomial" or self.basis == "rbf":
            transformed_state = self.featurize.fit_transform(states_stack)
            observed_mask=None
        elif self.basis == "one_hot":
            def states_to_ids(states):
                ids = []
                observed_mask = []
                for state in states:
                    state_tuple = tuple(state)
                    if state_tuple in self.state_to_id:
                        ids.append(self.state_to_id[state_tuple])
                        observed_mask.append(True)
                    else:
                        ids.append(-1)  # Use -1 for unobserved states
                        observed_mask.append(False)
                return np.array(ids), np.array(observed_mask)
    
            state_ids, observed_mask = states_to_ids(states_stack.reshape(-1, states_stack.shape[-1]))
    
            # Handle unobserved states
            valid_mask = state_ids != -1
            transformed_state = np.zeros((len(state_ids), len(self.state_to_id)))
            if np.sum(valid_mask)>0:
                transformed_state[valid_mask] = self.featurize.transform(
                    self.label_encoder.transform(state_ids[valid_mask]).reshape(-1, 1))
            
        else:  # do nothing
            pass
        return transformed_state, state_shape, observed_mask
        
    def predict_Q(self, state):
        if len(state) == 1:
            if len(state.shape) == 2:
                state = state.reshape((1,1,state.shape[1]))
            elif len(state.shape) == 1:
                state = state.reshape((1,1,state.shape[0]))
        transformed_states, state_shape, observed_mask = self.featurize_state(state)
        if not self.combine_actions:
            # return [self.q_function_list[a].predict(transformed_states).reshape(*state_shape) for a in range(len(self.action_space))]
            # transformed_states, state_shape, observed_mask = self.featurize_state(state)
        
            q_values = [self.q_function_list[a].predict(transformed_states).reshape(*state_shape) for a in range(len(self.action_space))]
        
           # Assign the lowest Q-value to unobserved states
            if observed_mask is not None:
               lowest_q_value = np.min([q.min() for q in q_values])-1
               for q in q_values:
                   q[~observed_mask] = lowest_q_value
        else:
            q_values = [None for _ in range(len(self.action_space))]
            for action_index, a in enumerate(self.action_space):
                state_action_long_vec = self.transform_state_action_long_vec(transformed_states, np.repeat(a, transformed_states.shape[0]))
                q_values[action_index] = self.q_function_list[0].predict(state_action_long_vec).reshape(*state_shape)
            
        return q_values
    
    
    def predict_action(self, state):
        if len(state.shape) < 3:
            raise ValueError('Reshape the state to (i, t, p).')
        
        q_values = self.predict_Q(state)
        # Initialize an array to store the chosen actions
        chosen_actions = np.zeros(q_values[0].shape, dtype=int)
    
        # Iterate over each set of Q-values for the states
        for i in range(q_values[0].shape[0]):
            for j in range(q_values[0].shape[1]):
                # Find the indices of the maximum Q-values
                value = [q_values[a][i, j] for a in range(len(self.action_space))]
                max_indices = np.where(value == np.max(value))[0]
        
                # Randomly select one of the indices
                chosen_actions[i, j] = np.random.choice(max_indices)
        
        return chosen_actions.reshape(state.shape[0], state.shape[1], 1)
        
        # # Initialize an array to store the chosen actions
        # chosen_actions = np.zeros(value.shape[0], dtype=int)
    
        # # Iterate over each set of Q-values for the states
        # for i in range(value.shape[0]):
        #     # Find the indices of the maximum Q-values
        #     max_indices = np.where(value[i] == np.max(value[i]))[0]
    
        #     # Randomly select one of the indices
        #     chosen_actions[i] = np.random.choice(max_indices)
    
        # return chosen_actions



    def weightedTD(self, test_clusters):
        self.load_data(test_clusters)
        if not self.combine_actions:
            self.Qmax = [np.max(np.vstack([self.q_function_list[a].predict(self.cluster[m].next_states_stacked)
                                                 for a in range(len(self.action_space))]), axis=0)
                              for m in range(self.m)]
        else:
            self.Qmax = [np.max(np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(self.cluster[m].next_states_stacked, np.repeat(a, self.cluster[m].next_states_stacked.shape[0])))
                                                 for a in range(len(self.action_space))]), axis=0)
                              for m in range(self.m)]
        
        for m in range(self.m):
            V, _ = self.V_generator(m)
            try:
                eigvals, eigvecs = np.linalg.eigh(V)
            except np.linalg.LinAlgError as e:
                if "Eigenvalues did not converge" in str(e):
                    print("Eigenvalues did not converge. Setting eigenvalues to 1.")
                    eigvals = np.ones_like(np.diag(V))  # Set all eigenvalues to 1
                    eigvecs = np.eye(len(V))  # Identity matrix as eigenvectors
                else:
                    print(e)  # Print the actual error message
                        # eigvals, eigvecs = np.linalg.eigh(V)
            V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
            self.Y_list[m] = V_invsqrt @ self.cluster[m].rewards_stacked + self.gamma * V_invsqrt @ self.Qmax[m]
            self.X_list[m] = V_invsqrt @ self.transform_state_action_long_vec(self.cluster[m].states_stacked,self.cluster[m].actions_stacked) 

        
        for action_index, _ in enumerate(self.action_space):
            for m in range(self.m):
                selected_indices = self.action_indices[action_index][m]
                self.X_action_list[action_index][m] = self.X_list[m][selected_indices]
                self.Y_action_list[action_index][m] = self.Y_list[m][selected_indices]
            if self.combine_actions:
                break
    
        td = np.empty((0))
        for action_index, _ in enumerate(self.action_space):
            X = np.vstack(self.X_action_list[action_index])
            Y = np.hstack(self.Y_action_list[action_index])
            prediction = self.q_function_list[action_index].predict(X)
            err = Y - prediction
            td = np.concatenate((td, err))
            if self.combine_actions:
                break
        return td
            
    
    
    def TD(self, test_clusters):
        '''
        Calculate the TD error on a testing data

        Parameters
        ----------
        test_clusters : testing cluster-arranged data

        Returns
        -------
        test_TD_list : TD error

        '''
        test_m = len(test_clusters)
        test_action_indices = [None for a in self.action_space]
        
        if not self.combine_actions:
            for action_index, a in enumerate(self.action_space):
                test_action_indices[action_index] = [np.where(test_clusters[m].actions_stacked == a)[0] for m in range(test_m)]
        else:
            test_action_indices[0] = [list(range(test_clusters[m].actions_stacked.shape[0])) for m in range(test_m)]

        check_fitted = self.is_model_fitted()
        if check_fitted:
            if not self.combine_actions:
                Qmax = np.vstack([np.max(np.vstack([self.q_function_list[a].predict(test_clusters[m].next_states_stacked)
                                                     for a in range(len(self.action_space))]), axis=0)
                                  for m in range(test_m)])
            else:
                Qmax = [np.max(np.vstack([self.q_function_list_old[0].predict(self.transform_state_action_long_vec(test_clusters[m].next_states_stacked, np.repeat(a, test_clusters[m].next_states_stacked.shape[0])))
                                                     for a in range(len(self.action_space))]), axis=0)
                                  for m in range(self.m)]

        else:
            Qmax = [np.zeros(test_clusters[m].N * test_clusters[m].T) for m in range(test_m)] 
            
        test_q_func_preds = [None for a in self.action_space]
        test_TD_list =  [np.empty(test_clusters[i].N * test_clusters[i].T) for i in range(test_m)]
        
        for action_index, _ in enumerate(self.action_space):
            selected_indices = test_action_indices[action_index]
            if not self.combine_actions:
                test_q_func_preds[action_index] = [self.q_function_list[action_index].predict(test_clusters[m].states_stacked[selected_indices[m]])  for m in range(test_m)]
            else:
                test_q_func_preds[action_index] = [self.q_function_list[0].predict(
                    self.transform_state_action_long_vec(test_clusters[m].states_stacked, 
                                                         test_clusters[m].actions_stacked)) 
                    for m in range(test_m)]
 
            for m in range(test_m):
                test_TD_list[m][selected_indices[m]] = test_clusters[m].rewards_stacked[selected_indices[m]] + \
                    self.gamma * Qmax[m][selected_indices[m]] - \
                    test_q_func_preds[action_index][m]
            if self.combine_actions:
                break
        return test_TD_list
    
    
    
    def clear_data(self):
        del self.cluster, self.X_list, self.Y_list, self.Y_action_list, self.X_action_list
        del self.action_indices,  self.TD_list, self.marginal_variance 
        gc.collect()
