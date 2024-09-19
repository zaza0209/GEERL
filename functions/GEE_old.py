# -*- coding: utf-8 -*-
"""
Delta Q evaluation
"""
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from collections import namedtuple
from sklearn.linear_model import LinearRegression
# from copy import deepcopy, copy
from sklearn.kernel_approximation import RBFSampler
# from numpy import array,argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
#%%
class Cluster():
    def __init__(self, States, Actions, Next_states, Rewards, evaluate_action,
                 basis='rbf', degree=2, rbf_dim=5, rbf_bw=1.0, centered=False,
                 RBFSampler_random_state=1, all_possible_categories = None):
        '''
        one cluster
        '''
        # self.cluster_size = np.where(Actions == evaluate_action)[0].shape[0]
        self.States = States
        self.Next_states = Next_states
        self.Rewards = Rewards
        self.Actions = Actions
        self.nActions = np.unique(self.Actions).shape[0]
        self.T = States.shape[1]
        self.N = States.shape[0]
        self.degree = degree
        self.basis= basis
        # if no rbf basis, then just a linear term
        if rbf_dim == 0:
            self.basis = 'polynomial'
            self.degree = 1

        if self.basis == "rbf":
            self.featurize = RBFSampler(gamma=rbf_bw, random_state=RBFSampler_random_state, n_components=rbf_dim)
            self.model = LinearRegression(fit_intercept=False)
        elif self.basis == "polynomial":
            self.featurize = PolynomialFeatures(degree=self.degree, include_bias=False)#(centered==False)
            self.model = LinearRegression(fit_intercept=False)
        elif self.basis == "one_hot":
            self.featurize = OneHotEncoder(sparse=False)
            self.model = LinearRegression(fit_intercept=False)
        else:
            self.model = None
            pass
        
        self.all_possible_categories = all_possible_categories
        self.transformed_states, self.transformed_next_states = self.featurize_state()        
        self.p = self.transformed_states.shape[2]
        
    def featurize_state(self):
        """
        Returns the transformed representation for a state.
        """
        States_stack = self.States.transpose(2, 0, 1).reshape(self.States.shape[2], -1).T
        Next_states_stack = self.Next_states.transpose(2, 0, 1).reshape(self.Next_states.shape[2], -1).T
    
        combined_state = np.vstack([States_stack, Next_states_stack])
        
        if self.basis == "polynomial" or self.basis == "rbf":
            transformed_state = self.featurize.fit_transform(combined_state)
            identification_metric = self.featurize.fit_transform(np.zeros([1,self.States.shape[2]]))
            state_shape = (self.States.shape[0], self.States.shape[1])
            transformed_states_1 = transformed_state[:States_stack.shape[0]].reshape((*state_shape, -1)) - identification_metric
            transformed_states_2 = transformed_state[States_stack.shape[0]:].reshape((*state_shape, -1)) - identification_metric
            return transformed_states_1, transformed_states_2
    
        elif self.basis == "one_hot":
            self.label_encoder = LabelEncoder()
            # self.one_hot_encoder = OneHotEncoder()
            
            # If you know all possible categories beforehand:
            # all_possible_categories = np.array([/* all possible categories */])
            self.label_encoder.fit(self.all_possible_categories)
            self.featurize.fit(self.label_encoder.transform(self.all_possible_categories).reshape(-1, 1))
            
            integer_encoded = self.label_encoder.transform(combined_state.flatten())
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            transformed_state = self.featurize.transform(integer_encoded)
            state_shape = (self.States.shape[0], self.States.shape[1])
            transformed_states_1 = transformed_state[:States_stack.shape[0]].reshape((*state_shape, -1))
            transformed_states_2 = transformed_state[States_stack.shape[0]:].reshape((*state_shape, -1))
            return transformed_states_1[:, :, :-1], transformed_states_2[:, :, :-1]



            # label_encoder = LabelEncoder()
            # integer_encoded = label_encoder.fit_transform(combined_state.flatten())
            # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            # transformed_state = self.featurize.fit_transform(integer_encoded)
            # state_shape = (self.States.shape[0], self.States.shape[1])
            # transformed_states_1 = transformed_state[:States_stack.shape[0]].reshape((*state_shape, -1))
            # transformed_states_2 = transformed_state[States_stack.shape[0]:].reshape((*state_shape, -1))
            # return transformed_states_1, transformed_states_2
    
        else:  # do nothing
            pass

#%%
class GEE_OPE():
    def __init__(self, cluster, m, 
                 evaluate_action = 1, 
                 basis='rbf', degree=2,
                 rbf_dim=5, rbf_bw=1.0, n_actions=2, centered=False, 
                 RBFSampler_random_state=1, States_next = None, ridge_lambda = 0,
                 estimate_covariance_ornot = True, marginal_variance = 1):
        '''
        initialization
        :param cluster: a list of object Cluster
        :param env: an object of RLenv
        :param degree: degree of polynomial basis used for functional approximation of Q function
        :param gamma: discount rate of rewards
        :param evaluate_action: evaluate policy pi(a|s) = 1 if evaluate_action=a
        :param m: number of clusters
        :param cluster_size: the size of each cluster
        '''

        self.T = cluster[0].T
        self.p = cluster[0].p+1
        self.m = m
        self.beta = None
        self.beta_old = None
        self.evaluate_action = evaluate_action
        self.ridge_lambda = ridge_lambda
        self.cluster = cluster
        self.marginal_variance = None
        self.estimate_covariance_ornot = estimate_covariance_ornot
        self.cluster_size = [cluster[i].N for i in range(self.m)]
        self.X_list = [None for i in range(self.m)]
        self.Y_list = [None for i in range(self.m)]
        self.decor_Y_list = [None for i in range(self.m)]
        self.decor_X_list = [None for i in range(self.m)]
        self.on_policy_index = [np.zeros((self.cluster_size[i], self.T)) for i in range(self.m)]
        self.Rm_list = [None for i in range(self.m)]

        # for prediction only
        self.basis = self.cluster[0].basis
        self.degree = self.cluster[0].degree
        self.featurize = self.cluster[0].featurize
        self.model = self.cluster[0].model
    
        if self.basis == 'one_hot':
            self.label_encoder = self.cluster[0].label_encoder
            
    def estimate_marginal_variance(self):
        '''
        Estimate the marginal variance: sufficient to know the marginal variance in proportion
        which is in proportion to the Phi(S)^2
        '''
        if self.marginal_variance is None:                    
            self.marginal_variance = [np.linalg.norm(self.X_list[m], axis=1) ** 2 + self.ridge_lambda for m in range(self.m)]
        elif self.marginal_variance == 1:
            self.marginal_variance = [np.ones(self.X_list[m].size) for m in range(self.m)]
    
    def estimate_cluster_covariance(self):
        '''

        Estimate the working correlation matrix

        '''
        TD_list = [self.V_invsqrt_times_TD_list[m] for m in range(self.m)]
        if self.iternum >0 :
            TD_list = [np.dot(V_sqrt, TD_list[m]) for m, (V_sqrt) in enumerate(self.V_generator())]

        residual_product = 0
        residual_square = 0
        df = 0
        N = 0
    
        # Estimate correlation coefficient. only subjects within the same cluster has non-zero correlation
        print(' estimate_covariance_ornot',self.estimate_covariance_ornot)
        if self.estimate_covariance_ornot:
            for m in range(self.m):
                split_lengths = np.sum(self.on_policy_index[m], axis=1)
                TD_cluster_split = np.split(TD_list[m], (self.p*np.cumsum(split_lengths)[:-1]).astype(int))
                marginal_variance_split = np.split(self.marginal_variance[m], (self.p*np.cumsum(split_lengths)[:-1]).astype(int))
                
                def one_residual_product(i1, i2, index):
                    i1_index = np.sum(self.on_policy_index[m][i1, :index]).astype(int)
                    i2_index = np.sum(self.on_policy_index[m][i2, :index]).astype(int)
                    i1_mapped_indices = list(range(self.p*i1_index, self.p*i1_index+self.p))
                    i2_mapped_indices = list(range(self.p*i2_index, self.p*i2_index+self.p))
                    tmp = (2 * TD_cluster_split[i1][i1_mapped_indices] * TD_cluster_split[i2][i2_mapped_indices] /
                                               (np.sqrt(marginal_variance_split[i1][i1_mapped_indices]) *
                                                np.sqrt(marginal_variance_split[i2][i2_mapped_indices])))
                    return tmp
                
                # residual product
                for i1 in range(self.cluster_size[m]):
                    on_policy_i1 = self.on_policy_index[m][i1]
                    for i2 in range(i1 + 1, self.cluster_size[m]):
                        on_policy_i2 = self.on_policy_index[m][i2]
                        both_on_policy = on_policy_i1.astype(bool) & on_policy_i2.astype(bool)
                        on_policy_indices = np.where(both_on_policy)[0]
                        residual_product_tmp = np.hstack([one_residual_product(i1, i2, index) for index in on_policy_indices])
                        df +=  residual_product_tmp.size * 2
                        residual_product += np.sum(residual_product_tmp)
                
                # residual square
                residual_square += np.sum(np.concatenate([TD_cluster_split[i] ** 2 / marginal_variance_split[i] ** 2 for i in range(self.cluster_size[m])]))
                N += TD_list[m].shape[0]
                
            self.dispersion_param = residual_square / (N - self.beta.shape[0])
            self.rho = residual_product / (self.dispersion_param * (df - self.beta.shape[0]))
    

    def calculate_V(self, m):
        '''
        Estimating the covariance

        '''
        # print('calculate_V: self.estimate_covariance_ornot', self.estimate_covariance_ornot)
        on_policy_indices = self.on_policy_index[m].flatten().nonzero()[0]
        if self.estimate_covariance_ornot:
            off_block = self.rho * np.identity(self.T)
            Rm = (1 - self.rho) * np.identity(self.cluster_size[m] * self.T) + \
                 np.kron(off_block, np.ones((self.cluster_size[m], self.cluster_size[m])))
        
            # on_policy_indices = self.on_policy_index[m].flatten().nonzero()[0]
            Rm = Rm[on_policy_indices][:, on_policy_indices]
            Rm_kron = np.kron(Rm, np.eye(self.p))
            A_sqrt = np.diag(self.marginal_variance[m] ** 0.5)
        
            # V = self.dispersion_param * A_sqrt @ Rm_kron @ A_sqrt
            V_sqrt = scipy.linalg.sqrtm(self.dispersion_param * A_sqrt @ Rm_kron @ A_sqrt)
        else:
            print('Cov not estimated')
            V_sqrt = np.identity(len(on_policy_indices))
        return V_sqrt

    def V_generator(self):
        for m in range(self.m):
            yield self.calculate_V(m)

       
    def create_design_matrix_cluster(self):
        for m in range(self.m):
            all_x = []
            all_y = []
            on_policy_indices = np.argwhere(self.cluster[m].Actions == self.evaluate_action)
            transformed_next_states = self.cluster[m].transformed_next_states #[:, :, 1:]
            transformed_states = self.cluster[m].transformed_states #[:, :, 1:]
            self.on_policy_index[m][on_policy_indices[:, 0], on_policy_indices[:, 1]] = 1
            for j, t in on_policy_indices:
                Phi_action_eva_next = transformed_next_states[j, t]
                Phi_action_current = transformed_states[j, t]
                Phi_diff = Phi_action_current - Phi_action_eva_next
                Phi_diff = np.hstack([Phi_diff, 1])
    
                Phi_action_current = np.hstack([Phi_action_current, 1])
                Phi_reward = Phi_action_current * self.cluster[m].Rewards[j, t]
                x_onestep = np.outer(Phi_action_current, Phi_diff)
                all_y.append(Phi_reward)
                all_x.append(x_onestep)
    
            self.Y_list[m] = np.hstack(all_y)
            self.X_list[m] = np.vstack(all_x)


    def decorrelated_XY(self):
        if self.iternum >0 and self.estimate_covariance_ornot:
            for m, (V_sqrt) in enumerate(self.V_generator()):
                V_invsqrt = np.linalg.inv(V_sqrt)
                self.decor_Y_list[m] = np.dot(V_invsqrt, self.Y_list[m])
                self.decor_X_list[m] = np.dot(V_invsqrt, self.X_list[m])
        else:
            self.decor_Y_list = [self.Y_list[m] for m in range(self.m)]
            self.decor_X_list = [self.X_list[m] for m in range(self.m)]
    
    def estimate_beta_variance(self):
        I1 = np.zeros((self.beta.shape[0], self.beta.shape[0]))
        I0 = np.zeros((self.beta.shape[0], self.beta.shape[0]))
    
        for m in range(self.m):
            cov_Y = np.dot(self.V_invsqrt_times_TD_list[m], self.V_invsqrt_times_TD_list[m].T)
            X_T_cov_Y_X = np.dot(self.decor_X_list[m].T, np.dot(cov_Y, self.decor_X_list[m]))
            I1 += X_T_cov_Y_X
            I0 += np.dot(self.decor_X_list[m].T, self.decor_X_list[m])
    
        self.beta_cov = np.dot(np.linalg.inv(I0), np.dot(I1, np.linalg.inv(I0)))

    def solve_GEE(self):
        self.decorrelated_XY()
        X = np.vstack(self.decor_X_list)
        Y = np.hstack(self.decor_Y_list)
    
        self.beta_old = np.zeros(X.shape[1]) if self.beta is None else self.beta.copy()
        self.beta = np.linalg.lstsq(X.reshape(Y.size, -1), Y, rcond=None)[0]
    
        self.V_invsqrt_times_TD_list = [self.decor_Y_list[m] - np.dot(self.decor_X_list[m], self.beta) for m in range(self.m)]
    
    def fit(self, max_iter, stopping_threshold=1e-5):
        self.create_design_matrix_cluster()
        self.estimate_marginal_variance()
        self.converge = 0
        self.iternum = 0
        for l in range(max_iter):
            self.solve_GEE()
            self.estimate_cluster_covariance()
    
            if l > 0:
                norm_ratio = np.linalg.norm(self.beta - self.beta_old) / np.linalg.norm(self.beta)
                if norm_ratio < stopping_threshold:
                    self.converge = 1
                    self.iternum = l+1
                    break
    
        self.estimate_beta_variance()
        self.eta = self.beta[-1]
        self.eta_var = self.beta_cov[-1, -1]
        GEE_res = namedtuple("GEE_res", ["eta", "eta_var", "beta", "cov", 'converge', 'iter_num'])
        return GEE_res(self.beta[-1], self.beta_cov[-1, -1], self.beta, self.beta_cov, self.converge, self.iternum)

    def featurize_state(self, state):
        '''
        Transform state for prediction

        Parameters
        ----------
        state : one single state or a states for multiple stage. should be of shape [N, T, p]

        '''
        States_stack = state.transpose(2, 0, 1).reshape(state.shape[2], -1).T
        
        if self.basis == "polynomial" or self.basis == "rbf":
            transformed_state = self.featurize.transform(States_stack)
            identification_metric = self.featurize.transform(np.zeros([1,state.shape[2]]))
            state_shape = (state.shape[0], state.shape[1])
            transformed_states = transformed_state.reshape((*state_shape, -1)) - identification_metric
            return transformed_states
    
        elif self.basis == "one_hot":
            integer_encoded = self.label_encoder.transform(States_stack.flatten())
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            transformed_state = self.featurize.transform(integer_encoded)
            state_shape = (state.shape[0], state.shape[1])
            transformed_states = transformed_state.reshape((*state_shape, -1))
            return transformed_states[:,:,:-1]
    
        else:  # do nothing
            pass
        
    def predict_Q(self, state):
        transformed_states = self.featurize_state(state)
        return np.dot(transformed_states, self.beta[:-1])