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

import numpy as np
# import pandas as pd
# from scipy import linalg as spl
import statsmodels
from statsmodels.tools.validation import bool_like
import scipy
import statsmodels
from collections import namedtuple
import warnings
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    NotImplementedWarning,
    OutputWarning,
)

class CovStruct:
    """
    Base class for correlation and covariance structures.

    An implementation of this class takes the residuals from a
    regression model that has been fit to grouped data, and uses
    them to estimate the within-group dependence structure of the
    random errors in the model.

    The current state of the covariance structure is represented
    through the value of the `dep_params` attribute.

    The default state of a newly-created instance should always be
    the identity correlation matrix.
    """

    def __init__(self):

        # Parameters describing the dependency structure
        self.dep_params = None

        # Keep track of the number of times that the covariance was
        # adjusted.
        self.cov_adjust = []

        # Method for projecting the covariance matrix if it is not
        # PSD.
        # self.cov_nearest_method = cov_nearest_method



    def initialize(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        self.model = model



    def update(self, params):
        """
        Update the association parameter values based on the current
        regression coefficients.

        Parameters
        ----------
        params : array_like
            Working values for the regression parameters.
        """
        raise NotImplementedError



    def covariance_matrix(self, index):
        """
        Returns the working covariance or correlation matrix for a
        given cluster of data.

        Parameters
        ----------
        endog_expval : array_like
           The expected values of endog for the cluster for which the
           covariance or correlation matrix will be returned
        index : int
           The index of the cluster for which the covariance or
           correlation matrix will be returned

        Returns
        -------
        M : matrix
            The covariance or correlation matrix of endog
        is_cor : bool
            True if M is a correlation matrix, False if M is a
            covariance matrix
        """
        raise NotImplementedError
    def V_invsqrt(self, index, action_index = None):
        V, isvar = self.covariance_matrix(index)
        eigvals, eigvecs = np.linalg.eigh(V)
        V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        return V_invsqrt, isvar
    
    def decorrelate(self, b, index, action_index = None):
        V, isvar = self.covariance_matrix(index, action_index)
        eigvals, eigvecs = np.linalg.eigh(V)
        V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        
        # Solve the system of linear equations
        x = scipy.linalg.solve_triangular(V_invsqrt, b, lower=True)
        return x



    def summary(self):
        """
        Returns a text summary of the current estimate of the
        dependence structure.
        """
        raise NotImplementedError
#%% Independence
class Independence(CovStruct):
    """
    An independence working dependence structure.
    """


    def update(self):
        # Nothing to update
        return



    def covariance_matrix(self, index):
        dim = self.model.N*self.model.T
        return np.eye(dim, dtype=np.float64), True



    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        v = stdev ** 2
        rslt = []
        for x in rhs:
            if x.ndim == 1:
                rslt.append(x / v)
            else:
                rslt.append(x / v[:, None])
        return rslt



    def summary(self):
        return ("Observations within a cluster are modeled "
                "as being independent.")

#%% exchangeable
class Exchangeable(CovStruct):
    def __init__(self):

        super(Exchangeable, self).__init__()
        self.dep_params = 0.
    def initialize(self, model):
        self.model = model
        if model.MC_rho is not None:
            self.dep_params = model.MC_rho
        
    def update(self):
        if self.model.MC_rho is None:
            
            nobs = self.model.T * np.sum(self.model.cluster_size)

            residsq_sum, scale = 0, 0
            n_pairs = 0
            for m in range(self.model.m):
                # expval, _ = cached_means[i]
                stdev = np.sqrt(self.model.marginal_variance[m])
                resid = self.model.TD_list[m] / stdev
                # f = weights_li[i] if has_weights else 1.

                ssr = np.sum(resid * resid)
                scale += ssr
                # fsum1 += self.model.cluster_size[m] * self.model.T

                residsq_sum += (resid.sum() ** 2 - ssr) / 2
                ngrp = len(resid)
                npr = 0.5 * ngrp * (ngrp - 1)
                # fsum2 += npr
                n_pairs += npr
            
            ddof = self.model.p
            scale /=  (nobs - ddof)
            self.dispersion_param = scale
            residsq_sum /= (n_pairs - ddof) 
            # print('nobs',nobs,'scale',scale,'residsq_sum',residsq_sum)
            self.dep_params = residsq_sum / scale
            # print(self.dep_params)
        else:
            self.dep_params = self.model.MC_rho
        
    def covariance_matrix(self, index, action_index = None):
        if action_index is None:
            dim = self.model.cluster_size[index] * self.model.T
            A_sqrt = np.diag(self.model.marginal_variance[index] ** 0.5)
           
        else:
            dim = len(self.model.action_indices[action_index][index])
            A_sqrt = np.diag(self.model.marginal_variance[index][self.model.action_indices[action_index][index]] ** 0.5)
           
        R = self.dep_params * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(R, 1)
        
        V = A_sqrt @ R @ A_sqrt
            
        return V, False
    
    def V_invsqrt(self, index, action_index = None):
        '''

        Parameters
        ----------
        index : int
            index of cluster.
        action_index : int, optional
            index of action. The default is None.

        Returns
        -------
        V_invsqrt : inverse of the sqrt of V
        bool
            whehter it is for a variance matrix.

        '''
        if action_index is None:
            dim = self.model.cluster_size[index] * self.model.T
        else:
            dim = len(self.model.action_indices[action_index][index])
        # Eigenvalues
        eigvals = np.array([dim * self.dep_params + 1 - self.dep_params] + [(1 - self.dep_params)] * (dim-1))
        
        # Eigenvectors
        eigvecs = np.zeros((dim, dim))
        eigvecs[:, 0] = np.ones(dim) / np.sqrt(dim)
        for i in range(1, dim):
            eigvecs[i-1, i] = np.sqrt((dim-i)/dim/(i))
            eigvecs[i:, i] = -np.sqrt((dim-i)/dim/(i+1))
        
        V_invsqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        
        return V_invsqrt, False
        
        
    def decorrelate(self, b, index, action_index=None):
        V, isvar = self.covariance_matrix(index, action_index)
        eigvals, eigvecs = np.linalg.eigh(V)
        V_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Solve the system of linear equations
        x = scipy.linalg.solve(V_sqrt, b, lower=True)
        return x
        
#%% statsmodels.genmod.cov_struct.CovStruct
class Exchangeable_sm(statsmodels.genmod.cov_struct.CovStruct):
    def __init__(self):

        super(Exchangeable_sm, self).__init__()
        self.dep_params = 0.
    
    def setparams(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        self.model = model
    def cov(self, endog, exog, index=None, params=None):
        # Calculate the covariance matrix based on your custom correlation structure
        # endog: response variable
        # exog: predictor variables
        # index: optional grouping variable indices

        n = len(endog)  # Number of observations
        cov_matrix = (1 -  self.dep_params) * np.identity(n) + self.dep_params * np.ones([n, n])

        return cov_matrix

    def covariance_matrix(self, endog, params):
        # Calculate the covariance matrix based on the correlation matrix
        # params: covariance parameters

        corr_matrix = self.cov(endog, None, params=params)  # Get the correlation matrix

        return corr_matrix, False  # Return the covariance matrix and the boolean indicating whether it's a correlation matrix
    def assign_params(self, params):
        self.dep_params = params
    def update(self, params):
        # self.dep_params = params
        pass
    #     self.dep_params = self.model.cov_struct.dep_params
#%%
class statsExchangeable(statsmodels.genmod.cov_struct.CovStruct):
    """
    An exchangeable working dependence structure.
    """

    def __init__(self):

        super(statsExchangeable, self).__init__()

        # The correlation between any two values in the same cluster
        self.dep_params = 0.
    
    def setparams(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        self.model = model
        
    def update(self, params):

        endog = self.model.endog_li

        self.nobs = self.model.nobs

        varfunc = self.model.family.variance

        cached_means = self.model.cached_means

        has_weights = self.model.weights is not None
        weights_li = self.model.weights

        self.residsq_sum, self.scale = 0, 0
        self.fsum1, self.fsum2, self.n_pairs = 0., 0., 0.
        for i in range(self.model.num_group):
            # print('=== i', i, '===')
            expval, _ = cached_means[i]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev
            f = weights_li[i] if has_weights else 1.

            ssr = np.sum(resid * resid)
            self.scale += f * ssr
            self.fsum1 += f * len(endog[i])
            # print('len(endog[i])',len(endog[i]))
            self.residsq_sum += f * (resid.sum() ** 2 - ssr) / 2
            self.ngrp = len(resid)
            # print('self.ngrp', self.ngrp)
            self.npr = 0.5 * self.ngrp * (self.ngrp - 1)
            self.fsum2 += f * self.npr
            self.n_pairs += self.npr
        # print('residsq_sum',self.residsq_sum,'scale', self.scale, 'npr', self.npr, 'fsum1', self.fsum1, 'fsum2', self.fsum2, 'nobs', self.nobs, 'n_pairs',self.n_pairs)
        ddof = self.model.ddof_scale
        self.scale /= (self.fsum1 * (self.nobs - ddof) / float(self.nobs))
        self.residsq_sum /= self.scale
        self.dep_params = self.residsq_sum / \
            (self.fsum2 * (self.n_pairs - ddof) / float(self.n_pairs))
        print(' self.dep_params', self.dep_params)
        
    def assign_params(self, params):
        self.dep_params = params
        
    def covariance_matrix(self, expval, index):
        dim = len(expval)
        dp = self.dep_params * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(dp, 1)
        return dp, True

    

    def covariance_matrix_solve(self, expval, index, stdev, rhs):

        k = len(expval)
        c = self.dep_params / (1. - self.dep_params)
        c /= 1. + self.dep_params * (k - 1)

        rslt = []
        for x in rhs:
            if x.ndim == 1:
                x1 = x / stdev
                y = x1 / (1. - self.dep_params)
                y -= c * sum(x1)
                y /= stdev
            else:
                x1 = x / stdev[:, None]
                y = x1 / (1. - self.dep_params)
                y -= c * x1.sum(0)
                y /= stdev[:, None]
            rslt.append(y)

        return rslt



    def summary(self):
        return ("The correlation between two observations in the " +
                "same cluster is %.3f" % self.dep_params)


#%% add noise to data at each time point for subjects within the same cluster
class Exchangeable_subjects(CovStruct):
    def __init__(self):
        super(Exchangeable_subjects, self).__init__()
        self.dep_params = 0.
        
    def initialize(self, model):
        self.model = model
        if model.MC_rho is not None:
            self.dep_params = model.MC_rho
            
    def update(self):
        if self.model.MC_rho is None:
            
            # residual_square = 0
            # N = 0
            # residual_product = 0
            
            # for m in range(self.model.m):
            #     TD_normalized = self.model.TD_list[m] / self.model.marginal_variance[m]
            #     residual_square += np.sum(TD_normalized ** 2)
            #     N += self.model.cluster_size[m] * self.model.T
            
            #     # Vectorized computation of residual product
            #     for i1 in range(self.model.cluster_size[m] - 1):
            #         start_idx = sum(self.model.cluster_size[:m]) + i1 * self.model.T
            #         end_idx = start_idx + self.model.T
            #         TD_cluster = TD_normalized[start_idx:end_idx]
            
            #         # Summing up all the subsequent clusters and multiplying with the current cluster
            #         sum_subsequent_clusters = np.sum(TD_normalized[end_idx:], axis=0)
            #         residual_product += np.sum(TD_cluster * sum_subsequent_clusters)
            
            # df = 0.5 * N - 0.5 * sum(self.model.cluster_size)
            
            # self.dispersion_param = residual_square / N
            # self.dep_params = residual_product / (self.dispersion_param * df)
            
            # # Clipping the dep_params between -1 and 1
            # self.dep_params = np.clip(self.dep_params, -1, 1)


            residual_product = 0
            residual_square = 0
            df = 0
            N = 0
        
            for m in range(self.model.m):
                TD_cluster_split = np.split(self.model.TD_list[m]/self.model.marginal_variance[m], self.model.cluster_size[m])
        
                # Compute residual product
                for i1 in range(self.model.cluster_size[m]):
                    residual_product_tmp = TD_cluster_split[i1]* np.sum(TD_cluster_split[i1+1:], axis=0)
                    residual_product += np.sum(residual_product_tmp)
        
                # Compute residual square
                residual_square += np.sum((self.model.TD_list[m]/self.model.marginal_variance[m]) ** 2)
                N += self.model.cluster_size[m] * self.model.T
                df += 0.5 * self.model.cluster_size[m] * (self.model.cluster_size[m] - 1) * self.model.T
           
            self.dispersion_param = residual_square / N #(N - self.model.p)
            self.dep_params = residual_product / (self.dispersion_param * df)# (df - self.model.p))
            # Clipping the dep_params between -1 and 1
            self.dep_params = np.clip(self.dep_params, -1, 1)
    
    
        else:
            self.dep_params = self.model.MC_rho
            
    def covariance_matrix(self, index, action_index = None):
        off_block = self.dep_params * np.identity(self.model.T)
        Rm = (1 - self.dep_params) * np.identity(self.model.cluster_size[index] * self.model.T) + \
             np.kron(np.ones((self.model.cluster_size[index], self.model.cluster_size[index])),off_block)

        if action_index is not None:
            selected_indices = self.model.action_indices[action_index][index]
            Rm = Rm[selected_indices][:, selected_indices]
            
        V = Rm
            
        return V, False
#%% statsmodel 
class Exchangeable_subjects_sm(statsmodels.genmod.cov_struct.CovStruct):
    def __init__(self):

        super(Exchangeable_subjects_sm, self).__init__()
        self.dep_params = 0.
        
    def setparams(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        # print('cov',model)
        self.model = model
        self.T = model.T
        self.cluster_size = model.cluster_size
        self.action_indices = model.action_indices
        # self.marginal_variance = model.marginal_variance

    def covariance_matrix(self, endog, index=None):
        # Calculate the covariance matrix based on the correlation matrix
        # index: optional grouping variable indices
        off_block = self.dep_params * np.identity(self.T)
        Rm = (1 - self.dep_params) * np.identity(self.cluster_size[index] * self.T) + \
             np.kron(np.ones((self.cluster_size[index], self.cluster_size[index])),off_block)
        
        selected_indices = self.action_indices[self.action_index][index]
        Rm = Rm[selected_indices][:, selected_indices]
        # A_sqrt = np.diag(self.marginal_variance[index][selected_indices[index]] ** 0.5)
            
        corr_matrix = Rm  # Get the correlation matrix
        # print('Rm', Rm, 'action_index',self.action_index, 'selected_indices', selected_indices)
        return corr_matrix, False  # Return the covariance matrix and the boolean indicating whether it's a correlation matrix
    def assign_params(self, params):
        self.dep_params = params
    def update(self, params):
        # self.dep_params = params
        pass
#%% AR
class Autoregressive(CovStruct):
    def __init__(self):
        super(Autoregressive, self).__init__()
        self.dep_params = 0.
        
    def initialize(self, model):
        self.model = model
        if model.MC_rho is not None:
            self.dep_params = model.MC_rho
        
    def update(self):
        if self.model.MC_rho is None:
            lag0, lag1 = 0.0, 0.0
            for m in range(self.model.m):
                stdev =  np.sqrt(self.model.marginal_variance[m])
                resid = self.model.TD_list[m] / stdev

                n = len(resid)
                if n > 1:
                    lag1 += np.sum(resid[0:-1] * resid[1:]) / (n - 1)
                    lag0 += np.sum(resid**2) / n

            self.dep_params = lag1 / lag0
        else:
            self.dep_params = self.model.MC_rho

    def covariance_matrix(self, index, action_index=None):
        if action_index is None:
            dim = self.model.cluster_size[index] * self.model.T
            A_sqrt = np.diag(self.model.marginal_variance[index] ** 0.5)
        else:
            dim = len(self.model.action_indices[action_index][index])
            A_sqrt = np.diag(self.model.marginal_variance[index][self.model.action_indices[action_index][index]] ** 0.5)

        R = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            for j in range(dim):
                R[i,j] = self.dep_params ** abs(i - j)
        
        V = A_sqrt @ R @ A_sqrt
            
        return V, False
#%% statsmodel autoregressive
class Autoregressive_sm(statsmodels.genmod.cov_struct.CovStruct):
    """
    A first-order autoregressive working dependence structure.

    The dependence is defined in terms of the `time` component of the
    parent GEE class, which defaults to the index position of each
    value within its cluster, based on the order of values in the
    input data set.  Time represents a potentially multidimensional
    index from which distances between pairs of observations can be
    determined.

    The correlation between two observations in the same cluster is
    dep_params^distance, where `dep_params` contains the (scalar)
    autocorrelation parameter to be estimated, and `distance` is the
    distance between the two observations, calculated from their
    corresponding time values.  `time` is stored as an n_obs x k
    matrix, where `k` represents the number of dimensions in the time
    index.

    The autocorrelation parameter is estimated using weighted
    nonlinear least squares, regressing each value within a cluster on
    each preceding value in the same cluster.

    Parameters
    ----------
    dist_func : function from R^k x R^k to R^+, optional
        A function that computes the distance between the two
        observations based on their `time` values.

    References
    ----------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    def __init__(self, dist_func=None, grid=None):

        super(Autoregressive_sm, self).__init__()
        self.dep_params = 0.
    def setparams(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        # print('cov',model)
        self.model = model
        # self.T = model.T
        # self.cluster_size = model.cluster_size
        # self.action_indices = model.action_indices
        # # self.marginal_variance = model.marginal_vari
        
    def assign_params(self, params):
        self.dep_params = params
        
    def update(self, params):
        pass

    def covariance_matrix(self, endog, params):
        # Calculate the covariance matrix based on the correlation matrix
        # params: covariance parameters
        # Calculate the covariance matrix based on your custom correlation structure
        # endog: response variable
        # exog: predictor variables
        # index: optional grouping variable indices
        n = len(endog)

        R = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                R[i,j] = self.dep_params ** abs(i - j)
            
        return R, False  # Return the covariance matrix and the boolean indicating whether it's a correlation matrix
    

    def _update_nogrid(self, params):

        endog = self.model.endog_li
        time = self.model.time_li

        # Only need to compute this once
        if self.designx is not None:
            designx = self.designx
        else:
            designx = []
            for i in range(self.model.num_group):

                ngrp = len(endog[i])
                if ngrp == 0:
                    continue

                # Loop over pairs of observations within a cluster
                for j1 in range(ngrp):
                    for j2 in range(j1):
                        designx.append(self.dist_func(time[i][j1, :],
                                                      time[i][j2, :]))

            designx = np.array(designx)
            self.designx = designx

        scale = self.model.estimate_scale()
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means

        # Weights
        var = 1. - self.dep_params ** (2 * designx)
        var /= 1. - self.dep_params ** 2
        wts = 1. / var
        wts /= wts.sum()

        residmat = []
        for i in range(self.model.num_group):

            expval, _ = cached_means[i]
            stdev = np.sqrt(scale * varfunc(expval))
            resid = (endog[i] - expval) / stdev

            ngrp = len(resid)
            for j1 in range(ngrp):
                for j2 in range(j1):
                    residmat.append([resid[j1], resid[j2]])

        residmat = np.array(residmat)

        # Need to minimize this
        def fitfunc(a):
            dif = residmat[:, 0] - (a ** designx) * residmat[:, 1]
            return np.dot(dif ** 2, wts)

        # Left bracket point
        b_lft, f_lft = 0., fitfunc(0.)

        # Center bracket point
        b_ctr, f_ctr = 0.5, fitfunc(0.5)
        while f_ctr > f_lft:
            b_ctr /= 2
            f_ctr = fitfunc(b_ctr)
            if b_ctr < 1e-8:
                self.dep_params = 0
                return

        # Right bracket point
        b_rgt, f_rgt = 0.75, fitfunc(0.75)
        while f_rgt < f_ctr:
            b_rgt = b_rgt + (1. - b_rgt) / 2
            f_rgt = fitfunc(b_rgt)
            if b_rgt > 1. - 1e-6:
                raise ValueError(
                    "Autoregressive: unable to find right bracket")

        from scipy.optimize import brent
        self.dep_params = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])

#%% AutoEx
class Autoex(CovStruct):
    '''
    Autoregressive+exchangeable
    Corr(X_{n+t},X_n) = 
    \frac{\phi^t var(\epsilon) /(1-\phi^2) +var(alpha)}{var(\epsilon) /(1-\phi^2) +var(alpha)}
    '''
    def __init__(self, var_td=0, set_weight=None):
        super(Autoex, self).__init__()
        # dep_params = namedtuple("dep_params", ["autoregressive_part", "exchangeable_part"])
        self.dep_params = {"var_alpha":0,  "var_epsilon/1-phi^2":0, "autoregressive_coef":0, 'var_td':0}
        self.var_td=var_td
        self.set_weight = set_weight
        
    def initialize(self, model):
        self.model = model
        # dep_params = namedtuple("dep_params", ["autoregressive_part", "exchangeable_part"])
        # self.dep_params = dep_params(0,0)
        
    def update(self):
        need_update = 0
        if self.model.MC_rho is None:
            need_update = 1
        else:
            for key, value in self.model.MC_rho.items():
                if value is None:
                    need_update = 1 
                    break
        if need_update:
            ## assume VAR(TD)!=0
            if self.var_td:
                lag0, lag1, lag2, lag3 = 0.0, 0.0, 0.0, 0.0
                for m in range(self.model.m):
                    TD_cluster_split = np.split(self.model.TD_list[m]/self.model.marginal_variance[m], self.model.cluster_size[m])
            
                    # Compute residual product
                    for i1 in range(self.model.cluster_size[m]):
                        resid = TD_cluster_split[i1]
                        n = len(resid)
                        if n > 3:
                            lag3 += np.sum(resid[:-3] * resid[3:])/(n-3)
                            lag2 += np.sum(resid[:-2] * resid[2:])/(n-2)
                            lag1 += np.sum(resid[0:-1] * resid[1:]) / (n - 1)
                            lag0 += np.sum(resid**2) / n
                N_total =np.sum(self.model.cluster_size)     
                if lag0 == 0.0:
                    raise ValueError("The sample size(T) is too small! Please try another correlation structure.")
                lag0 /= N_total
                lag1 /= N_total
                lag2 /= N_total
                lag3 /= N_total
                
                phi_1 = self.model.MC_rho['autoregressive_coef'] if self.model.MC_rho is not None and self.model.MC_rho['autoregressive_coef'] is not None else (lag2-lag3)/(lag1-lag2) 
                var_epsilon_1 = self.model.MC_rho['var_epsilon/1-phi^2'] if self.model.MC_rho is not None and self.model.MC_rho['var_epsilon/1-phi^2'] is not None else (lag1-lag2)/(phi_1*(1-phi_1))
                var_alpha_1 = self.model.MC_rho["var_alpha"] if self.model.MC_rho is not None and self.model.MC_rho['var_alpha'] is not None else lag1 - phi_1 * var_epsilon_1 
                var_td_1 = self.model.MC_rho["var_td"] if self.model.MC_rho is not None and self.model.MC_rho['var_td'] is not None else lag0 - var_epsilon_1 - var_alpha_1 
                print('phi_1', phi_1, 'var_epsilon_1', var_epsilon_1, 'var_alpha_1', var_alpha_1, 'var_td_1',var_td_1)
  
                # method 2:
                ssq = 0
                df=0
                for m in range(self.model.m):
                    TD_cluster_split = np.split(self.model.TD_list[m]/self.model.marginal_variance[m], self.model.cluster_size[m])
                    
                    # # Compute residual product
                    # for i1 in range(self.model.cluster_size[m]-1):
                    #     for i2 in range(i1+1, self.model.cluster_size[m]):
                    #         t1 = TD_cluster_split[i1]
                    #         t2 = TD_cluster_split[i2]
                    #         ssq += np.sum(t1*t2)
                    # df+= self.model.T*self.model.cluster_size[m]*(self.model.cluster_size[m]-1)/2
       
                    
                    # Compute residual product
                    for i1 in range(self.model.cluster_size[m]-1):
                        t1 = np.sum(TD_cluster_split[i1])
                        for i2 in range(i1+1, self.model.cluster_size[m]):
                            t2 = np.sum(TD_cluster_split[i2])
                            ssq += t1 * t2
                    df+= self.model.T*self.model.T*self.model.cluster_size[m]*(self.model.cluster_size[m]-1)/2
                if df >0:    
                    var_alpha_2 = self.model.MC_rho["var_alpha"] if self.model.MC_rho is not None and self.model.MC_rho['var_alpha'] is not None else ssq/df
                    phi_2 = self.model.MC_rho['autoregressive_coef'] if self.model.MC_rho is not None and self.model.MC_rho['autoregressive_coef'] is not None else (lag2-var_alpha_2)/(lag1 - var_alpha_2)
                    var_epsilon_2 = self.model.MC_rho['var_epsilon/1-phi^2'] if self.model.MC_rho is not None and self.model.MC_rho['var_epsilon/1-phi^2'] is not None else  (lag1-var_alpha_2)/phi_2 
                    var_td_2 = self.model.MC_rho["var_td"] if self.model.MC_rho is not None and self.model.MC_rho['var_td'] is not None else  lag0 - var_alpha_2 - var_epsilon_2 
                    print('phi_2', phi_2, 'var_epsilon_2', var_epsilon_2, 'var_alpha_2', var_alpha_2,'var_td_2',var_td_2)

                if self.set_weight == None:
                    weight = (N_total*self.model.T)/((N_total*self.model.T)+df)
                else:
                    weight = self.set_weight
                if df>0:
                    self.dep_params["autoregressive_coef"] = weight* phi_1 + (1-weight)*phi_2
                    self.dep_params["var_epsilon/1-phi^2"] = weight*var_epsilon_1 + (1-weight) * var_epsilon_2
                    self.dep_params["var_alpha"] = weight*var_alpha_1 + (1-weight) * var_alpha_2
                    self.dep_params["var_td"] = weight*var_td_1 + (1-weight) * var_td_2
                else:
                    self.dep_params["autoregressive_coef"] = phi_1 
                    self.dep_params["var_epsilon/1-phi^2"] = var_epsilon_1  
                    self.dep_params["var_alpha"] = var_alpha_1 
                    self.dep_params["var_td"] = var_td_1  

            # assume var(TD)=0
            else:
                lag0, lag1, lag2 = 0.0, 0.0, 0.0
                # df_1=0
                for m in range(self.model.m):
                    TD_cluster_split = np.split(self.model.TD_list[m]/self.model.marginal_variance[m], self.model.cluster_size[m])
            
                    # Compute residual product
                    for i1 in range(self.model.cluster_size[m]):
                        resid = TD_cluster_split[i1]
                        n = len(resid)
                        if n > 2:
                            lag2 += np.sum(resid[:-2] * resid[2:])/(n-2)
                            lag1 += np.sum(resid[0:-1] * resid[1:]) / (n - 1)
                            lag0 += np.sum(resid**2) / n
                            # df_1+=n
                N_total =np.sum(self.model.cluster_size)     
                if lag0 == 0.0:
                    raise ValueError("The sample size(T) is too small! Please try another correlation structure.")
                lag0 /= N_total
                lag1 /= N_total
                lag2 /= N_total
                
                phi_1=(lag1-lag2)/(lag0-lag1)
                var_epsilon_1 = (lag0-lag1)/(1-phi_1)
                var_alpha_1 = lag0 - var_epsilon_1
                print('phi_1', phi_1, 'var_epsilon_1', var_epsilon_1, 'var_alpha_1', var_alpha_1)
                # utilizing data from different subjects
                ssq = 0
                df=0
                for m in range(self.model.m):
                    TD_cluster_split = np.split(self.model.TD_list[m]/self.model.marginal_variance[m], self.model.cluster_size[m])
                    
                    # # Compute residual product
                    # for i1 in range(self.model.cluster_size[m]-1):
                    #     for i2 in range(i1+1, self.model.cluster_size[m]):
                    #         t1 = TD_cluster_split[i1]
                    #         t2 = TD_cluster_split[i2]
                    #         ssq += np.sum(t1*t2)
                    # df+= self.model.T*self.model.cluster_size[m]*(self.model.cluster_size[m]-1)/2
                
                    # Compute residual product
                    for i1 in range(self.model.cluster_size[m]-1):
                        t1 = np.sum(TD_cluster_split[i1])
                        for i2 in range(i1+1, self.model.cluster_size[m]):
                            t2 = np.sum(TD_cluster_split[i2])
                            ssq += t1 * t2
                    df+= self.model.T*self.model.T*self.model.cluster_size[m]*(self.model.cluster_size[m]-1)/2
            
                if df >0:    
                    var_alpha_2 = ssq/df
                    var_epsilon_2 = lag0 - var_alpha_2
                    phi_2 = (lag1 - var_alpha_2)/var_epsilon_2
                    print('phi_2', phi_2, 'var_epsilon_2', var_epsilon_2, 'var_alpha_1', var_alpha_2)
                else:
                    self.set_weight =1
                if self.set_weight == None:
                    df1=N_total*self.model.T
                    weight = df1/(df1+df)
                else:
                    weight = self.set_weight
                if df==0:
                    weight=1
                    print('weight', weight)
                    self.dep_params["autoregressive_coef"] = phi_1  
                    self.dep_params["var_epsilon/1-phi^2"] = var_epsilon_1  
                    self.dep_params["var_alpha"] = var_alpha_1 
                    self.dep_params["var_td"] = 0
                else:
                    print('weight', weight)
                    self.dep_params["autoregressive_coef"] = weight* phi_1 +(1-weight)*phi_2
                    self.dep_params["var_epsilon/1-phi^2"] =weight*var_epsilon_1 +(1-weight) * var_epsilon_2
                    self.dep_params["var_alpha"] =weight*var_alpha_1 + (1-weight) * var_alpha_2
                    self.dep_params["var_td"] = 0

        else:
            self.dep_params = self.model.MC_rho

    def covariance_matrix(self, index, action_index=None):
        # dim = self.model.cluster_size[index] * self.model.T
        # R = np.zeros((dim, dim), dtype=np.float64)
        # for i in range(dim):
        #     for j in range(dim):
        #         R[i,j] = (self.dep_params["autoregressive_coef"] ** abs(i - j) * self.dep_params['var_epsilon/1-phi^2'] + self.dep_params['var_alpha']) /\
        #             (self.dep_params['var_epsilon/1-phi^2'] + self.dep_params['var_alpha'])
        
        # if action_index is not None:
        #     selected_indices = self.model.action_indices[action_index][index]
        #     R = R[selected_indices][:, selected_indices]
            
        R = np.zeros((self.model.T, self.model.T), dtype=np.float32)
        for i in range(self.model.T):
            for j in range(self.model.T):
                R[i,j] = self.dep_params["autoregressive_coef"] ** abs(i - j) * self.dep_params['var_epsilon/1-phi^2'] 
                   
        R = np.kron(np.eye(self.model.cluster_size[index]), R)
        R = R + np.ones(self.model.cluster_size[index] * self.model.T) * self.dep_params['var_alpha'] 
        R = R/(self.dep_params['var_epsilon/1-phi^2'] + self.dep_params['var_alpha']+ self.dep_params['var_td'])
        
        if action_index is not None:
            selected_indices = self.model.action_indices[action_index][index]
            R = R[selected_indices][:, selected_indices]
            
        
        return R, False

#%% autoex_sm
class Autoex_sm(statsmodels.genmod.cov_struct.CovStruct):
    def __init__(self):

        super(Autoex_sm, self).__init__()
        # dep_params = namedtuple("dep_params", ["autoregressive_part", "exchangeable_part"])
        self.dep_params = {"autoregressive_part":0,  "exchangeable_part":0}
        
    def setparams(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        self.model = model
        self.T = model.T
        self.cluster_size = model.cluster_size
        self.action_indices = model.action_indices

    def covariance_matrix(self, endog, index=None):
    
        R = np.zeros((self.T,self.T), dtype=np.float32)
        for i in range(self.T):
            for j in range(self.T):
                R[i,j] = self.dep_params["autoregressive_coef"] ** abs(i - j) * self.dep_params['var_epsilon/1-phi^2'] 
                   
        R = np.kron(np.eye(self.cluster_size[index]), R)
        R = R + np.ones(self.cluster_size[index] * self.T) * self.dep_params['var_alpha'] 
        R = R/(self.dep_params['var_epsilon/1-phi^2'] + self.dep_params['var_alpha']+self.dep_params['var_td'])
        
        selected_indices = self.action_indices[self.action_index][index]
        R = R[selected_indices][:, selected_indices]

        return R, False
    
    # def covariance_matrix(self, endog, index=None):
    #     selected_indices = self.action_indices[self.action_index][index]

    #     # Create a smaller matrix R for the selected indices
    #     dim = len(selected_indices)
    #     R = np.zeros((dim, dim), dtype=np.float64)

    #     # Efficiently fill R using broadcasting and vectorized operations
    #     row_indices, col_indices = np.meshgrid(selected_indices, selected_indices, indexing='ij')
    #     time_diff = np.abs(row_indices - col_indices)
        
    #     # Calculate the autoregressive part
    #     R = self.dep_params["autoregressive_coef"] ** time_diff * self.dep_params['var_epsilon/1-phi^2']
        
    #     # Add the exchangeable part (var_alpha)
    #     if 'var_alpha' in self.dep_params:
    #         R += np.ones((dim, dim)) * self.dep_params['var_alpha']

    #     # if 'var_td' in self.dep_params:
    #     R = R / (self.dep_params['var_epsilon/1-phi^2'] + self.dep_params['var_alpha'] + self.dep_params['var_td'])

    #     return R, False
    
    def assign_params(self, params):
        self.dep_params = params
    def update(self, params):
        pass