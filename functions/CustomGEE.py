# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:06:28 2024

@author: Lenovo
"""

import statsmodels.api as sm
import numpy as np

class CustomGEE(sm.GEE):
    def _update_mean_params(self):
        """
        Returns
        -------
        update : array_like
            The update vector such that params + update is the next
            iterate when solving the score equations.
        score : array_like
            The current value of the score equations, not
            incorporating the scale parameter.  If desired,
            multiply this vector by the scale parameter to
            incorporate the scale.
        """

        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, "weights_li", None)

        cached_means = self.cached_means

        varfunc = self.family.variance

        bmat, score = 0, 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat

            rslt = self.cov_struct.covariance_matrix_solve(
                    expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return None, None
            vinv_d, vinv_resid = tuple(rslt)
            
            if self.cov_struct.GEE_model.expected_next_states_action_list[self.cov_struct.action_index][i] is None:
                bmat += np.dot(dmat.T, vinv_d)
                score += np.dot(dmat.T, vinv_resid)
            else:
                bmat += vinv_d
                score += vinv_resid

        try:
            update = np.linalg.solve(bmat, score)
        except np.linalg.LinAlgError:
            update = np.dot(np.linalg.pinv(bmat), score)

        self._fit_history["cov_adjust"].append(
            self.cov_struct.cov_adjust)

        return update, score
    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        cov_robust : array_like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        cov_naive : array_like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        cmat : array_like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """

        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, "weights_li", None)
        varfunc = self.family.variance
        cached_means = self.cached_means

        # Calculate the naive (model-based) and robust (sandwich)
        # covariances.
        bmat, cmat = 0, 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat

            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return None, None, None, None
            vinv_d, vinv_resid = tuple(rslt)
            
            if self.cov_struct.GEE_model.expected_next_states_action_list[self.cov_struct.action_index][i] is None:
                bmat += np.dot(dmat.T, vinv_d)
                dvinv_resid = np.dot(dmat.T, vinv_resid)
            else:
                bmat += vinv_d
                dvinv_resid =  vinv_resid
                
            cmat += np.outer(dvinv_resid, dvinv_resid)

        scale = self.estimate_scale()

        try:
            bmati = np.linalg.inv(bmat)
        except np.linalg.LinAlgError:
            bmati = np.linalg.pinv(bmat)

        cov_naive = bmati * scale
        cov_robust = np.dot(bmati, np.dot(cmat, bmati))

        cov_naive *= self.scaling_factor
        cov_robust *= self.scaling_factor
        return cov_robust, cov_naive, cmat
    
    def _bc_covmat(self, cov_naive):

        cov_naive = cov_naive / self.scaling_factor
        endog = self.endog_li
        exog = self.exog_li
        varfunc = self.family.variance
        cached_means = self.cached_means
        scale = self.estimate_scale()

        bcm = 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (dmat,))
            if rslt is None:
                return None
            vinv_d = rslt[0]
            vinv_d /= scale

            hmat = np.dot(vinv_d, cov_naive)
            hmat = np.dot(hmat, dmat.T).T

            f = self.weights_li[i] if self.weights is not None else 1.

            aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (aresid,))
            if rslt is None:
                return None
            srt = rslt[0]
            if self.cov_struct.GEE_model.expected_next_states_action_list[self.cov_struct.action_index][i] is None:
                srt = f * np.dot(dmat.T, srt) / scale
            else:
                srt = f * srt / scale

        cov_robust_bc = np.dot(cov_naive, np.dot(bcm, cov_naive))
        cov_robust_bc *= self.scaling_factor

        return cov_robust_bc
# Usage of your customized GEE class
# my_model = CustomGEE(...)
# my_result = my_model.fit(...)
