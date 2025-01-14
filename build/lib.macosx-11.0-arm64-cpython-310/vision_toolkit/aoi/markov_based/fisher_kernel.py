# -*- coding: utf-8 -*-

import numpy as np


class FisherKernel:
    
    def __init__(self, input, hmm_model):
        
        self.obs = input
        self.hmm_model = hmm_model

        self.fisher_vector = self.compute_fisher_vectors()


    def compute_fisher_vectors(self):
        obs = self.obs
        gamma, xi = self.hmm_model.reevaluate_moments(obs.T)
        n_s = self.hmm_model.n_s
        n_ = obs.shape[1]
        d_ = obs.shape[0]

        mu = self.hmm_model.centers
        sigma = self.hmm_model.covars
        ## Initialize Fisher vector
        fv = []

        ## Compute derivative of the log-likelihood wrt the transition matrix
        edges_post = np.sum(xi[:-1], axis=0)
        transition_mat_ = self.hmm_model.transition_matrix

        p_ = np.where(transition_mat_ > 0)
        dT = np.zeros_like(transition_mat_)
        dT[p_] = edges_post[p_] / transition_mat_[p_]

        ## Normalize by the number of observations and add to Fisher vector
        fv += list(dT.flatten() / n_)

        ## Compute derivative of the log-likelihood wrt the centers and covars
        for k in range(n_s):
            mu_ = mu[k]
            sigma_ = sigma[k]
            sigma_inv = np.linalg.inv(sigma_)

            dMu = [
                gamma[t, k] * sigma_inv @ (obs[:, t] - mu_).reshape(d_, 1)
                for t in range(n_)
            ]
            ## Sum over observations and normalize by the number of observations
            dMu = np.sum(dMu, axis=0) / n_
            ## Add to Fisher vector
            fv += list(dMu.flatten())

            dCov = [
                gamma[t, k]
                * sigma_inv
                @ (obs[:, t] - mu_).reshape(d_, 1)
                @ (obs[:, t] - mu_).reshape(1, d_)
                @ sigma_inv
                - sigma_inv
                for t in range(n_)
            ]
            ## Sum over observations and normalize by the number of observations
            dCov = 0.5 * np.sum(dCov, axis=0) / n_
            ## Add to Fisher vector
            fv += list(dCov.flatten())

        return np.array(fv)
