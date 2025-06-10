# -*- coding: utf-8 -*-

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from vision_toolkit.aoi.markov_based.c_HMM import c_HMM


class AoIHMM:
    def __init__(
        self, Z, centers_, covars_, initial_distribution_, transition_mat_, n_iter
    ):
        """


        Parameters
        ----------
        Z : TYPE
            DESCRIPTION.
        centers_ : TYPE
            DESCRIPTION.
        covars_ : TYPE
            DESCRIPTION.
        initial_distribution_ : TYPE
            DESCRIPTION.
        transition_mat_ : TYPE
            DESCRIPTION.
        n_iter : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.obs = input
        self.aoi_seq = Z

        self.n_iter = n_iter

        self.centers = centers_
        self.covars = covars_
        self.n_s = centers_.shape[0]

        self.initial_distribution = initial_distribution_
        self.transition_matrix = transition_mat_

    def infer_parameters(self, observations):
        """


        Parameters
        ----------
        observations : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

      
        Z, theta, moments = c_HMM.baum_welch(
            observations.T, self.n_s, self.n_iter, self.centers
        )
        self.aoi_seq = Z

        self.initial_distribution = theta[0]
        self.transition_matrix = theta[1]
        self.centers = theta[2]
        self.covars = theta[3]

    def reevaluate_moments(self, observations):
        """


        Parameters
        ----------
        observations : TYPE
            DESCRIPTION.

        Returns
        -------
        gamma : TYPE
            DESCRIPTION.
        xi : TYPE
            DESCRIPTION.

        """

        (gamma, xi) = c_HMM.evaluate_posterior_moments(
            observations,
            self.centers,
            self.covars,
            self.initial_distribution,
            self.transition_matrix,
        )
        return gamma, xi


    ## Not used -> C implementation
    def log_baum_welch(self):
        """
        Log implementation of the Baum–Welch algorithm.

        Returns
        -------
        theta : TYPE
            DESCRIPTION.

        """
        x_s = self.obs
        n_s = self.n_s

        n_ = x_s.shape[1]
        d_ = x_s.shape[0]

        # Initialize parameter set
        log_pi = np.log(np.ones(n_s) / n_s)
        log_T = np.log(np.ones((n_s, n_s)) * (1 / n_s))

        _means = self.centers_
        _vars = np.zeros((n_s, d_, d_))

        for i in range(n_s):
            _vars[i] = np.diag(np.ones(2)) * 1

        # Initialize emission probabilities (size n_ × n_s).
        log_emm_prob = np.zeros((n_, n_s))

        for k in range(n_s):
            log_emm_prob[:, k] = multivariate_normal.logpdf(x_s.T, _means[k], _vars[k])

        log_alpha = np.zeros((n_, n_s))
        log_beta = np.zeros((n_, n_s))

        print("Processing Baum Welch...")

        iter_ = 0
        while iter_ < self.n_iter:
            print("--- Iteration number: %s ---" % (iter_ + 1))
            # E-step forward-backward algorithm.
            for k in range(n_s):
                log_alpha[0, k] = log_pi[k] + log_emm_prob[0, k]

            for n in range(1, n_):
                for k in range(n_s):
                    tmp = np.array(
                        [log_alpha[n - 1, j] + log_T[j, k] for j in range(n_s)]
                    )
                    log_alpha[n, k] = logsumexp(tmp) + log_emm_prob[n, k]

            log_beta[n_ - 1] = 0

            for n in reversed(range(n_ - 1)):
                for k in range(n_s):
                    tmp = np.array(
                        [
                            log_beta[n + 1, j] + log_emm_prob[n + 1, j] + log_T[k, j]
                            for j in range(n_s)
                        ]
                    )

                    log_beta[n, k] = logsumexp(tmp)

            # Compute first posterior moment gamma
            log_gamma = log_alpha + log_beta
            log_evidence = logsumexp(log_alpha[n_ - 1])

            log_gamma -= log_evidence
            gamma = np.exp(log_gamma)
            gamma /= np.sum(gamma, axis=1).reshape((n_, 1))

            # Compute second posterior moment xi.
            log_xi = np.empty((n_, n_s, n_s))

            for n in range(n_ - 1):
                tmp = np.empty((n_s, n_s))

                for k in range(n_s):
                    for j in range(n_s):
                        tmp[k, j] = (
                            log_alpha[n, k]
                            + log_beta[n + 1, j]
                            + log_emm_prob[n + 1, j]
                            + log_T[k, j]
                            - log_evidence
                        )

                log_xi[n] = tmp

            log_pi = log_gamma[0] - logsumexp(log_gamma[0])

            for i in range(n_s):
                for j in range(n_s):
                    log_T[i, j] = (
                        logsumexp(log_xi[1:, i, j]) - logsumexp(log_xi[1:, i, :]) + 1e-8
                    )
                    if math.isnan(log_T[i, j]):
                        log_T[i, j] = (
                            logsumexp(log_xi[1:, i, j][~np.isnan(log_xi[1:, i, j])])
                            - logsumexp(log_xi[1:, i, :][~np.isnan(log_xi[1:, i, :])])
                            + 1e-8
                        )

            # Compute new means.
            for k in range(n_s):
                _means[k] = x_s @ gamma[:, k]
                _means[k] /= np.sum(gamma[:, k]) + 1e-8

            # Compute new covariances.
            for k in range(n_s):
                _vars[k] = np.zeros((d_, d_))
                for n in range(n_):
                    dev = x_s[:, n] - _means[k]
                    _vars[k] += gamma[n, k] * np.outer(dev, dev.T)

                _vars[k] /= np.sum(gamma[:, k]) + 1e-8
                _vars[k] += np.diag(1e-8 * np.ones(d_))

            # Recompute emission probabilities using inferred parameters.
            for k in range(n_s):
                log_emm_prob[:, k] = multivariate_normal.logpdf(
                    x_s.T, _means[k], _vars[k]
                )

            iter_ += 1

        # print(np.exp(log_T))
        Z = np.argmax(gamma, axis=1)
        theta = (log_pi, log_T, _means, log_emm_prob)
        return Z, theta
