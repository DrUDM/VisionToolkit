# -*- coding: utf-8 -*-


 
import numpy as np
from typing import List, Tuple, Iterator

from scipy.stats import rv_continuous
from hmmlearn.hmm import _BaseHMM
 
from scipy.stats import norm
from scipy.special import logsumexp


class EMHistory:
    def __init__(self, converged, iterations, logtots):
        self.converged = converged
        self.iterations = iterations
        self.logtots = logtots

def loglikelihood(base, reduced, z, lhmm, N, eps=1e-10):
    total = 0
    for i in range(len(base)):
        for j in range(len(reduced)):
            total += z[i, j] * (np.log(reduced['ω'][j] / (z[i, j] + eps)) + N * lhmm[i, j])
    return total

def cluster(base, reduced, τ, N, maxiter=100, tol=1e0):
    assert maxiter >= 0, "maxiter must be non-negative"
    
    z = np.zeros((len(base), len(reduced)))
    history = EMHistory(converged=False, iterations=0, logtots=[])
    
    logtot = -float('inf')
    logtotp = logtot
    
    for it in range(1, maxiter + 1):
        try:
            # Placeholder for vhem_step, which must be implemented
            reduced, lhmm, z = vhem_step(base, reduced, τ, N)
            logtotp = loglikelihood(base, reduced, z, lhmm, N)
        except Exception as e:
            print(f"Error: {e}")
            break
        
        history.logtots.append(logtotp)
        history.iterations += 1
        
        print(f"Iteration {it}: logtot = {logtotp}, diff = {logtotp - logtot}")
        
        if logtotp - logtot < tol:
            history.converged = True
            break
        
        logtot = logtotp
    
    return history, z, reduced

# Example placeholder for vhem_step function
def vhem_step(base, reduced, τ, N):
    # This function needs to be implemented.
    # Returning dummy values for demonstration purposes
    reduced = {'ω': np.ones(len(reduced))}
    lhmm = np.random.rand(len(base), len(reduced))
    z = np.random.rand(len(base), len(reduced))
    return reduced, lhmm, z





# Placeholder classes for HMM, H3M, MixtureModel, etc.
class H3M:
    def __init__(self, models, weights):
        self.M = models  # List of HMMs
        self.ω = weights  # Weights of the HMMs

class HMM:
    def __init__(self, a, A, B):
        self.a = a  # Initial probabilities
        self.A = A  # Transition matrix
        self.B = B  # Emission distributions (list of MixtureModel)

class MixtureModel:
    def __init__(self, components, priors):
        self.components = components  # List of component distributions
        self.prior = priors  # Prior probabilities of components

class Normal:
    def __init__(self, mean, std_dev):
        self.μ = mean
        self.σ = std_dev

# LogSumExp accumulator (a numerical stable way to compute log-sum-exp)
class LogSumExpAcc:
    def __init__(self):
        self.values = []

    def add(self, value):
        self.values.append(value)

    def sum(self):
        max_val = max(self.values)
        return max_val + np.log(sum(np.exp(val - max_val) for val in self.values))

def vhem_step_E(base, reduced, τ, N):
    num_base = len(base.M)
    num_reduced = len(reduced.M)

    # Initialize storage for expectations and aggregates
    logη = [[None for _ in range(num_reduced)] for _ in range(num_base)]
    lhmm = np.zeros((num_base, num_reduced))
    νagg1 = [[None for _ in range(num_reduced)] for _ in range(num_base)]
    νagg = [[None for _ in range(num_reduced)] for _ in range(num_base)]
    ξagg = [[None for _ in range(num_reduced)] for _ in range(num_base)]
    logz = np.zeros((num_base, num_reduced))
    
    logωj = np.log(reduced.ω)

    for i in range(num_base):
        Mi = base.M[i]
        logai = np.log(Mi.a)
        logAi = np.log(Mi.A)
        logzacc = LogSumExpAcc()

        for j, Mj in enumerate(reduced.M):
            Ki, Kj = len(Mi.a), len(Mj.a)
            logν = np.zeros((τ, Kj, Ki))
            logξ = np.zeros((τ, Kj, Kj, Ki))

            νagg1[i][j] = np.zeros(Kj)
            νagg[i][j] = np.zeros((Kj, Ki))
            ξagg[i][j] = np.zeros((Kj, Kj))

            logη[i][j], _, (lhmm[i, j], logϕ, logϕ1) = loglikelihood_va(Mi, Mj, τ)

            for ρ in range(Kj):
                for β in range(Ki):
                    logν[0, ρ, β] = logai[β] + logϕ1[β, ρ]
                    νagg1[i][j][ρ] += np.exp(logν[0, ρ, β])
                    νagg[i][j][ρ, β] += np.exp(logν[0, ρ, β])

            for t in range(1, τ):
                logtmps = np.zeros((Kj, Ki))

                for β in range(Ki):
                    for ρp in range(Kj):
                        acc = LogSumExpAcc()
                        for βp in range(Ki):
                            acc.add(logν[t - 1, ρp, βp] + logAi[βp, β])
                        logtmps[ρp, β] = acc.sum()

                for ρ in range(Kj):
                    for β in range(Ki):
                        acc = LogSumExpAcc()
                        for ρp in range(Kj):
                            logξ[t, ρp, ρ, β] = logtmps[ρp, β] + logϕ[t, β, ρp, ρ]
                            ξagg[i][j][ρp, ρ] += np.exp(logξ[t, ρp, ρ, β])
                            acc.add(logξ[t, ρp, ρ, β])
                        logν[t, ρ, β] = acc.sum()
                        νagg[i][j][ρ, β] += np.exp(logν[t, ρ, β])

            logz[i, j] = logωj[j] + (N * base.ω[i] * lhmm[i, j])
            logzacc.add(logz[i, j])

        logz[i, :] -= logzacc.sum()

    return logz, logη, lhmm, νagg, νagg1, ξagg

def vhem_step(base, reduced, τ, N):
    logz, logη, lhmm, νagg, νagg1, ξagg = vhem_step_E(base, reduced, τ, N)
    z = np.exp(logz)

    # M-step
    newω = np.sum(z, axis=0) / np.sum(z)
    newM = []

    for j in range(len(reduced.M)):
        Mj = reduced.M[j]
        Kj = len(Mj.a)

        # Update initial probabilities
        newa = np.zeros(Kj)
        norm = 0
        for ρ in range(Kj):
            newa[ρ] = sum(z[i, j] * base.ω[i] * νagg1[i][j][ρ] for i in range(len(base.M)))
            norm += newa[ρ]
        newa /= norm

        # Update transition matrix
        newA = np.zeros((Kj, Kj))
        for ρ in range(Kj):
            norm = 0
            for ρp in range(Kj):
                for i in range(len(base.M)):
                    newA[ρ, ρp] += z[i, j] * base.ω[i] * ξagg[i][j][ρ, ρp]
                norm += newA[ρ, ρp]
            newA[ρ, :] /= norm

        # Update observation mixtures
        newB = []
        for ρ in range(Kj):
            components = []
            weights = []
            for l in range(len(Mj.B[ρ].components)):
                # Calculate new parameters (mean and variance) for the normal distributions
                mean = 0  # Compute mean here
                variance = 0  # Compute variance here
                weights.append(0)  # Update weights
                components.append(Normal(mean, np.sqrt(variance)))
            newB.append(MixtureModel(components, weights))

        newM.append(HMM(newa, newA, newB))

    return H3M(newM, newω), lhmm, z



class H3M:
    """
    H3M class representing a collection of HMMs with associated weights.
    """
    def __init__(self, M: List['AbstractHMM'], ω: List[float] = None):
        if ω is None:
            ω = [1 / len(M)] * len(M)  # Uniform weights if not provided
        self._assert_h3m(M, ω)
        self.M = M  # List of HMMs
        self.ω = np.array(ω)  # Probabilistic weights (normalized)

    @staticmethod
    def _assert_h3m(M: List['AbstractHMM'], ω: List[float]) -> bool:
        if len(M) != len(ω):
            raise ValueError("The number of HMMs must match the number of weights.")
        if not np.isclose(sum(ω), 1.0) or any(w < 0 for w in ω):
            raise ValueError("Weights must form a valid probability vector (non-negative and sum to 1).")
        return True

    def __len__(self) -> int:
        """
        Returns the number of HMMs in the collection.
        """
        return len(self.M)

    def __iter__(self) -> Iterator[Tuple['AbstractHMM', float]]:
        """
        Iterates over the HMMs and their corresponding weights.
        """
        return iter(zip(self.M, self.ω))

# Utility Functions
def is_prob_vector(vector: List[float]) -> bool:
    """
    Checks if a vector is a valid probability vector.
    """
    return np.isclose(sum(vector), 1.0) and all(v >= 0 for v in vector)


class LogSumExpAcc:
    """
    Accumulator for computing log-sum-exp efficiently.
    """
    def __init__(self):
        self.m = float('-inf')  # Current maximum value
        self.s = 0.0  # Scaled sum of exponentials

    def add(self, val: float):
        """
        Adds a single value to the accumulator.
        """
        if val == float('-inf'):
            return
        elif val <= self.m:
            self.s += math.exp(val - self.m)
        else:
            self.s *= math.exp(self.m - val)
            self.s += 1.0
            self.m = val

    def add_multiple(self, vals: List[float]):
        """
        Adds multiple values to the accumulator.
        """
        for val in vals:
            self.add(val)

    def sum(self) -> float:
        """
        Returns the log-sum-exp of the accumulated values.
        """
        return math.log(self.s) + self.m
    
    
    
    


def loglikelihood_mc_distribution(a: rv_continuous, b: rv_continuous, N: int) -> float:
    """
    Monte Carlo estimation of log-likelihood for two distributions.
    
    Parameters:
    - a: Distribution to sample from.
    - b: Distribution to evaluate log-likelihood.
    - N: Number of samples.

    Returns:
    - Mean log-likelihood of samples.
    """
    samples = a.rvs(size=N)
    log_likelihoods = b.logpdf(samples)
    return np.mean(log_likelihoods)

def loglikelihood_mc_hmm(a: _BaseHMM, b: _BaseHMM, τ: int, N: int) -> float:
    """
    Monte Carlo estimation of log-likelihood for two HMMs.

    Parameters:
    - a: HMM to sample from.
    - b: HMM to evaluate log-likelihood.
    - τ: Length of sequences to sample.
    - N: Number of sequences.

    Returns:
    - Mean log-likelihood of sequences.
    """
    log_likelihoods = []
    for _ in range(N):
        # Generate a sequence from HMM `a` of length `τ`
        sequence, _ = a.sample(τ)
        # Compute log-likelihood for HMM `b`
        log_likelihood = b.score(sequence)
        log_likelihoods.append(log_likelihood)
    return np.mean(log_likelihoods)





# Variational lower bound for Gaussian mixtures
def loglikelihood_normal(a, b):
    if b.stddev == 0:
        return np.inf if a.mean == b.mean else 0
    else:
        return -0.5 * (
            np.log(2 * np.pi)
            + np.log(b.stddev**2)
            + (a.stddev**2 / b.stddev**2)
            + ((b.mean - a.mean)**2 / b.stddev**2)
        )

def loglikelihood_va_mixture(a, b):
    I, J = len(a.weights), len(b.weights)

    lb = 0.0
    logl = np.zeros((I, J))
    logeta = np.zeros((I, J))

    logp = np.log(b.weights)

    for i, ai in enumerate(a.components):
        for j, bj in enumerate(b.components):
            logl[i, j] = loglikelihood_normal(ai, bj)
            logeta[i, j] = logp[j] + logl[i, j]
        logeta[i, :] -= logsumexp(logeta[i, :])

    for i in range(I):
        p = a.weights[i]
        for j in range(J):
            lb += p * np.exp(logeta[i, j]) * (logp[j] - logeta[i, j] + logl[i, j])

    return lb, logeta

# Variational lower bound for HMMs
def loglikelihood_va_hmm(hmm1, hmm2, lgmm, τ):
    assert hmm1.n_components == lgmm.shape[0]
    assert hmm2.n_components == lgmm.shape[1]
    assert τ >= 0

    K, L = hmm1.n_components, hmm2.n_components

    lhmm = 0.0
    logl = np.zeros((τ + 1, K, L))
    logphi = np.zeros((τ, K, L, L))
    logphi1 = np.zeros((K, L))

    loga2 = np.log(hmm2.startprob_)
    logA2 = np.log(hmm2.transmat_)

    for t in range(τ, 1, -1):
        for β in range(K):
            for ρp in range(L):
                for ρ in range(L):
                    logphi[t, β, ρp, ρ] = (
                        logA2[ρp, ρ] + lgmm[β, ρ] + logl[t + 1, β, ρ]
                    )

                norm = logsumexp(logphi[t, β, ρp, :])
                logphi[t, β, ρp, :] -= norm

                for βp in range(K):
                    logl[t, βp, ρp] += hmm1.transmat_[βp, β] * norm

    for β in range(K):
        for ρ in range(L):
            logphi1[β, ρ] = loga2[ρ] + lgmm[β, ρ] + logl[2, β, ρ]

        norm = logsumexp(logphi1[β, :])
        logphi1[β, :] -= norm

        lhmm += hmm1.startprob_[β] * norm

    return lhmm, logphi, logphi1

def loglikelihood_va(hmm1, hmm2, τ):
    Ki, Kj = hmm1.n_components, hmm2.n_components

    logeta = np.empty((Ki, Kj), dtype=object)
    lgmm = np.zeros((Ki, Kj))

    for β, Miβ in enumerate(hmm1.emissionprob_):
        for ρ, Mjρ in enumerate(hmm2.emissionprob_):
            lb, logeta[β, ρ] = loglikelihood_va_mixture(Miβ, Mjρ)
            lgmm[β, ρ] = lb

    return logeta, lgmm, loglikelihood_va_hmm(hmm1, hmm2, lgmm, τ)
