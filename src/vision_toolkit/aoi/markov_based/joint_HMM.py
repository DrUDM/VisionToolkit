# -*- coding: utf-8 -*-

from scipy.special import logsumexp
import numpy as np

 

np.random.seed(1)







class JointHMM:
 
    def __init__(self, 
                 hmm_base, 
                 nb_states,
                 n_iters = 5, base_weights = None,
                 n_vsm = 50, n_vseq = 15):
        """
        Inputs:
            - hmm_base = base hidden Markov mixture model
            - nb_states = number of states for each hmm from both  the base 
                          model and the reduced model
            - n_iters = number of iterations for the the variational em
        """
        
        self.hmm_base = hmm_base
        
        self.n_vseq = n_vseq
        self.n_vsm = n_vsm
        
        self.K_b = len(hmm_dict)
        self.n_s = nb_states
        
        if base_weights == None: 
            self.w_b = np.ones(self.K_b) * (1/self.K_b)
        
        else:
            self.w_b = base_weights    
            
        self.n_i = n_iters
        
        self.variational_em(K_r = 2)
        
        
    def variational_em(self, K_r):
        
        K_b = self.K_b
        n_s = self.n_s

        n_vseq = self.n_vseq
        n_vsm = self.n_vsm
        
        hmm_b = self.hmm_base
        w_b = self.w_b

        # Initialize reduced weigths
        w_r = np.ones(K_r) * (1/K_r)
        
        # Randomly initialize reduced HMM
        hmm_r = {}

        for k in range(K_r):
            
            pi = np.random.random(n_s)
            
            mu = np.random.random((2, n_s))
            sgm = np.zeros((2, 2, n_s))
            for n_ in range(n_s):
                tpm = np.random.random((2, 2))
                sgm[:,:,n_] = tpm @ tpm.T
           
            
            t_m = np.random.random((n_s, n_s))
            for n_ in range(n_s):
                t_m[n_] /= np.sum(t_m[n_])
            
            hmm = [pi, t_m, mu, sgm]
            
            hmm_r.update({k: hmm})
    
        for _ in range(self.n_i):
                
            # Process E-Step
            
            # Compute partial lower bound for each pair of HMM (B, R) 
            # and each pair of emission state
            p_lb = np.ones((K_b, K_r, n_s, n_s)) * -np.log(2*np.pi)
            
            for k_b in range(K_b):
                for k_r in range(K_r):
                    for n_r in range(n_s):
                        
                        mu_r = hmm_r[k_r][2][:,n_r]
                        
                        # Only compute it once
                        sgm_r = hmm_r[k_r][3][:,:,n_r]
                        sgm_r_inv = np.linalg.inv(sgm_r)
                        sgm_r_det = np.linalg.det(sgm_r)
                        
                        for n_b in range(n_s):
                            
                            mu_b = hmm_b[k_b][2][:,n_b]
                            sgm_b = hmm_b[k_b][3][:,:,n_b]
                            
                            tmp = np.log(sgm_r_det) 
                            tmp += np.trace(sgm_r_inv @ sgm_b) 
                            tmp += ((mu_r - mu_b) 
                                    @ sgm_r_inv 
                                    @ (mu_r - mu_b))
                             
                            p_lb[k_b, k_r, n_b, n_r] -= 0.5 * tmp
                            
            # Recursively compute the complete lower bound for 
            # each pair of HMM (B, R)  
            c_lb = np.zeros((K_b, K_r, n_vseq + 1, n_s, n_s)) 
            log_phi = np.zeros((K_b, K_r, n_vseq, n_s, n_s, n_s))
             
            for t in reversed(range(1, n_vseq)):
                for k_b in range(K_b):
                    trans_m_b = hmm_b[k_b][1]
                    
                    for k_r in range(K_r):
                        trans_m_r = hmm_r[k_r][1] 
                        
                        for s_n_r in range(n_s):
                            for s_n_b in range(n_s):
                                
                                # Compte the log-phi variable
                                for n_r in range(n_s):
                                    
                                    loc_c_lb = c_lb[k_b, k_r, t+1, n_r, s_n_b]
                                    loc_p_lb = p_lb[k_b, k_r, s_n_b, n_r]
                                     
                                    log_phi[k_b, k_r, t, 
                                            s_n_r, s_n_b, n_r] = (np.log(trans_m_r[s_n_r, n_r]) 
                                                                  + loc_p_lb 
                                                                  + loc_c_lb)
                                 
                                log_phi[k_b, k_r, t, 
                                        s_n_r, s_n_b] -= logsumexp(log_phi[k_b, k_r, t, 
                                                                           s_n_r, s_n_b])
                                  
                                # Compute the complete lower bound
                                tmp = 0
                                for n_ in range(n_s):
                                    
                                    l_l =np.array(
                                        [np.log(trans_m_r[s_n_r, l]) 
                                         + p_lb[k_b, k_r, n_, l] 
                                         + c_lb[k_b, k_r, t+1, l, n_]  for l in range(n_s)]
                                            )
                                    tmp += trans_m_b[s_n_b, n_] * logsumexp(l_l)
                                  
                                c_lb[k_b, k_r, t, s_n_r, s_n_b] = tmp 
             
            # Initialize terminal log-phi variable and terminal complete lower bound
            log_phi_t = np.zeros((K_b, K_r, n_s, n_s))
            lb = np.zeros((K_b, K_r))
            
            for k_b in range(K_b): 
                pi_b = hmm_b[k_b][0]
                
                for k_r in range(K_r): 
                    pi_r = hmm_r[k_r][0]
                     
                    for s_n_b in range(n_s):
                        
                        # Compute the terminal log-phi variable
                        for n_r in range(n_s):
                            
                            loc_c_lb = c_lb[k_b, k_r, 1, n_r, s_n_b]
                            loc_p_lb = p_lb[k_b, k_r, s_n_b, n_r]
                            
                            log_phi_t[k_b, k_r, 
                                      s_n_b, n_r] = (np.log(pi_r[n_r]) 
                                                     + loc_p_lb 
                                                     + loc_c_lb)
                            
                        log_phi_t[k_b, k_r, 
                                  s_n_b] -= logsumexp(log_phi_t[k_b, k_r, 
                                                                s_n_b])
                    
                    # Compute the terminal complete lower bound                                                       
                    tmp = 0
                    for n_ in range(n_s): 
                        
                        l_l =np.array(
                            [np.log(pi_r[l]) 
                             + p_lb[k_b, k_r, n_, l] 
                             + c_lb[k_b, k_r, 1, l, n_]  for l in range(n_s)]
                                )
                        
                        tmp += pi_b[n_] * logsumexp(l_l)
                        
                    lb[k_b, k_r] = tmp 
               
            # Compute the log-assignment variable z(i,j)
            log_a_v = np.zeros((K_b, K_r)) 
            
            for k_b in range(K_b): 
                for k_r in range(k_r): 
                    
                    log_a_v[k_b, k_r] = np.log(w_r[k_r]) + (n_vsm * w_b[k_b] * lb[k_b, k_r])
                
                log_a_v[k_b] -= logsumexp(log_a_v[k_b])
                
                
            # Recursively compute the summary variables xi and nu
            log_xi = np.zeros((K_b, K_r, n_vseq, n_s, n_s, n_s))
            log_nu = np.zeros((K_b, K_r, n_vseq, n_s, n_s)) 
            
            # Compute initial log-nu
            for k_b in range(K_b):
                pi_b = hmm_b[k_b][0] 
                
                for k_r in range(K_r):   
                    for n_b in range(n_s): 
                        log_nu[k_b, k_r, 0, n_b] = np.log(pi_b[n_b]) 
                        
                        for n_r in range(n_s): 
                            log_nu[k_b, k_r, 0, n_b, n_r] += log_phi_t[k_b, k_r, n_b, n_r]
               
            # Recursively compute the summary variables xi and nu for t=2,...,n_vseq
            for t in range(1, n_vseq): 
                for k_b in range(K_b):  
                    trans_m_b = hmm_b[k_b][1] 
                    
                    for k_r in range(K_r):  
                        for n_r in range(n_s): 
                            for n_b in range(n_s): 
                                for n_r_p in range(n_s):
                                    
                                    tmp = np.array(
                                        [log_nu[k_b, k_r, t-1, n_b_p, n_r_p] 
                                         + trans_m_b[n_b_p, n_b] 
                                         for n_b_p in range(n_s)]
                                            )
                                    
                                    # Compute log-xi
                                    log_xi[k_b, k_r, t, 
                                           n_r, n_b, n_r_p] = (logsumexp(tmp) 
                                                               + log_phi[k_b, k_r, t, 
                                                                         n_r, n_b, n_r_p])
                                
                                # Compute log-nu           
                                log_nu[k_b, k_r, t,
                                       n_b, n_r] = logsumexp(log_xi[k_b, k_r, t, 
                                                                    n_r, n_b])
                          
                                                                     
            # Compute log-init nu, log-sum nu and log-sum xi
            log_i_nu = np.zeros((K_b, K_r, n_s))
            log_s_nu = np.zeros((K_b, K_r, n_s, n_s))
            log_sum_xi = np.zeros((K_b, K_r, n_s, n_s))
            
            for k_b in range(K_b): 
                for k_r in range(K_r):   
                    for n_r in range(n_s): 
            
                        log_i_nu[k_b, k_r, n_r] = logsumexp(log_nu[k_b, k_r, 0, :, n_r])
                        
                        for n_b in range(n_s):
                            log_s_nu[k_b, k_r, n_b, n_r] = logsumexp(log_nu[k_b, k_r, :, n_b, n_r])
            
                        for n_r_p in range(n_s):
                            log_sum_xi[k_b, k_r, n_r, n_r_p] = logsumexp(np.array(
                                [logsumexp(log_xi[k_b, k_r, t, n_r, :, n_r_p]) for t in range(1, n_vseq)]
                                    ))
                            
            
            # Process M-Step
    
            # Compute new weigths for the reduced model
            for k_r in range(K_r):
                w_r[k_r] = np.exp(logsumexp(log_a_v[:, k_r]) - np.log(K_b))
             
            # Compute new initial distribution pi for the reduced model
            for k_r in range(K_r):
                
                for n_r in range(n_s):
                    
                    tmp = log_a_v[:, k_r] + log_i_nu[:, k_r, n_r] + w_b
                    hmm_r[k_r][0][n_r] = logsumexp(tmp)
                     
                hmm_r[k_r][0]  -= logsumexp(hmm_r[k_r][0])
                hmm_r[k_r][0] = np.exp(hmm_r[k_r][0])
               
            # Compute new transition probabilities for the reduced model
            for k_r in range(K_r):  
                for n_r in range(n_s):
                    
                    for n_r_p in range(n_s):
                        
                        tmp = log_a_v[:, k_r] + w_b + log_sum_xi[:, k_r, n_r, n_r_p]
                        hmm_r[k_r][1][n_r, n_r_p] = logsumexp(tmp)   
                        
                    hmm_r[k_r][1][n_r] -= logsumexp(hmm_r[k_r][1][n_r])
                    hmm_r[k_r][1][n_r] = np.exp(hmm_r[k_r][1][n_r])
                     
                    
            # Function to compute the log-weighted sum operator 
            def log_omega_function(f, k_r, n_r):
        
                tmp_1 = [] 
                
                for k_b in range(K_b):
                    
                    # Problem with f[k_b, n_b] which can be negative
                    tmp_2 = logsumexp(np.array(
                        [log_s_nu[k_b, k_r, n_b, n_r] + np.log(np.abs(f[k_b, n_b])) for n_b in range(n_s)]
                            ), axis = 0)
                     
                    tmp_2 += np.log(w_b[k_b]) + log_a_v[k_b, k_r]
                    
                    tmp_1.append(tmp_2)
                    
                return logsumexp(np.array(tmp_1), axis = 0)
                    
            
            # Function to compute the weighted sum operator 
            def omega_function(f, k_r, n_r):
                
                tmp_1 = []
                
                for k_b in range(K_b):
                    
                    tmp_2 = np.sum(np.array(
                        [np.exp(log_s_nu[k_b, k_r, n_b, n_r]) * f[k_b, n_b] for n_b in range(n_s)]
                            ), axis = 0)
                     
                    tmp_2 *= w_b[k_b] * np.exp(log_a_v[k_b, k_r])
                    
                    tmp_1.append(tmp_2)
                    
                return np.sum(np.array(tmp_1), axis = 0)
            
            
            # Create matrix to compute variance matrices          
            sgm_m = np.zeros((K_r, n_s, K_b, n_s, 
                              2, 2))
            for k_r in range(K_r):
                for n_r in range(n_s):
                    for k_b in range(K_b):
                        for n_b in range(n_s):
                            
                            d_m = hmm_b[k_b][2][:, n_b] - hmm_r[k_r][2][:, n_r]
                            sgm_m[k_r, n_r, k_b, n_b] = (hmm_b[k_b][3][:,:,n_b]
                                                         + d_m.reshape(2,1) @ d_m.reshape(1,2))
                     
            # Create matrix to compute mean vectors
            mu_m = np.zeros((K_b, n_s, 2))
            for k_b in range(K_b):
                for n_b in range(n_s):
                    mu_m[k_b, n_b] = hmm_b[k_b][2][:, n_b]
                    
            one_m = np.ones((K_b, n_s))
             
            # Compute new mean vectors and variance matrices for the reduced model
            for k_r in range(K_r):
            
                for n_r in range(n_s):
           
                    '''
                    denom = log_omega_function(one_m, k_r, n_r)
                    
                    hmm_r[k_r][3][:,:,n_r] = np.exp(log_omega_function(sgm_m[k_r, n_r], k_r, n_r)
                                                    - denom) 
                    print(hmm_r[k_r][3][:,:,n_r])
                    
                    hmm_r[k_r][2][:, n_r] = np.exp(log_omega_function(mu_m, k_r, n_r)
                                                   - denom) 
                    '''
                    
                    denom = omega_function(one_m, k_r, n_r)
                    
                    hmm_r[k_r][3][:,:,n_r] = (omega_function(sgm_m[k_r, n_r], k_r, n_r) 
                                              / denom) 
                    hmm_r[k_r][2][:, n_r] = (omega_function(mu_m, k_r, n_r) 
                                             / denom)
                                    
                 
                     
                    
                    
                    
'''      
nb_states = 3
K_b = 4
hmm_dict = {}

for k in range(K_b):
    
    pi = np.random.random(nb_states)
    T_mat = np.random.random((nb_states, nb_states))
    mu = np.random.random((2, nb_states)) 
    
    
    Sigma = np.zeros((2, 2, nb_states))
    for n_ in range(nb_states):
        tpm = np.random.random((2, 2))
        Sigma[:,:,n_] = tpm @ tpm.T
    
    hmm = [pi, T_mat, mu, Sigma]
    
    hmm_dict.update({k: hmm})
  
j_hmm = JointHMM(hmm_dict, nb_states)   
    
    
'''    









class VHEM:
 
    def __init__(self, 
                 base, reduce,  
                 n_iters = 5,
                 n_vsm = 100):

        self.base = base
        self.reduce = reduce

        self.K_b = base.K
        self.K_r = reduce.K
        
        self.tau = n_vsm
        
        ##To improve
        partial_lb = np.zeros((self.K_b, self.K_r, 15, 15, 15, 15))
        log_eta = np.zeros((self.K_b, self.K_r, 15, 15, 15, 15))
        l_gmm = np.zeros((self.K_b, self.K_r, 15, 15))
        
        for k_b in range(self.K_b):
            for k_r in range(self.K_r):
                
                
                hmm_local_b = base.models[k_b] 
                hmm_local_r = reduce.models[k_r]
                  
                S_r, S_b = hmm_local_r.number_states, hmm_local_b.number_states
                
                for s_b in range (S_b):
                    for s_r in range (S_r):
                        
                        gmm_local_b = hmm_local_b.emissions[s_b]
                        L_b = gmm_local_b.number_emissions
                        
                        gmm_local_r = hmm_local_r.emissions[s_r]
                        L_r = gmm_local_r.number_emissions
                        
                        for l_b in range(L_b):
                            local_log_etas = list()
                            denom = list() 
                            
                            sigma_b = hmm_local_b.emissions[s_b].covars[l_b]
                            mu_b = hmm_local_b.emissions[s_b].means[l_b]
                            
                             
                            for l_r in range(L_r):
                                
                                sigma_r = hmm_local_r.emissions[s_r].covars[l_r]
                                c_ = hmm_local_r.emissions[s_r].weights_gmm[l_r]
                                
                             
                                det_sigma = np.linalg.det(sigma_r)
                                inv_sigma = np.linalg.inv(sigma_r)
                                
                                mu_r = hmm_local_r.emissions[s_r].means[l_r]
                                
                                partial_lb_ = -0.5 * (np.log(2 * np.pi)
                                                     + np.log(det_sigma)
                                                     + np.trace(inv_sigma @ sigma_b) 
                                                     +  (mu_r - mu_b) @ inv_sigma @ (mu_r - mu_b).T)
                                partial_lb[k_b, k_r, s_b, s_r, l_b, l_r] = partial_lb_
                                
                                num_log_eta = np.log(c_) + partial_lb_
                                local_log_etas.append(num_log_eta)
                                
                                denom.append(num_log_eta)
                            
                            
                            denom = logsumexp(denom)
                            local_log_etas = np.array(local_log_etas) - denom
                        
                            log_eta[k_b, k_r, s_b, s_r, l_b, : L_r] = local_log_etas
                            
                            
              
        for k_b in range(self.K_b):
            for k_r in range(self.K_r):
                
                
                hmm_local_b = base.models[k_b] 
                hmm_local_r = reduce.models[k_r]
                  
                S_r, S_b = hmm_local_r.number_states, hmm_local_b.number_states
                
                for s_b in range (S_b):
                    local_gmm = list()
                    for s_r in range (S_r):
                        
                        gmm_local_b = hmm_local_b.emissions[s_b]
                        L_b = gmm_local_b.number_emissions
                        
                        gmm_local_r = hmm_local_r.emissions[s_r]
                        L_r = gmm_local_r.number_emissions
                         
                        local_gmm_m = list()
                        
                        for l_b in range(L_b):
                            c_b = hmm_local_b.emissions[s_b].weights_gmm[l_b] 
                         
                            local_gmm_l = list()
                            for l_r in range(L_r):
                                c_r = hmm_local_r.emissions[s_r].weights_gmm[l_r]
                                
                                tmp = (log_eta[k_b, k_r, s_b, s_r, l_b, l_r]
                                       + logsumexp([np.log(c_r), 
                                                    -log_eta[k_b, k_r, s_b, s_r, l_b, l_r],
                                                    + partial_lb[k_b, k_r, s_b, s_r, l_b, l_r] ]))
                                local_gmm_l.append(tmp)
                
                            local_gmm_m.append(np.log(c_b) + logsumexp(local_gmm_l))    
            
                        l_gmm[k_b, k_r, s_b, s_r] = logsumexp(local_gmm_m)
                  
    
        # Equation 37
        log_phi = np.zeros((self.tau, self.K_b, self.K_r, 15))
        l_hmm = np.zeros((self.tau+1, self.K_b, self.K_r))
        for t in reversed(range(1, self.tau)):
            print(t)
            for k_b in range(self.K_b):
                 
                hmm_local_b = base.models[k_b] 
                a_b = hmm_local_b.trans_mat
                
                for k_r in range(self.K_r):
                    
                    hmm_local_r = reduce.models[k_r] 
                    a_r = hmm_local_r.trans_mat
                    
                    S_r, S_b = hmm_local_r.number_states, hmm_local_b.number_states
                 
                    tmp_b = [] 
                    for s_b in range (S_b): 
                        tmp_r = []
                        
                        for s_r in range (S_r):
                            # For l_hmm
                            tmp_r.append(a_r[s_r, s_r] 
                                         * np.exp(np.exp(l_gmm[k_b, k_r, s_b, s_r])
                                                  + l_hmm[t, k_b, k_r]))
                             
                        tmp_b.append(a_b[s_b, s_b] * logsumexp(tmp_r))
                    
                    tmp = np.sum(tmp_b)
                    l_hmm[t, k_b, k_r] = tmp
                    
            # Suite Equation 37 pour phi
            for k_b in range(self.K_b):
               
                for k_r in range(self.K_r):
                    
                    hmm_local_r = reduce.models[k_r] 
                    a_r = hmm_local_r.trans_mat
                    
                    S_r = hmm_local_r.number_states 
                  
                    tmp_phi_r = []
                    for s_r in range (S_r):   
                        
                        # For log_phi
                        tmp_phi_r.append(a_r[s_r, s_r] 
                                         * np.exp(np.exp(l_gmm[k_b, k_r, s_b, s_r])
                                                  + l_hmm[t, k_b, k_r]))
                    denom = logsumexp(tmp_phi_r) 
                    
                    local_log_phi = np.log(tmp_phi_r) - denom
                    
                    log_phi[t, k_b, k_r, : S_r] = local_log_phi
                        
                        
                        
  
                
                
                 
    
 
class GMM:
    def __init__(self, M, d=2):
         
        self.means = [(np.random.normal(0, 10, size = d)) for _ in range(M)] 
        self.covars = [np.diag(np.ones((d))) for _ in range(M)]
        self.number_emissions = M
        self.weights_gmm = np.ones(M) / M
 

class HMM:
    def __init__(self, S):
        self.trans_mat = np.random.dirichlet([1] * S, size=S)
        self.init_prob = np.random.dirichlet([1] * S)
        self.number_states = S
        self.emissions = [GMM(np.random.randint(1, 6), d=2) for _ in range(S)]
        

# Initialization
class H3M:
    def __init__(self, models, weights=None):
        
        self.K = len(models)
        
        self.models = dict({})
        for i in range((len(models))):
            self.models.update({i: models[i]})
            
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)



base_models = [HMM(np.random.randint(1, 11)) for _ in range(100)]
base = H3M(base_models)
reduce = H3M(np.random.choice(base_models, 2))


(base.models[0].emissions[0].means)

VHEM(base, reduce)




















    
    
    