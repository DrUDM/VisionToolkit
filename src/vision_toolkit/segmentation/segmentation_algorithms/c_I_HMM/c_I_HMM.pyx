from scipy.stats import norm

from scipy.special import logsumexp
import numpy as np
 
from vision_toolkit.utils.segmentation_utils import interval_merging
from vision_toolkit.utils.segmentation_utils import centroids_from_ints

 

def process_IHMM(dict data_set, dict config):
    """
    I-HMM algorithm.
    """
    
    cdef double[:] a_s = data_set['absolute_speed']# / vf_diag
    
    cdef double i_low_vel = config['HMM_init_low_velocity']
    cdef double i_high_vel = config['HMM_init_high_velocity']
    cdef double i_var = config['HMM_init_variance']
    
    cdef int n_iter = config['HMM_nb_iters']
    cdef int s_f = config['sampling_frequency']
    
    theta = baum_welch(a_s, 2, n_iter,
                       i_low_vel, i_high_vel,
                       i_var)
  
    s_s = Viterbi(theta[1], theta[3], theta[0])
 
    fix_s = np.argmin(theta[2]) 
 
    wi_fix = np.where(s_s[:-1] == fix_s)[0]  
    
    # Add index + 1 to fixation since velocities are computed from two data points
    wi_fix = np.array(sorted(set(list(wi_fix) + list(wi_fix + 1)))) 
    
    i_fix = np.array([False]*config['nb_samples'])  
    i_fix[wi_fix] = True
  
    x_a = data_set['x_array']
    y_a = data_set['y_array']
 
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size = np.ceil(config['min_sac_duration']*s_f), 
                              )
    
    if config['verbose']:
        print('   Saccadic intervals identified with minimum duration: {s_du} sec'.format(s_du=config['min_sac_duration']))
    
    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True]*config['nb_samples'])
    
    for s_int in s_ints:
        i_fix[s_int[0]: s_int[1]+1] = False
          
    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config['min_fix_duration']*s_f))
    
    for i in range(1, len(s_ints)):
        
        s_int = s_ints[i]
        o_s_int = s_ints[i-1]
        
        if s_int[0] - o_s_int[-1] < fix_dur_t:
            i_fix[o_s_int[-1]: s_int[0]+1] = False
    
    if config['verbose']:
        print('   Close saccadic intervals merged with duration threshold: {f_du} sec'.format(f_du=config['min_fix_duration']))
     
    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]
    
    f_ints = interval_merging(
        wi_fix,   
        min_int_size = np.ceil(config['min_fix_duration']*s_f), 
        max_int_size = np.ceil(config['max_fix_duration']*s_f),
        status = data_set['status'],
        proportion = config['status_threshold']
        )
    
    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints,
                                x_a, y_a)
    
    # Recompute saccadic intervals
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size = np.ceil(config['min_sac_duration']*s_f),
        status = data_set['status'],
        proportion = config['status_threshold']
        )
    
    if config['verbose']:
        print('   Fixations ans saccades identified using availability status threshold: {s_th}'.format(s_th=config['status_threshold']))
    
    assert len(f_ints) == len(ctrds), "Interval set and centroid set have different lengths"
    
    # Keep track of index that were effectively labeled
    i_lab = np.array([False]*config['nb_samples'])  
    
    for f_int in f_ints:
        i_lab[f_int[0]: f_int[1]+1] = True
        
    for s_int in s_ints:
        i_lab[s_int[0]: s_int[1]+1] = True
         
    return dict({
        'is_labeled': i_lab,
        'fixation_intervals': f_ints, 
        'saccade_intervals': s_ints,
        'centroids': ctrds, 
            })


def baum_welch(double[:] a_s, double d_n_s, int n_iters,
               double i_low_vel, double i_high_vel,
               double i_var):
    
    cdef int n = 0
    cdef int k = 0
    cdef int l = 0
    
    cdef int n_ = len(a_s)
    cdef int n_s = int(d_n_s)
 
    # Initialize parameter set
    pi = np.ones(n_s) / d_n_s
    
    T = np.ones((n_s, n_s)) * (1/d_n_s) 
    cdef double[:,:] c_T = T
     
    
    _means = np.array([i_low_vel, i_high_vel])
    _vars = np.ones(n_s) * i_var
 
    # Initialize emission probabilities (size n_ × n_s).
    emm_prob = np.zeros((n_, n_s))
 
    for k in range(0, n_s): 
        emm_prob[:, k] = norm.pdf(a_s, _means[k], np.sqrt(_vars[k])) 
        
    cdef double[:,:] c_emm_prob = emm_prob
        
    cdef double [:,:] alpha = np.zeros((n_, n_s))
    cdef double [:,:] beta = np.zeros((n_, n_s))
    cdef double[:] norm_ = np.zeros(n_)
    
    cdef double[:] tmp = np.zeros(n_s)
    cdef double[:,:] tmp_ = np.zeros((n_s, n_s))
   
    cdef int a 
    
    iter_ = 0
    while iter_ < n_iters:
 
        #print("--- Iteration number: %s ---" % (iter_+1))
        # E-step forward-backward algorithm. 
     
        
        for k in range(0, n_s):
    
            alpha[0, k] = pi[k] * emm_prob[0, k]
    
        norm_[0] = np.sum(alpha[0])
        
        for l in range(0, n_s):
            alpha[0,l] = alpha[0,l] / norm_[0]
        
        with nogil:
            
            for n in range(1, n_):
        
                for k in range(0, n_s):
                    
                    alpha[n, k] = 0.0
                    
                    for l in range(0, n_s): 
                        alpha[n, k] = alpha[n, k] + (alpha[n-1,l] * c_T[l, k] * c_emm_prob[n, k])
              
                norm_[n] = 0.0
                
                for l in range(0, n_s):
                    norm_[n] = norm_[n] + alpha[n,l]  
                
                for l in range(0, n_s):
                    alpha[n,l] = alpha[n,l] / norm_[n]
          
        for l in range(0, n_s):
            beta[n_-1,l] = (1/norm_[n-1])
       
        with nogil:
            
            for n in reversed(range(0, n_-1)):
        
                for k in range(0, n_s):
                    
                    beta[n, k] = 0.0
                    
                    for l in range(0, n_s):
                        beta[n, k] = beta[n, k] + beta[n+1,l] * c_emm_prob[n+1,l] * c_T[k,l]

                for l in range(0, n_s):
                    beta[n,l] = beta[n,l] / norm_[n]
            
        # Compute first posterior moment gamma
        gamma = np.asarray(alpha) * np.asarray(beta)
        gamma /= np.sum(gamma, axis = 1).reshape((n_, 1))
     
    
        # Compute second posterior moment xi.
        xi = np.empty((n_, n_s, n_s))
    
        for n in range(0, n_-1):
 
            for k in range(0, n_s):
    
                for j in range(0, n_s):
    
                    tmp_[k, j] = (gamma[n, k]
                                 * beta[n+1, j]
                                 * emm_prob[n+1, j]
                                 * T[k, j]) / beta[n, k]
    
            xi[n] = tmp_
 
        pi = gamma[0] / np.sum(gamma[0])
    
        assert pi.size == n_s 
    
        for i in range(0, n_s):
    
            for j in range(0, n_s):
    
                T[i, j] = (np.sum(xi[:-1, i, j])
                           / np.sum(gamma[:-1,i]))
                
        T /= np.sum(T, axis =1).reshape((n_s,1))
        c_T = T
        
        # Compute new means.
        for k in range(0, n_s):
    
            _means[k] = np.asarray(a_s).reshape(1, n_) @ gamma[:, k]  
            _means[k] /= np.sum(gamma[:, k]) 
    
        # Compute new covariances.
        for k in range(0, n_s):
     
            _vars[k] = np.sum(gamma[:, k] * (a_s - _means[k])**2)  
            _vars[k] /= np.sum(gamma[:, k])        
  
        # Recompute emission probabilities using inferred parameters. 
        for k in range(0, n_s): 
            emm_prob[:, k] = norm.pdf(a_s, _means[k], np.sqrt(_vars[k])) 
    
        c_emm_prob = emm_prob
        
        iter_ += 1 
 
    log_emm_prob = np.zeros((n_, n_s)) 
    
    for k in range(0, n_s): 
        log_emm_prob[:, k] = norm.logpdf(a_s, _means[k], np.sqrt(_vars[k])) 
    
    theta = (np.log(pi + 0.000000001), np.log(T + 0.000000001), _means, log_emm_prob)

    return theta


def Viterbi(log_T, log_emm_prob, log_pi):
    """
    Log implementation of the Viterbi algorithm.  
    Inputs:
        - log_T = log state transition matrix (n_s x n_s)
        - log_emm_prob = log emission probability matrix (n_ X n_s)
        - log_pi = log initial state distribution (1 X n_s) 
    Outputs:
        - D = Accumulated probability matrix (n_s x n_). D[i, n] is the
            highest probability along a single state sequence (s_1,…,s_n)
            that accounts for the first n observations and ends in state
            s_n=alpha_i
        - S = Optimal state sequence (1 x n_)
        - backtracking = Backtracking matrix
    """

    # Parameters &
    n_s = log_T.shape[0]  # Number of states
    n_ = len(log_emm_prob)  # Length of observation sequence

    # Variables
    D = np.zeros((n_s, n_))
    backtracking = np.zeros((n_s, n_-1)).astype(np.int32)
    S = np.zeros(n_).astype(np.int32)

    # Initialization
    D[:, 0] = log_pi + log_emm_prob[0, :]

    # Computation
    for n in range(1, n_):  # Loop over observations

        for i in range(n_s):  # Loop over states

            sum = D[:, n-1] + log_T[:, i]
            D[i, n] = np.max(sum) + log_emm_prob[n, i]
            backtracking[i, n-1] = np.argmax(sum)

    # Backtracking
    S[-1] = np.argmax(D[:, -1])

    for n in range(n_-2, -1, -1):

        S[n] = backtracking[int(S[n+1]), n]

    return S 
