from scipy.stats import multivariate_normal 
import numpy as np
 
#np.random.seed(1)


def evaluate_posterior_moments(double[:,:] a_s, 
                               _means, _vars, pi, T):
    
    cdef int n = 0
    cdef int k = 0
    cdef int l = 0
    
    cdef int n_ = a_s.shape[0]
    cdef int d_ = a_s.shape[1] 
    
    cdef int n_s = T.shape[0]
    cdef double[:,:] c_T = T
    
    
    emm_prob = np.zeros((n_, n_s)) 
    for k in range(0, n_s): 
        emm_prob[:, k] = multivariate_normal.pdf(a_s, _means[k], _vars[k])
        
    cdef double[:,:] c_emm_prob = emm_prob
        
    cdef double [:,:] alpha = np.zeros((n_, n_s))
    cdef double [:,:] beta = np.zeros((n_, n_s))
    cdef double[:] norm_ = np.zeros(n_)
    
    cdef double[:] tmp = np.zeros(n_s)
    cdef double[:,:] tmp_ = np.zeros((n_s, n_s)) 
        
    # Forward-backward algorithm. 
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
    xi = np.zeros((n_, n_s, n_s))

    for n in range(0, n_-1): 
        for k in range(0, n_s): 
            for j in range(0, n_s): 
                tmp_[k, j] = (gamma[n, k]
                             * beta[n+1, j]
                             * emm_prob[n+1, j]
                             * T[k, j]) / beta[n, k] 
        xi[n] = tmp_
        
    moments = (gamma, xi) 
    return moments
         
        
def baum_welch(double[:,:] a_s, int n_s, int n_iters,
               _means):
    
    cdef int n = 0
    cdef int k = 0
    cdef int l = 0
    
    cdef int n_ = a_s.shape[0]
    cdef int d_ = a_s.shape[1] 
 
    # Initialize parameter set
    pi = np.ones(n_s) / float(n_s)
    
    T = np.ones((n_s, n_s)) * (1/float(n_s)) 
    cdef double[:,:] c_T = T
  
    _vars = np.zeros((n_s, d_, d_)) 
    
    for i in range(n_s):
        _vars[i] = np.diag(np.ones(2)) * 2500
 
    # Initialize emission probabilities (size n_ × n_s).
    emm_prob = np.zeros((n_, n_s))
 
    for k in range(0, n_s): 
        emm_prob[:, k] = multivariate_normal.pdf(a_s, _means[k], _vars[k])
        
    cdef double[:,:] c_emm_prob = emm_prob
        
    cdef double [:,:] alpha = np.zeros((n_, n_s))
    cdef double [:,:] beta = np.zeros((n_, n_s))
    cdef double[:] norm_ = np.zeros(n_)
    
    cdef double[:] tmp = np.zeros(n_s)
    cdef double[:,:] tmp_ = np.zeros((n_s, n_s)) 
    #cdef int a 
    
    iter_ = 0
    
    
    while iter_ < n_iters:
        #print(iter_)
        ## Forward-backward algorithm. 
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
   
        ## Compute first posterior moment gamma
        gamma = np.asarray(alpha) * np.asarray(beta) 
        gamma /= np.sum(gamma, axis = 1).reshape((n_, 1))
        
        ## Compute second posterior moment xi.
        xi = np.zeros((n_, n_s, n_s))
    
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
                if np.sum(gamma[-1:, i]) > 1e-15:
                    T[i, j] = (np.sum(xi[:-1, i, j])
                               / np.sum(gamma[:-1,i]))
                else:
                    T[i,j] = 1e-15
                 
        T /= np.sum(T, axis =1).reshape((n_s,1))
        c_T = T
       
        ## Compute new means.
        for k in range(0, n_s):
       
            _means[k] = np.asarray(a_s).T @ gamma[:, k] 
          
            if np.sum(gamma[:, k]) > 1e-15:
                _means[k] /= np.sum(gamma[:, k]) 
        
        ## Compute new covariances. 
        for k in range(n_s): 
            _vars[k] = np.zeros((d_, d_))
            for n in range(n_):
            
                dev = np.asarray(a_s[n]) - _means[k]
                _vars[k] += np.asarray(gamma[n, k]) * np.outer(dev, dev.T)
               
            
            if np.sum(gamma[:, k]) > 1e-15:
                _vars[k] /= np.sum(gamma[:, k])
         
     
        ## Recompute emission probabilities using inferred parameters.  
        for k in range(0, n_s):    
            if np.all(_vars[k]<1e-15):
                _vars[k] += np.diag(np.ones(2))
       
            emm_prob[:, k] = multivariate_normal.pdf(a_s, _means[k], 
                                                     _vars[k] ,
                                                     allow_singular=True)  
        c_emm_prob = emm_prob 
        iter_ += 1 
        
        
        
    Z = np.argmax(gamma, axis=1)
    theta = (pi, T, _means, _vars)
    moments = (gamma, xi)
    
    return Z, theta, moments

 
