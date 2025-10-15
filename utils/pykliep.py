import numpy as np
import warnings

def _projected_gradient(A, b, alpha0, eps, max_iter):
    """
    Simplified version without numba to avoid matrix dimension issues
    """
    # Ensure all inputs are float32 and contiguous
    A = np.ascontiguousarray(A.astype(np.float32))
    b = np.ascontiguousarray(b.astype(np.float32))
    out = np.ascontiguousarray(alpha0.astype(np.float32))
    
    eps = np.float32(eps)
    
    # Check dimensions
    # print(f"Debug: A.shape={A.shape}, b.shape={b.shape}, out.shape={out.shape}")
    
    # Ensure b is 2D column vector
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    
    # Ensure out is 2D column vector  
    if out.ndim == 1:
        out = out.reshape(-1, 1)
        
    At = A.T.copy()
    bt_b = float((b.T @ b).item())  # Convert to scalar
    
    for iteration in range(max_iter):
        # A @ out should give shape (n_test, 1)
        Aout = A @ out
        
        # Avoid division by zero
        Aout = np.maximum(Aout, 1e-8)
        z = 1.0 / Aout
        
        # Update rule
        out += eps * (At @ z)
        
        # Constraint: b.T @ out = 1
        bt_out = float((b.T @ out).item())
        out += b * ((1.0 - bt_out) / bt_b)
        
        # Project to non-negative simplex
        out = np.maximum(out, 0.0)
        
        # Normalize
        bt_out = float((b.T @ out).item())
        if bt_out > 1e-8:
            out /= bt_out
            
    return out

class DensityRatioEstimator:
    """
    Class to accomplish direct density estimation implementing the original KLIEP 
    algorithm from Direct Importance Estimation with Model Selection
    and Its Application to Covariate Shift Adaptation by Sugiyama et al. 
    
    The training set is distributed via 
                                            train ~ p(x)
    and the test set is distributed via 
                                            test ~ q(x).
                                            
    The KLIEP algorithm and its variants approximate w(x) = q(x) / p(x) directly. The predict function returns the
    estimate of w(x). The function w(x) can serve as sample weights for the training set during
    training to modify the expectation function that the model's loss function is optimized via,
    i.e.
    
            E_{x ~ w(x)p(x)} loss(x) = E_{x ~ q(x)} loss(x).
    
    Usage : 
        The fit method is used to run the KLIEP algorithm using LCV and returns value of J 
        trained on the entire training/test set with the best sigma found. 

        Use the predict method on the training set to determine the sample weights from the KLIEP algorithm.
    """
    
    def __init__(self, max_iter=5000, num_params=[.1,.2], epsilon=1e-4, cv=3, sigmas=[.01,.1,.25,.5,.75,1], random_state=None, verbose=0):
        """ 
        Direct density estimation using an inner LCV loop to estimate the proper model. Can be used with sklearn
        cross validation methods with or without storing the inner CV. To use a standard grid search.
        
        
        max_iter : Number of iterations to perform
        num_params : List of number of test set vectors used to construct the approximation for inner LCV.
                     Must be a float. Original paper used 10%, i.e. =.1
        sigmas : List of sigmas to be used in inner LCV loop.
        epsilon : Additive factor in the iterative algorithm for numerical stability.
        """
        self.max_iter = max_iter
        self.num_params = num_params
        self.epsilon = epsilon
        self.verbose = verbose
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = 0
        
    def fit(self, X_train, X_test, alpha_0=None):
        """ Uses cross validation to select sigma as in the original paper (LCV).
            In a break from sklearn convention, y=X_test.
            The parameter cv corresponds to R in the original paper.
            Once found, the best sigma is used to train on the full set."""
        
        # LCV loop, shuffle a copy in place for performance.
        cv = self.cv
        chunk = int(X_test.shape[0]/float(cv))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled)
        
        j_scores = {}
        
        if type(self.sigmas) != list:
            self.sigmas = [self.sigmas]
        
        if type(self.num_params) != list:
            self.num_params = [self.num_params]
        
        if len(self.sigmas) * len(self.num_params) > 1:
            # Inner LCV loop
            for num_param in self.num_params:
                for sigma in self.sigmas:
                    j_scores[(num_param,sigma)] = np.zeros(cv)
                    for k in range(1,cv+1):
                        if self.verbose > 0:
                            print('Training: sigma: %s    R: %s' % (sigma, k))
                        X_test_fold = X_test_shuffled[(k-1)*chunk:k*chunk,:] 
                        j_scores[(num_param,sigma)][k-1] = self._fit(X_train=X_train, 
                                                         X_test=X_test_fold,
                                                         num_parameters = num_param,
                                                         sigma=sigma)
                    j_scores[(num_param,sigma)] = np.mean(j_scores[(num_param,sigma)])

            sorted_scores = sorted([x for x in j_scores.items() if np.isfinite(x[1])], key=lambda x :x[1], reverse=True)
            if len(sorted_scores) == 0:
                warnings.warn('LCV failed to converge for all values of sigma.')
                return self
            self._sigma = sorted_scores[0][0][1]
            self._num_parameters = sorted_scores[0][0][0]
            self._j_scores = sorted_scores
        else:
            self._sigma = self.sigmas[0]
            self._num_parameters = self.num_params[0]
            # best sigma
        self._j = self._fit(X_train=X_train, X_test=X_test_shuffled, num_parameters=self._num_parameters, sigma=self._sigma)

        return self # Compatibility with sklearn
        
    def _fit(self, X_train, X_test, num_parameters, sigma, alpha_0=None):
        """ Fits the estimator with the given parameters w-hat and returns J"""
        
        num_parameters = num_parameters
        
        if type(num_parameters) == float:
            num_parameters = int(X_test.shape[0] * num_parameters)

        self._select_param_vectors(X_test=X_test, 
                                   sigma=sigma,
                                   num_parameters=num_parameters)
        
        X_train = self._reshape_X(X_train)
        X_test = self._reshape_X(X_test)
        
        if alpha_0 is None:
            alpha_0 = np.ones(shape=(num_parameters,1))/float(num_parameters)
        
        self._find_alpha(X_train=X_train,
                         X_test=X_test,
                         num_parameters=num_parameters,
                         epsilon=self.epsilon,
                         alpha_0 = alpha_0,
                         sigma=sigma)
        
        return self._calculate_j(X_test,sigma=sigma)
    
    def _calculate_j(self, X_test, sigma):
        return np.log(self.predict(X_test,sigma=sigma)).sum()/X_test.shape[0]
    
    def score(self, X_test):
        """ Return the J score, similar to sklearn's API """
        return self._calculate_j(X_test=X_test, sigma=self._sigma)

    @staticmethod   
    def _reshape_X(X):
        """ Reshape input from mxn to mx1xn to take advantage of numpy broadcasting. """
        if len(X.shape) != 3:
            return X.reshape((X.shape[0],1,X.shape[1]))
        return X
    
    def _select_param_vectors(self, X_test, sigma, num_parameters):
        """ X_test is the test set. b is the number of parameters. """ 
        indices = np.random.choice(X_test.shape[0], size=num_parameters, replace=False)
        self._test_vectors = X_test[indices,:].copy()
        self._phi_fitted = True

    def _phi(self, X, sigma=None):
        if sigma is None:
            sigma = self._sigma
        X  = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)   # (N,d)
        V  = self._test_vectors.reshape(self._test_vectors.shape[0], -1).astype(np.float32, copy=False)  # (b,d)

        x2 = (X**2).sum(1)[:, None]          # (N,1)
        v2 = (V**2).sum(1)[None, :]          # (1,b)
        K  = np.exp(-(x2 + v2 - 2*X @ V.T) / (2.0 * sigma**2))

        return K

    def _find_alpha(self, alpha_0, X_train, X_test, num_parameters, sigma, epsilon):
        A = self._phi(X_test, sigma).astype(np.float32, copy=False)
        b = self._phi(X_train, sigma).mean(0, keepdims=True).T.astype(np.float32, copy=False)

        if alpha_0.dtype != np.float32:
            alpha_0 = alpha_0.astype(np.float32)

        # epsilon llega como float64; pásalo como float32
        eps32 = np.float32(epsilon)

        out = _projected_gradient(A, b, alpha_0, eps32, self.max_iter)
        self._alpha = out
        self._fitted = True
        
    def predict(self, X, sigma=None):
        """ Equivalent of w(X) from the original paper."""
        
        X = self._reshape_X(X)
        if not self._fitted:
            raise Exception('Not fitted!')
        return np.dot(self._phi(X, sigma=sigma), self._alpha).reshape((X.shape[0],))