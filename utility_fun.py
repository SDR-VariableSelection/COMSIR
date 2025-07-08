import numpy as np
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
import scipy.linalg as la
from statsmodels.stats.dist_dependence_measures import distance_correlation
from numba import jit
from functools import partial
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from multiprocessing import cpu_count

def sir(X, y, H=5):
    n, p = X.shape
    M = np.zeros((p, p))
    Exy = np.zeros((n, p))
    X0 = X - np.mean(X, axis = 0)
    YI = np.argsort(y.reshape(-1))
    A = np.array_split(YI, H) # split into 5 parts
    for i in range(H):
        tt = X0[A[i],].reshape(-1,p)
        Xh = np.mean(tt, axis = 0)#
        ni = len(A[i])
        Exy[A[i],] = np.tile(Xh, (ni, 1))
        # assert Exy[A[i],].shape == (ni, p)
        # a1 = np.ones(ni).reshape(ni, 1) @ Xh.reshape(1, p)
        # a2 = np.tile(Xh, (ni, 1))
        # assert np.allclose(a1, a2)
        Xh = Xh.reshape(1,p)
        # assert (Xh.T @ Xh).shape == (p,p)
        M = M + Xh.T @ Xh * ni/n
    return M, Exy

# %%
@jit(nopython=True)
def cap_soft_th(d, r, tol=1e-2):
    """
    Performs capping and soft thresholding using binary search.
    
    Args:
        d: input array of eigenvalues
        r: threshold parameter (typically the desired rank)
        tol: tolerance for convergence in binary search
    
    Returns:
        Array of thresholded values between 0 and 1
    """
    amin = 0
    amax = np.max(d)
    
    while True:
        amid = (amin + amax) / 2
        t = np.minimum(1, np.maximum(0, d - amid))

        if np.sum(t) <= r:
            amax = amid
        else:
            amin = amid

        if amax - amin < tol**2:
            break

    return t


# %%
def fantope(M1, d, lam_ratio=0.1, max_iter = 100):
    """
    Implements the Fantope projection algorithm for dimensionality reduction and matrix optimization.
    This algorithm finds a low-rank approximation of the input matrix while maintaining specific constraints.
    
    The algorithm uses an iterative approach with ADMM (Alternating Direction Method of Multipliers)
    to solve the optimization problem.
    
    Args:
        M1 (np.ndarray): Input matrix of shape (p, p) to be projected
        d (int): Target rank/dimension for the projection
        lam_ratio (float): Regularization parameter ratio relative to maximum eigenvalue (default: 0.1)
                          Controls the sparsity of the solution
    
    Returns:
        np.ndarray: The eigenvectors corresponding to the top d eigenvalues of the projected matrix,
                   ordered in descending order of importance
    
    Implementation Details:
    1. Initialization:
        - Compute maximum eigenvalue and lambda parameter
        - Initialize working matrices A, U (dual variables), and B (auxiliary variable)
    
    2. ADMM Iteration:
        - Update B: Project onto Fantope using eigendecomposition
        - Update A: Apply soft thresholding
        - Update U: Update dual variables
        - Check convergence using primal and dual residuals
    
    3. Convergence Criteria:
        - Algorithm stops when both primal and dual residuals are small enough
        - Maximum iterations set to 100 to ensure termination
    """
    # Get dimension of input matrix
    p = M1.shape[1]
    
    # Calculate regularization parameter
    max_lam = np.max(eigh(M1)[0])  # Maximum eigenvalue
    lam = max_lam * lam_ratio  # Scaled regularization parameter
    
    # ADMM parameters
    rho = 1.0  # Augmented Lagrangian parameter
    tol = d * 1e-6  # Convergence tolerance
    
    # Initialize optimization variables
    A = np.zeros((p, p))  # Primary variable
    U = np.zeros((p, p))  # Scaled dual variable
    B = np.zeros((p, p))  # Auxiliary variable
    
    # Pre-allocate arrays for eigendecomposition
    # This avoids repeated memory allocation in the loop
    eival = np.zeros(p)
    eivec = np.zeros((p, p))
    
    # Main ADMM iteration loop
    for i in range(max_iter):
        # Store previous A for convergence check
        oldA = A.copy()
        
        # Step 1: Update B (Fantope projection)
        temp1 = A - U + M1/rho  # Combined matrix for eigendecomposition
        eival, eivec = eigh(temp1)  # Eigendecomposition
        eival_threshold = cap_soft_th(eival, d)  # Apply thresholding
        assert len(eival_threshold)==p
        B = (eivec * eival_threshold) @ eivec.T  # Reconstruct matrix
        
        # Step 2: Update A (Soft thresholding)
        temp2 = B + U
        A = np.maximum(np.abs(temp2) - lam/rho, 0) * np.sign(temp2)
        
        # Step 3: Update dual variables
        U += B - A
        
        # Check convergence
        # Uses both primal and dual residuals
        primal_residual = norm(B - A)**2
        dual_residual = rho**2 * norm(A - oldA)**2
        
        if max(primal_residual, dual_residual) < tol:
            # print(i)
            break
    
    # Extract final solution
    # Compute eigendecomposition of final matrix
    values, vectors = eigh((A+A.T)/2)
    # Return top d eigenvectors in descending order
    return vectors[:, -d:][:, ::-1], values[::-1]

# %%
def smddm(x, y, lam=None):
    """
    Input:
        Y: n X q
        p: lags
        d: reduced rank
    Output:
        beta: qp X d
    """
    # dimensions of y
    n, p = x.shape
    # covariate
    sigma = np.cov(x.T)

    # MDDM
    Gam = sir(x, y)[0]
    lam_max = max(norm(Gam, axis = 1))
    if lam is None:
        lam = lam_max*0.3
    max_iter = 100
    t0 = norm(sigma, 2) ** (-2)
    t = 1
    M = np.zeros((p, p))
    B = np.zeros((p, p))

    obj = np.zeros(max_iter)
    for i in range(max_iter):
        # soft-threshold update M
        B_old = B
        M_old = M
        temp = M - t0 * (sigma @ B @ sigma - Gam)
        w = np.maximum(1 - lam * t0 / norm(temp, axis=1), 0)
        w = w.reshape(-1)
        M = (w * temp.T).T
        # symmetrization
        M[:,w==0] = 0
        M = (M+M.T)/2

        # update step
        t_new = (1 + np.sqrt(1 + 4 * t ** 2))/2

        # update B
        B = M + (t -1)/t_new * (M - M_old)

        # objective
        obj[i] = 1/2 * np.trace(M.T @ sigma @ M @ sigma) - np.trace(M.T @ Gam) + lam * np.sum(norm(M, axis=1))
        t = t_new

        if max(norm(B_old - B, 'fro'), norm(M_old - M, 'fro')) < 1e-8:
            break
    # check sparsity
    # print("w:", w)

    # Compute the eigenvalues and eigenvectors
    # values, vectors = eigh(M)
    # print("Eigenvectors:\n", eigenvectors)

    # # final estimate
    # beta2 = np.fliplr(eigenvectors)
    # beta2 = beta2[:,:d]
    # ind_nonzero = np.where(w>10e-5)[0]
    return M


# %%
def select_d(x, y): 
    M = smddm(x, y)
    n,p = x.shape
    factor = n ** (1/3) * p ** (1/3)
    values = eigh(M)[0]
    val = values[::-1]
    logdiff = np.log(val+1) - val 
    kk = np.array(range(1, p+1))
    d_seq = np.cumsum(logdiff)* n / (2 * sum(logdiff)) - factor * kk * (kk+1) / p

    max_index = np.argmax(d_seq) + 1
    return  max_index

# %%
## algorithm
# @profile
def comsir(X, Y, d, lam_ratio=0.1, max_iter = 100):
    """
    Sample size can be different, but dimension is the same. 
    d (int): Target rank/dimension for the projection

    """
    if isinstance(X, list) and isinstance(Y, list):
        # Case 1: Both X and Y are lists
        assert len(X) == len(Y), "Lengths of X and Y must match."
        T = len(X)

    elif isinstance(X, list) and isinstance(Y, np.ndarray):
        # Case 2: X is a list and Y is a NumPy array
        assert len(X) == Y.shape[1], "Length of X must match the number of columns in Y."
        T = len(X)
        Y = [Y[:, t] for t in range(T)]

    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        # Case 3: Both X and Y are NumPy arrays
        n = X.shape[0]
        if Y.shape == (n,):
            T = 1
            Y = Y.reshape((n,T))
        else:
            T = Y.shape[1]
            
        X = [X for _ in range(T)]
        Y = [Y[:, t] for t in range(T)]

    else:
        raise TypeError("X and Y must be either lists or NumPy arrays.")

    p =  X[0].shape[1]# Define the value of p here
    N = [x.shape[0] for x in X]
    M = np.zeros((p, p, T))
    V = np.zeros((p, d, T))
    XZ = np.zeros((p,d,T))
    XX = np.zeros((p,p,T))
    

    Exy = [np.zeros((i,p)) for i in N]
    Z = [np.zeros((i,d)) for i in N]  # Correct initialization of Z
    ## Step 1: 
    for i in range(T):
        M[:, :, i], Exy[i] = sir(X[i], Y[i])
        V[:, :, i] = fantope(M[:, :, i], d, lam_ratio=0.01, max_iter=max_iter)[0]
        Z[i] = Exy[i] @ V[:, :, i]
        XZ[:, :, i] = - X[i].T @ Z[i]
        XX[:, :, i] = X[i].T @ X[i]    

    ## Step 2: 
    lam_max = max(norm(XZ, axis=(1,2)))
    # print(lam_max)
    # if lam is None:
    lam = lam_max * lam_ratio
    ## initialization
    # max_iter = 100
    t0 = 1 / sum([norm(x.T @ x, 2) for x in X])
    t = 1
    W = np.zeros((p, d, T))
    B = np.zeros((p, d, T))
    obj = []
    temp = np.zeros((p, d, T))
    for i in range(max_iter): 
        # soft-threshold update W
        W_old = W.copy()
        for j in range(T):
            temp[:,:,j] = W[:, :, j] - t0 * (XZ[:, :, j] + XX[:, :, j] @ W[:, :, j])
        w = np.maximum(1 - lam * t0 / norm(temp, axis=(1,2)), 0)
        assert(w.shape == (p,))

        W = temp * w[:, np.newaxis, np.newaxis]
        # W1 = (temp.T * w).T
        # assert np.allclose(W,W1)

        # update step
        t_new = (1 + np.sqrt(1 + 4 * t ** 2))/2
        
        # update B
        B_old = B.copy()
        B = W + (t -1)/t_new * (W - W_old)

        # objective
        obj_value = sum([norm(Z[j] - X[j] @ B[:,:,j]) ** 2 for j in range(T)])/2 + lam * sum(norm(B, axis=(1,2)))
        obj.append(obj_value)
        
        t = t_new
        if max(norm(B_old - B), norm(W_old - W)) < 1e-8:
            break

    # print_stats()
    ## Step 3: 
    if np.any([norm(B[:,:,i], 'fro') for i in range(T)] == 0) or np.any([norm(V[:,:,i], 'fro') for i in range(T)] == 0):
        B = [B[:,:,i] for i in range(T)]
        V = [V[:,:,i] for i in range(T)]
        return B, V, obj
    else:
        B = [B[:,:,i] @ la.inv(sqrtm(B[:,:,i].T @ np.cov(X[i].T) @ B[:,:,i])) for i in range(T)]
        V = [V[:,:,i] @ la.inv(sqrtm(V[:,:,i].T @ np.cov(X[i].T) @ V[:,:,i])) for i in range(T)]
        return B, V, obj

# %%
## algorithm
# @profile
def comsir_parallel(X, Y, d, lam_ratio=0.1, max_iter = 100, n_jobs=2):
    """
    Sample size can be different, but dimension is the same. 
    d (int): Target rank/dimension for the projection

    """
    if isinstance(X, list) and isinstance(Y, list):
        # Case 1: Both X and Y are lists
        assert len(X) == len(Y), "Lengths of X and Y must match."
        T = len(X)

    elif isinstance(X, list) and isinstance(Y, np.ndarray):
        # Case 2: X is a list and Y is a NumPy array
        assert len(X) == Y.shape[1], "Length of X must match the number of columns in Y."
        T = len(X)
        Y = [Y[:, t] for t in range(T)]

    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        # Case 3: Both X and Y are NumPy arrays
        n = X.shape[0]
        if Y.shape == (n,):
            T = 1
            Y = Y.reshape((n,T))
        else:
            T = Y.shape[1]
            
        X = [X for _ in range(T)]
        Y = [Y[:, t] for t in range(T)]

    else:
        raise TypeError("X and Y must be either lists or NumPy arrays.")

    p =  X[0].shape[1]# Define the value of p here
    N = [x.shape[0] for x in X]
    M = np.zeros((p, p, T))
    V = np.zeros((p, d, T))
    XZ = np.zeros((p,d,T))
    XX = np.zeros((p,p,T))
    

    Exy = [np.zeros((i,p)) for i in N]
    Z = [np.zeros((i,d)) for i in N]  # Correct initialization of Z

    # ## Step 1: 
    def process_single_iteration(i, X, Y, d, max_iter = 100):
        # Compute M and Exy
        M_i, Exy_i = sir(X[i], Y[i])
        # Compute V using fantope
        V_i = fantope(M_i, d, lam_ratio=0.01, max_iter=max_iter)[0]
        # Compute Z
        Z_i = Exy_i @ V_i
        # Compute XZ
        XZ_i = - X[i].T @ Z_i
        # Compute XX
        XX_i = X[i].T @ X[i]
        return (M_i, V_i, Z_i, XZ_i, XX_i, Exy_i)
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_iteration, X=X, Y=Y, d=d, max_iter=max_iter)
    results = Parallel(n_jobs=n_jobs)(delayed(process_func)(i) for i in range(T))
    
    # Unpack results into respective arrays
    for i, (M_i, V_i, Z_i, XZ_i, XX_i, Exy_i) in enumerate(results):
        M[:, :, i] = M_i
        V[:, :, i] = V_i
        Z[i] = Z_i
        XZ[:, :, i] = XZ_i
        XX[:, :, i] = XX_i
        Exy[i] = Exy_i

    # ## Not parallel
    # for i in range(T):
    #     M[:, :, i], Exy[i] = sir(X[i], Y[i])
    #     V[:, :, i] = fantope(M[:, :, i], d, lam_ratio=0.01, max_iter=max_iter)
    #     Z[i] = Exy[i] @ V[:, :, i]
    #     XZ[:, :, i] = - X[i].T @ Z[i]
    #     XX[:, :, i] = X[i].T @ X[i]    

    ## Step 2: 
    lam_max = max(norm(XZ, axis=(1,2)))
    # print(lam_max)
    # if lam is None:
    lam = lam_max * lam_ratio
    ## initialization
    # max_iter = 100
    t0 = 1 / sum([norm(x.T @ x, 2) for x in X])
    t = 1
    W = np.zeros((p, d, T))
    B = np.zeros((p, d, T))
    obj = []
    temp = np.zeros((p, d, T))
    for i in range(max_iter): 
        # soft-threshold update W
        W_old = W.copy()
        for j in range(T):
            temp[:,:,j] = W[:, :, j] - t0 * (XZ[:, :, j] + XX[:, :, j] @ W[:, :, j])
        w = np.maximum(1 - lam * t0 / norm(temp, axis=(1,2)), 0)
        assert(w.shape == (p,))

        W = temp * w[:, np.newaxis, np.newaxis]
        # W1 = (temp.T * w).T
        # assert np.allclose(W,W1)

        # update step
        t_new = (1 + np.sqrt(1 + 4 * t ** 2))/2
        
        # update B
        B_old = B.copy()
        B = W + (t -1)/t_new * (W - W_old)

        # objective
        obj_value = sum([norm(Z[j] - X[j] @ B[:,:,j]) ** 2 for j in range(T)])/2 + lam * sum(norm(B, axis=(1,2)))
        obj.append(obj_value)
        
        t = t_new
        if max(norm(B_old - B), norm(W_old - W)) < 1e-8:
            break

    # print_stats()
    ## Step 3: 
    if np.any([norm(B[:,:,i], 'fro') for i in range(T)] == 0) or np.any([norm(V[:,:,i], 'fro') for i in range(T)] == 0):
        B = [B[:,:,i] for i in range(T)]
        V = [V[:,:,i] for i in range(T)]
        print("some Bt are zeros.")
        return B, V, obj
    else:
        B = [B[:,:,i] @ la.inv(sqrtm(B[:,:,i].T @ np.cov(X[i].T) @ B[:,:,i])) for i in range(T)]
        V = [V[:,:,i] @ la.inv(sqrtm(V[:,:,i].T @ np.cov(X[i].T) @ V[:,:,i])) for i in range(T)]
        return B, V, obj

# %%
# X,Y,d,beta,S = gen_data(100,100,1,1)
# comsir_parallel(X, Y, d, lam_ratio=0.2, max_iter=100)

# %%
def loss(B, Bhat):
    try:
        P_B = B @ la.inv(B.T @ B) @ B.T
    except Exception as e:
        print(f"Error with first B: {str(e)}")
        P_B = B
    try:
        P_Bhat = Bhat @ la.inv(Bhat.T @ Bhat) @ Bhat.T
    except Exception as e:
        print(f"Error with second B: {str(e)}")
        P_Bhat = Bhat
    return norm(P_B - P_Bhat, 'fro')

# %%
def var_measure(B, S):
    S1_final = np.where(norm(B,axis=1) > 0)[0]
    fp1 = set(S1_final)-set(S)
    fn1 = set(S)-set(S1_final)
    return len(fp1), len(fn1)

# %%
def evaluate_single_fold(args):
    """
    Helper function to evaluate a single fold for a specific task and lambda ratio.
    
    Parameters:
    -----------
    args : tuple
        Contains (X, Y, d, task_idx, train_idx, val_idx, lam_ratio)
    
    Returns:
    --------
    tuple : (lam_ratio, task_idx, val_score)
    """
    X, Y, d, task_idx, train_idx, val_idx, lam_ratio = args
    
    try:
        # Split data for current task
        X_train = [X[i] if i != task_idx else X[i][train_idx] for i in range(len(X))]
        Y_train = [Y[i] if i != task_idx else Y[i][train_idx] for i in range(len(Y))]
        X_val = X[task_idx][val_idx]
        Y_val = Y[task_idx][val_idx]
        
        # Train model with current lam_ratio
        result = comsir_parallel(X_train, Y_train, d, lam_ratio=lam_ratio, max_iter=100)[0]
        
        # Calculate validation score
        val_score = distance_correlation(X_val @ result[task_idx], Y_val)
        return lam_ratio, task_idx, val_score
        
    except Exception as e:
        print(f"Error with lam_ratio={lam_ratio} on task {task_idx}: {str(e)}")
        return lam_ratio, task_idx, float('-inf')

def cross_validate_lam_ratio_parallel(X, Y, d, n_splits=3, n_samples=10, n_jobs=None):
    """
    Perform parallel k-fold cross-validation to find the best lam_ratio parameter.
    
    Parameters:
    -----------
    X : list of arrays
        List of input matrices for each task
    Y : list of arrays
        List of output matrices for each task
    d : int
        Dimension parameter for comsir
    n_splits : int
        Number of folds for cross-validation
    n_samples : int
        Number of lam_ratio values to test
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available CPU cores.
    
    Returns:
    --------
    best_lam_ratio : float
        The best lam_ratio value found
    cv_results : dict
        Dictionary containing all tested values and their corresponding cross-validation scores
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Generate log-scale samples between 10^-2 and 1
    lam_ratios = np.logspace(-2, 0, n_samples)
    T = len(X)
    
    # Initialize arrays to store indices for each task
    indices = [np.arange(X[i].shape[0]) for i in range(T)]
    
    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate all combinations of tasks, folds, and lambda ratios
    all_combinations = []
    for task_idx in range(T):
        for train_idx, val_idx in kf.split(indices[task_idx]):
            for lam_ratio in lam_ratios:
                all_combinations.append((X, Y, d, task_idx, train_idx, val_idx, lam_ratio))
    
    # Parallel processing of all combinations
    print(f"Starting parallel processing with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_single_fold)(arg) for arg in all_combinations)
    
    # Process results
    cv_results = {lam: [] for lam in lam_ratios}
    for lam_ratio, _, val_score in results:
        cv_results[lam_ratio].append(val_score)
    
    # Calculate mean cross-validation score for each lam_ratio
    mean_cv_scores = {lam: np.mean(scores) for lam, scores in cv_results.items()}
    
    # Find best lam_ratio (maximum mean score)
    best_lam_ratio = max(mean_cv_scores.items(), key=lambda x: x[1])[0]
    
    return best_lam_ratio, cv_results


# %%
# X, Y, d, beta, S = gen_data(100, 100, 1, 1)
# best_lam_ratio, cv_results = cross_validate_lam_ratio_parallel(X, Y, d, n_splits=3, n_samples=10, n_jobs=6)
# best_lam_ratio