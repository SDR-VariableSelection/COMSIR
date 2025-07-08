
import numpy as np

def gen_data(n,p,Model,seed_id, T=2):
    rng = np.random.default_rng(seed_id)

    if Model == 0:
        ## model 1 (d = 1)
        d = 1
        beta1 = np.zeros((p,d))
        beta1[:5,0] = 0.3
        beta1[5:10,0] = 1
        beta2 = np.zeros((p,d))
        beta2[:5,0] = 1
        beta2[5:10,0] = 0.3
        x1 = rng.standard_normal((n,p))
        x2 = rng.standard_normal((n,p))
        # y1 = np.exp(x1 @ beta1[:,0]) + 0.2 * np.random.randn(n)
        # y2 = np.tanh(x2 @ beta2[:,0]) + 0.2 * np.random.randn(n)
        y1 = (x1 @ beta1[:,0]) + 0.2 * rng.standard_normal(n)
        y2 = (x2 @ beta2[:,0]) + 0.2 * rng.standard_normal(n)
        S = [0,1,2,3,4,5,6,7,8,9]
        beta = [beta1, beta2]
        X = [x1,x2]
        Y = [y1,y2]
    elif Model == 1:
        ## model 2 (d = 2)
        d = 2
        beta1 = np.zeros((p,d))
        beta1[:2,0] = 1
        beta1[7:10,0] = 0.3
        beta1[2:7,1] = 1
        beta2 = np.zeros((p,d))
        beta2[:5,0] = 1
        beta2[5:10,1] = 0.3
        beta3 = np.zeros((p,d))
        beta3[5:10,0] = 1
        beta3[:5,1] = 0.3
        
        X = [rng.standard_normal((n, p)) for _ in range(3)]

        y1 = (X[0] @ beta1[:,0]) + np.tanh(X[0] @ beta1[:,1]) + 0.2 * rng.standard_normal(n)
        y2 = np.exp(X[1] @ beta2[:,0]) * np.sign(X[1] @ beta2[:,1]) + 0.2 * rng.standard_normal(n)
        y3 = (X[2] @ beta3[:,0]) ** 3 + 2 * np.arctan(X[2] @ beta3[:,1]) + 0.2 * rng.standard_normal(n)

        S = [0,1,2,3,4,5,6,7,8,9]
        beta = [beta1, beta2, beta3]
        Y = [y1,y2,y3]
    elif Model == 2:
        ## model 2 (d = 2)
        d = 2
        beta_patterns = [
            (2, 0.3, 1),   # beta1: first 2 cols 0.3, rest 1
            (4, 0.3, 1),   # beta2: first 4 cols 0.3, rest 1
            (6, 0.3, 1),   # beta3: first 6 cols 0.3, rest 1
            (6, 1, 0.3),   # beta4: first 6 cols 1, rest 0.3
            (4, 1, 0.3),   # beta5: first 4 cols 1, rest 0.3
            (2, 1, 0.3),   # beta6: first 2 cols 1, rest 0.3
        ]

        # Generate beta matrices
        beta = []
        for split_idx, val1, val2 in beta_patterns:
            beta_i = np.zeros((p, d))
            beta_i[:split_idx, 0] = rng.uniform(1,5,size=split_idx)
            beta_i[split_idx:10, 1] = rng.uniform(1,5,size=10-split_idx)
            beta.append(beta_i)

        # Generate X matrices
        X = [rng.standard_normal((n, p)) for _ in range(6)]

        # Generate Y values
        Y = []
        for x, b in zip(X, beta):
            y = np.exp(x @ b[:,0]) + np.tanh(x @ b[:,1]) + 0.2 * rng.standard_normal(n)
            # (x @ b[:, 0]) / (0.5 + (1.5 + x @ b[:, 1])**2) + 0.2 * rng.standard_normal(n)
        
            # np.exp(x @ b[:,0]) * np.sign(x @ b[:,1]) + 0.2 * rng.standard_normal(n)
            
            Y.append(y)

        S = list(range(10))

    elif Model == 3:
        d = 2
        q = 10
        # Generate beta matrices
        beta = []
        split = rng.integers(2, q-2, T)
        vals = rng.uniform(1, 5, size=(T, 2))
        for split_idx, val in zip(split,vals):
            beta_i = np.zeros((p, d))
            beta_i[:split_idx, 0] = rng.uniform(1,5,size=split_idx) #val[0]
            beta_i[split_idx:q, 1] = rng.uniform(1,5,size=q-split_idx) #val[1]
            beta.append(beta_i)

        # Generate X matrices
        X = [rng.standard_normal((n, p)) for _ in range(T)]

        # Generate Y values
        Y = []
        models = rng.integers(1,5,T)
        for x, b, mod in zip(X, beta, models):
            if mod == 1:
                y = (x @ b[:,0]) + np.tanh(x @ b[:,1]) + 0.2 * rng.standard_normal(n)
            elif mod == 2:
                y = np.exp(x @ b[:,0]) * np.sign(x @ b[:,1]) + 0.2 * rng.standard_normal(n)
            elif mod == 3:
                y = (x @ b[:,0]) ** 3 + 2 * np.arctan(x @ b[:,1]) + 0.2 * rng.standard_normal(n)
            elif mod == 4:
                y = (x @ b[:,0]) / (0.5 + (1.5 + x @ b[:,1])**2) + 0.2 * rng.standard_normal(n)
            Y.append(y)

        S = list(range(q))
    
    return X,Y,d,beta,S