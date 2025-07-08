# %%
## packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
import scipy.linalg as la
from line_profiler import LineProfiler
import time
from statsmodels.stats.dist_dependence_measures import distance_correlation
import pandas as pd
from numba import jit
from functools import partial
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from multiprocessing import cpu_count
import seaborn as sns
from gen_data import *
from utility_fun import *

plt.rcParams.update({'font.size': 14})
# %%
def one_run(n, p, Model, seed_id, t=2):
    """
    Run one experiment and return results in a structured DataFrame.
    
    Parameters:
    -----------
    n : int
        Sample size
    p : int
        Number of features
    Model : int
        Model identifier
    seed_id : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing experiment results
    """
    # Generate data
    X, Y, d, beta, S = gen_data(n, p, Model, seed_id, t)
    T = len(X)
    
    # Get results using different methods
    # best_lam_ratio, _ = cross_validate_lam_ratio_parallel(X, Y, d, n_splits=5, n_samples=20, n_jobs=20)
    best_lam_ratio = 0.15
    B1, _, _ = comsir_parallel(X, Y, d, lam_ratio=best_lam_ratio)
    B2 = [comsir(X[i], Y[i], d, lam_ratio=best_lam_ratio)[0][0] for i in range(T)]

    Xc = np.concatenate(X)
    Yc = np.concatenate(Y)
    assert Xc.shape==(T*n, p) and Yc.shape==(T*n,)
    B3 = comsir(Xc, Yc, d, lam_ratio=best_lam_ratio)[0][0]
    
    # Create dictionary to store results
    results = {}
    
    # Store metadata
    results.update({
        'n': n,
        'p': p,
        'Model': Model,
        'seed_id': seed_id
    })
    
    # Store error metrics
    results['multitask'] = np.mean([loss(beta[i], B1[i]) for i in range(T)])
    results['individual'] = np.mean([loss(beta[i], B2[i]) for i in range(T)])
    results['single'] = np.mean([loss(beta[i], B3) for i in range(T)])

    # Store variance measures for each time point
    for t in range(T):
        var_measures_B1 = var_measure(B1[t], S)
        var_measures_B2 = var_measure(B2[t], S)
        
        results.update({
            f'FP_t{t}_mul': var_measures_B1[0],
            f'FN_t{t}_mul': var_measures_B1[1],
            f'FP_t{t}_ind': var_measures_B2[0],
            f'FN_t{t}_ind': var_measures_B2[1]
        })
    
    var_measures_B3 = var_measure(B3, S)
    results.update({
        f'FP_sin': var_measures_B3[0],
        f'FN_sin': var_measures_B3[1]
    })
    
    # Create DataFrame with meaningful column names
    df = pd.DataFrame([results])
    
    # Round numerical values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    return df

def combine_runs(runs_list):
    return pd.concat(runs_list, ignore_index=True)

# %% repetitions
rep = 1
# %%
# Example 1 and 2: 
data_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_run)(para[0],para[1],Model, seedid)
    for seedid in range(rep)
    for para in [[1000, 1000], [1000, 1200], [1000, 1500]]
    for Model in [1,2]
    ) 

all_results = combine_runs(data_list)
all_results
fname = f'comsir.pkl'
all_results.to_pickle(fname, compression='gzip')     

# %%
## reproduce Figure 1 in the paper

sim = pd.read_pickle('comsir.pkl', compression='gzip')
print(sim.shape)
sim = sim.rename(columns={
    'n': 'Sample Size',
    'p': 'Feature Dimension',
    'Model': 'Example',
    'multitask': 'COMSIR',
    'individual': 'Individual',
    'single': 'Single'
})
sim.head()

df_loss = pd.melt(sim,
        id_vars=['Sample Size','Feature Dimension','Example'],
        value_vars=['COMSIR', 'Individual', 'Single'],
        var_name='Method')

df_loss.Example = df_loss.Example.astype(int)
df_loss['Sample Size'] = df_loss['Sample Size'].astype(int)
df_loss['Feature Dimension'] = df_loss['Feature Dimension'].astype(int)

df_loss['Setting'] = 'n='+ df_loss['Sample Size'].astype(str) + ' \np=' + df_loss['Feature Dimension'].astype(str)

palette_dict = {'COMSIR': '#d62728', 
                'Individual': '#1f77b4', 
                'Single': '#2ca02c' 
                }

g1 = sns.catplot(data=df_loss, x='Setting', y='value',
                  col='Example', 
                  col_wrap=2,
                  hue='Method', 
                  kind='box',
                  palette="grey",
                  legend=True)
g1.set_ylabels("General Loss")
g1.figure.savefig('comsir.pdf', bbox_inches='tight')

# %%
# Example 2: Sample size
p = 1500
n_values = list(range(500, 1600, 100))
models = [2]
# Create all combinations
parameters = [(n, p, model, seedid) 
             for seedid in range(rep)
             for n in n_values
             for model in models]

data_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_run)(*params) for params in parameters
)

all_results = combine_runs(data_list)
all_results
fname = f'samplesize.pkl'
all_results.to_pickle(fname, compression='gzip')  


#%% reproduce Figure 2 in the paper
sim = pd.read_pickle('samplesize.pkl', compression='gzip')
print(sim.shape)
sim = sim.rename(columns={
    'n': 'Sample Size',
    'p': 'Feature Dimension',
    'Model': 'Model'
    # Add more columns to rename as needed
})

df_loss = pd.melt(sim,
        id_vars=['Sample Size','Feature Dimension','Model'],
        value_vars=['COMSIR', 'Individual', 'Single'],
        var_name='Method')

df_loss.Model = df_loss.Model.astype(int)
df_loss['Sample Size'] = df_loss['Sample Size'].astype(int)
df_loss['Feature Dimension'] = df_loss['Feature Dimension'].astype(int)

df_loss['Setting'] = 'n='+ df_loss['Sample Size'].astype(str) + ' \np=' + df_loss['Feature Dimension'].astype(str)

palette_dict = {'COMSIR': '#d62728', 
                'Individual': '#1f77b4', 
                'Single': '#2ca02c' 
                }

marker_dict = {
    'COMSIR': 'o',        # circle
    'Individual': 's',    # square
    'Single': 'D'         # diamond
}

g1 = sns.relplot(
    data=df_loss[df_loss.Model == 3], 
    x='Sample Size', 
    y='value',
    hue='Method',
    style='Method',              # differentiate by marker
    markers=marker_dict,         # assign marker shapes
    kind='line',
    palette=palette_dict,      # omit for no color, or remove for grayscale
    errorbar=('se', 1),
    legend=True,
    height=5,
    aspect=1.5
)
g1.set_ylabels("Projection Loss")
g1.figure.savefig('samplesize_ex2.pdf', bbox_inches='tight')


#  %% Example 3: Task number
p = 1000
n = 800
t_values = list(range(2, 21, 1))
model = 3

# Create all combinations
parameters = [(n, p, model, seedid, t) 
             for seedid in range(rep)
             for t in t_values]

data_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_run)(*params) for params in parameters
)

all_results = combine_runs(data_list)
fname = f'tasks_p{p}_allT.pkl'
all_results.to_pickle(fname, compression='gzip')   

# %% reproduce Figure 3 in the paper
sim = pd.read_pickle('tasks_p1000_allT.pkl', compression='gzip')
print(sim.shape)
sim = sim.rename(columns={
    'n': 'Sample Size',
    'p': 'Feature Dimension',
})

df_loss = pd.melt(sim,
        id_vars=['Sample Size','Feature Dimension','Task'],
        value_vars=['COMSIR', 'Individual', 'Single'],
        var_name='Method')

df_loss['Task'] = df_loss['Task'].astype(str)
df_loss['Sample Size'] = df_loss['Sample Size'].astype(int)
df_loss['Feature Dimension'] = df_loss['Feature Dimension'].astype(int)

g1 = sns.relplot(data=df_loss, 
                x='Task', 
                y='value',
                hue='Method', 
                kind='line',
                palette=palette_dict,
                style='Method', 
                markers=marker_dict,         # assign marker shapes
                errorbar = ('se',1),
                legend=True,
                height=5,
                aspect=2)
g1.set_ylabels("Projection Loss")
g1.set_xlabels("Number of Tasks")
g1.figure.savefig('tasks.pdf', bbox_inches='tight')


#%% reproduce Table 1 in the paper
p = 1500
n_seq = [1000,1200,1500]
t = 8
models = [1,2,3]

# Create all combinations
parameters = [(n, p, model, seedid, t) 
             for seedid in range(rep)
             for n in n_seq
             for model in models]

data_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_run)(*params) for params in parameters
)
all_results = combine_runs(data_list)
fname = f'tasks_p{p}_t{t}_three.pkl'
all_results.to_pickle(fname, compression='gzip')  


#%% reproduce Table 2 in the paper
def onerun_d(n,p,model,seed,T):
    X, Y = gen_data(n,p,model,seed,T)[:2]
    est_d = np.max([select_d(x, y) for x,y in zip(X,Y)])
    # dd = Parallel(n_jobs=T, verbose=10)(
    #     delayed(select_d)(x,y) for x,y in zip(X,Y)
    # )
    # est_d = np.max(dd)

    results = {}
    # Store metadata
    results.update({
        'Task': T,
        'n': n,
        'p': p,
        'Model': model,
        'seed_id': seed,
        'dim': est_d,
    })

    df = pd.DataFrame([results])
    return df

p = 1500
n_seq = [1000,1200,1500]
t = 8
models = [1,2,3]

# Create all combinations
parameters = [(n, p, model, seedid, t) 
             for seedid in range(rep)
             for n in n_seq
             for model in models]

data_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(onerun_d)(*params) for params in parameters
)
all_results = combine_runs(data_list)
all_results
fname = f'dim_p{p}_t{t}_S10_random.pkl'
all_results.to_pickle(fname, compression='gzip')     

sim = pd.read_pickle(fname, compression='gzip')
grouped_stats = sim.groupby(['Model','p'],dropna=True).agg(['mean', 'std'])
print(grouped_stats)