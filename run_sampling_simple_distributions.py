#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maarten l. Jung

Runs the Bayesian independence sampler with simple combinations of prior 
distributions and likelihood functions.
"""

import numpy as np
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
from pathlib import Path

jsonpickle_numpy.register_handlers()


def calc_ess(samples, step_size = 1, max_lag = 2000):
    # based on BEAST https://beast.community/
    # code adapted from the GitHub page (commit d2b36cf)
    # https://github.com/beast-dev/beast-mcmc/blob/f62cf395998ce1cd0538412177a13e672f96e9ac/src/dr/inference/trace/TraceCorrelation.java
    #
    # The effective samples size (ess) is defined as the number of samples devided by the (integrated) autocorrelation time (act).
    # The act is the delay at which the autocorrelation between the sample sequence and the delayed/shifted sample sequence is zero.
    # The act is calculated by first finding the approx. point where the autocovariance function (acf) is zero (indicated by the sum of 
    # adjacent values being less than zero). The acf is then assumed to be approx. linear and the act can thus be approximated by twice 
    # the area under the acf curve divided by the value of the acf at delay = 0.
    
    num_samples = len(samples)
    
    assert num_samples >= 10, "At least 10 samples needed!"
    
    if type(samples) is not np.ndarray:
        samples = np.array(samples)
    
    biggest_lag = min(num_samples-1, max_lag)
    
    # autocovariance function
    autocov = np.zeros(biggest_lag, dtype = np.float64)
    
    cent_samples = samples - samples.mean()
    
    for lag in range(biggest_lag):
      
        autocov[lag] = (cent_samples[:num_samples-lag] @ cent_samples[lag:]) / (num_samples - lag)
        
        if lag == 0: # (autocovariance at lag 0) == variance of samples
            integrated_autocov = autocov[lag]
        elif lag % 2 == 0:
            # sum of adjacent pairs of autocovariances must be positive (Geyer, 1992)
            sum_adj_pairs = autocov[lag-1] + autocov[lag]
            if sum_adj_pairs > 0:
                integrated_autocov += 2.0 * sum_adj_pairs
            else:
                break
        
    # integrated autocorrelation time
    if autocov[0] == 0:
        act = 0
    else:
        act = (step_size * integrated_autocov) / autocov[0]
    
    # effective sample size  
    if act == 0:
        ess = 1
    else:
        ess = (step_size * num_samples) / act
    
    return ess


def bayes_indep_sampler(prior, likelihood, ess_crit = 500, max_iter = 10**5):
    
    if type(likelihood) is not np.ndarray:
        likelihood = np.array(likelihood, dtype = np.float64) 
     
    prior_draws = np.random.choice(len(prior), size = max_iter + 1, replace = True, p = prior)
    lh_prior_draws = likelihood[prior_draws]
    uniform_draws = np.random.uniform(size = max_iter)
    
    posterior_samples = np.zeros(max_iter, dtype = np.int32)    
    pos_last_accepted = 0
   
    for i in range(max_iter):
        acceptance_value = lh_prior_draws[i+1] / lh_prior_draws[pos_last_accepted]
        if uniform_draws[i] < acceptance_value:
            pos_last_accepted = i + 1
        posterior_samples[i] = prior_draws[pos_last_accepted]
        
        if i >= ess_crit:
            if calc_ess(posterior_samples[:i+1]) >= ess_crit:
                return posterior_samples[:i+1]
   
    return posterior_samples


# Prior: N(50, 20^2); Likelihood: N(50, 20^2) ---------------------------------
    
## Run sampling ---------------------------------------------------------------
    
num_rep = 5000
ess_crits = [500, 1000, 5000]

prior = norm.pdf(range(1, 100), 50, 20)
prior /= prior.sum()
lik = prior

dict_sampling = {key: [] for key in ["ess_criterion", "posterior_samples", "n_posterior_samples"]}
for ess_crit in ess_crits:
    print(f"\nPrior: N(50, 20^2), Likelihood: N(50, 20^2); ESS criterion: {ess_crit}\n")
    pos_samples = Parallel(n_jobs = -1, verbose = 50) \
                          (delayed(bayes_indep_sampler)(prior, lik, ess_crit) for _ in range(num_rep))
    n_pos_samples = np.array([len(samples) for samples in pos_samples], dtype = np.int32)        
    
    dict_sampling["ess_criterion"].append(ess_crit)        
    dict_sampling["posterior_samples"].append(pos_samples)
    dict_sampling["n_posterior_samples"].append(n_pos_samples)

## Save sampling data ---------------------------------------------------------
    
folder = Path.cwd() / "data"
if not folder.is_dir():
    folder.mkdir()
fname_out = folder / f"sampling_prior_normal_lik_normal_ess_crit_{'_'.join(map(str, ess_crits))}_rep_{num_rep}.json"         

pickled = jsonpickle.encode(dict_sampling)
with open(fname_out, 'w') as file_out:
    json.dump(pickled, file_out)   

all_n_samples = np.array(dict_sampling["n_posterior_samples"], dtype = np.int32)
index = pd.MultiIndex.from_product([range(s) for s in all_n_samples.shape], 
                                   names = ["ess_criterion", "repetition"])
df = pd.DataFrame(all_n_samples.flatten(), index = index).reset_index()
df = df.rename(columns = {0 : "n_samples"})

df["ess_criterion"] = df["ess_criterion"].replace(range(len(ess_crits)), ess_crits)
df["repetition"] = df["repetition"] + 1

fname_out = folder / "n_samples_prior_normal_lik_normal_all_ess_crits.csv"
df.to_csv(fname_out, index = False)

# Prior: N(50, 20^2); Likelihood: ~ 1/Prior -----------------------------------
    
## Run sampling ---------------------------------------------------------------

prior = norm.pdf(range(1, 100), 50, 20)
prior /= prior.sum()
lik = 1.0 / prior

dict_sampling = {key: [] for key in ["ess_criterion", "posterior_samples", "n_posterior_samples"]}
for ess_crit in ess_crits:
    print(f"\nPrior: N(50, 20^2); Likelihood: ~ 1/Prior; ESS criterion: {ess_crit}\n")
    pos_samples = Parallel(n_jobs = -1, verbose = 50) \
                          (delayed(bayes_indep_sampler)(prior, lik, ess_crit) for _ in range(num_rep))
    n_pos_samples = np.array([len(samples) for samples in pos_samples], dtype = np.int32)        
    
    dict_sampling["ess_criterion"].append(ess_crit)        
    dict_sampling["posterior_samples"].append(pos_samples)
    dict_sampling["n_posterior_samples"].append(n_pos_samples)

## Save sampling data ---------------------------------------------------------
    
fname_out = folder / f"sampling_prior_normal_lik_recinormal_ess_crit_{'_'.join(map(str, ess_crits))}_rep_{num_rep}.json"         

pickled = jsonpickle.encode(dict_sampling)
with open(fname_out, 'w') as file_out:
    json.dump(pickled, file_out)     

all_n_samples = np.array(dict_sampling["n_posterior_samples"], dtype = np.int32)
index = pd.MultiIndex.from_product([range(s) for s in all_n_samples.shape], 
                                   names = ["ess_criterion", "repetition"])
df = pd.DataFrame(all_n_samples.flatten(), index = index).reset_index()
df = df.rename(columns = {0 : "n_samples"})

df["ess_criterion"] = df["ess_criterion"].replace(range(len(ess_crits)), ess_crits)
df["repetition"] = df["repetition"] + 1

fname_out = folder / "n_samples_prior_normal_lik_recinormal_all_ess_crits.csv"
df.to_csv(fname_out, index = False)    
    
# Prior: Unif(0, 99); Likelihood ~ N(50, 20^2) --------------------------------
    
## Run sampling ---------------------------------------------------------------

prior = np.repeat(1.0/99, 99)
lik = norm.pdf(range(1, 100), 50, 20)

dict_sampling = {key: [] for key in ["ess_criterion", "posterior_samples", "n_posterior_samples"]}
for ess_crit in ess_crits:
    print(f"\nPrior: Unif(0, 99); Likelihood ~ N(50, 20^2); ESS criterion: {ess_crit}\n")
    pos_samples = Parallel(n_jobs = -1, verbose = 50) \
                          (delayed(bayes_indep_sampler)(prior, lik, ess_crit) for _ in range(num_rep))
    n_pos_samples = np.array([len(samples) for samples in pos_samples], dtype = np.int32)        
    
    dict_sampling["ess_criterion"].append(ess_crit)        
    dict_sampling["posterior_samples"].append(pos_samples)
    dict_sampling["n_posterior_samples"].append(n_pos_samples)

## Save sampling data ---------------------------------------------------------
    
fname_out = folder / f"sampling_prior_unif_lik_normal_ess_crit_{'_'.join(map(str, ess_crits))}_rep_{num_rep}.json"         

pickled = jsonpickle.encode(dict_sampling)
with open(fname_out, 'w') as file_out:
    json.dump(pickled, file_out)
    
all_n_samples = np.array(dict_sampling["n_posterior_samples"], dtype = np.int32)
index = pd.MultiIndex.from_product([range(s) for s in all_n_samples.shape], 
                                   names = ["ess_criterion", "repetition"])
df = pd.DataFrame(all_n_samples.flatten(), index = index).reset_index()
df = df.rename(columns = {0 : "n_samples"})

df["ess_criterion"] = df["ess_criterion"].replace(range(len(ess_crits)), ess_crits)
df["repetition"] = df["repetition"] + 1

fname_out = folder / "n_samples_prior_unif_lik_normal_all_ess_crits.csv"
df.to_csv(fname_out, index = False)     
