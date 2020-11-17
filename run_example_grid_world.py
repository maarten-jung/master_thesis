#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# code adapted from
# https://github.com/SSchwoebel/BalancingControl/blob/master/run_example_multiarmedbandid.py
# (commit f1d86ba)
# and 
# https://github.com/SSchwoebel/ActiveInferenceBPBethe/blob/master/run_example_gridworld.py
# (commit 76301d4)
"""
Simulates (Bayesian habit-learning) agents with different habitual tendencies 
which repeatedly navigate through a grid world
"""
import world 
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import itertools
import numpy as np
import json
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from pathlib import Path
import gc

jsonpickle_numpy.register_handlers()

"""
run function
"""
def run_agent(par_list, trials, T, ns, na, nr, nc, deval=False):
    
    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, avg, Rho, utility = par_list
        
    """
    create matrices
    """
       
    #generating probability of observations in each state
    A = np.eye(ns) # observation uncertainty = 0
        
    
    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))
    
    L = 4 # grid length
    c = 1.0 # transition uncertainty = 0
    actions = np.array([[-1,0],[0,-1], [1,0], [0,1]])
    
    cert_arr = np.zeros(ns)
    for s in range(ns):
        x = s//L
        y = s%L
            
        cert_arr[s] = c
        for u in range(na):
            x = s//L+actions[u][0]
            y = s%L+actions[u][1]
            
            #check if state goes over boundary
            if x < 0:
                x = 0
            elif x == L:
                x = L-1
                
            if y < 0:
                y = 0
            elif y == L:
                y = L-1
            
            s_new = L*x + y
            if s_new == s:
                B[s, s, u] = 1
            else:
                B[s, s, u] = 1-c
                B[s_new, s, u] = c
                
            
    # agent's initial estimate of reward generation probability
    C_agent = np.zeros((nr, ns, nc))
    for c in range(nc):
        C_agent[:,:,c] = np.array([[0.0, 1.0] if state == 11 else [1.0, 0.0] for state in range(ns)]).T
    
    # context transition matrix
    transition_matrix_context = np.zeros((nc, nc))
    transition_matrix_context[:,:] = 1.0
                            
    """
    create environment (grid world)
    """
    
    environment = env.GridWorld(A, B, Rho, trials = trials, T = T)
    
    
    """
    create policies
    """
    
    pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))
    
    npi = pol.shape[0]
    
    # concentration parameters
    alphas = np.zeros((npi, nc)) + learn_pol

    prior_pi = alphas / alphas.sum(axis=0)
    
    
    """
    set state prior (where agent thinks it starts)
    """
    
    state_prior = np.zeros((ns))
    
    state_prior[1] = 1.0

    """
    set action selection method
    """

    if avg:
    
        ac_sel = asl.AveragedSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    else:
        
        ac_sel = asl.MaxSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    
  
    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, 
                                           transition_matrix_context, 
                                           state_prior, 
                                           utility, 
                                           prior_pi, 
                                           alphas, 
                                           dirichlet_rew_params=None, 
                                           T=T)
    
    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns, 
                      prior_context = None,
                      learn_habit = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)
    

    """
    create world
    """
    
    w = world.World(environment, bayes_pln, trials = trials, T = T)
    
    """
    simulate experiment
    """
    
    if not deval:       
        w.simulate_experiment(range(trials))
        
    else:
        w.simulate_experiment(range(trials//2))
        # reset utility to implement devaluation
        ut = utility[1:].sum()
        bayes_prc.prior_rewards[2:] = ut / (nr-2)
        bayes_prc.prior_rewards[:2] = (1-ut) / 2
        
        w.simulate_experiment(range(trials//2, trials))   
  
    return w

def main():

    """
    set parameters
    """
    
    na = 4 # number of actions
    T = na + 1 # number of time steps in each miniblock/episode
    ns = 16 # number of states    
    nr = 2 # number of rewards (0 and 1)
    nc = 1 # number of contexts
    # conditional distribution over rewards given states
    Rho = np.array([[0.0, 1.0] if state == 11 else [1.0, 0.0] for state in range(ns)]).T 
    
    avg = True # type of action selection (average vs. mode)
    u = 0.999 
    utility = np.array([1-u, u])
    # modifies concentration parameters of dirichlet distribution
    # smaller values -> stronger habits
    alphas = [1, 10, 100]
    
    mb = 50 # number of miniblocks/episodes
    rep = 100 # number of repetitions
    
   
    """
    run simulations
    """ 
    
    folder = Path.cwd() / "data"
    if not folder.is_dir():
        folder.mkdir()
    
    worlds = [[None]*rep for _ in range(len(alphas))]
    for i, alpha in enumerate(alphas):      
        for r in range(rep):
            print(f"habit {i+1} out of {len(alphas)}, repetition {r+1} out of {rep}")
            worlds[i][r] = run_agent((alpha, avg, Rho, utility), mb, T, ns, na, nr, nc)            
        # save worlds separately for each alpha
        fname = folder / f"worlds_alpha_{round(alpha, 1)}_mb_{mb}_rep_{rep}.json"         
        pickled = pickle.encode(worlds[i])
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)    
              
    gc.collect()   
    
    return worlds 
 

if __name__ == "__main__":
    results = main()
