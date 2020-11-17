#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# code adapted from
# https://github.com/SSchwoebel/ActiveInferenceBPBethe/blob/master/run_example_gridworld.py
# (commit 76301d4)
"""
Plots the policies (paths) chosen by agents with different habitual tendencies
and calculates the average KL divergence from the marginal prior distributions 
over policies to the normalized likelihood functions for each episode
"""

import numpy as np
import pandas as pd
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from pathlib import Path
import matplotlib.pylab as plt
import seaborn as sns

jsonpickle_numpy.register_handlers()

# computes the KL divergernce from p to q (for discrete p and q)
def kl_div(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

folder_data = Path.cwd() / "data"
folder_fig = Path.cwd() / "figures"
if not folder_fig.is_dir():
    folder_fig.mkdir()

L = 4 # grid length
ns = 16 # number of states  
na = 4 # number of actions
T = na + 1 # number of time steps in each miniblock/episode
trials = 50
u = 0.999 
utility = np.array([1-u, u])
policy_range = np.arange(1, 4**na + 1)
goal = 11

df_kl_div = pd.DataFrame({"habitual_tendency": [],
                          "episode": [],
                          "kl_div": []})

alpha_values = [1, 10, 100]

for alpha in alpha_values:
    fname_in = folder_data / f"worlds_alpha_{alpha}_mb_50_rep_100.json"
    
    with open(fname_in, 'r') as file_in:
        data = json.load(file_in)    
    worlds = jsonpickle.decode(data)
    
    n_succ = 0
    kl_div_array = np.full((len(worlds), trials), np.nan)
    for w_ind, w in enumerate(worlds):
       
        n_succ += (w.environment.hidden_states[:,-1]==goal).sum()
      
        #find successful and unsuccessful runs
        successfull = np.where(w.environment.hidden_states[:,-1]==goal)[0]
        unsuccessfull = np.where(w.environment.hidden_states[:,-1]!=goal)[0]
        total = len(successfull)
        
        #set up figure
        factor = 2
        fig = plt.figure(figsize=[factor*5,factor*5])
        
        ax = fig.gca()
            
        #plot start and goal state
        start_goal = np.zeros((L,L))
        
        start_goal[0,1] = 1.
        start_goal[-2,-1] = -1.
        colors = ["golden yellow", "light grey", "cobalt"]
        u = sns.heatmap(start_goal, vmin=-1, vmax=1, zorder=2,
                        ax = ax, linewidths = 2, alpha=0.7, cmap=sns.xkcd_palette(colors),
                        xticklabels = False,
                        yticklabels = False,
                        cbar=False)
        ax.invert_yaxis()
        
        #find paths and count them
        n = np.zeros((ns, na))
        
        for i in successfull:
            
            for j in range(T-1):
                d = w.environment.hidden_states[i, j+1] - w.environment.hidden_states[i, j]
                if d not in [1,-1,4,-4,0]:
                    print("ERROR: beaming")
                if d == 1:
                    n[w.environment.hidden_states[i, j],0] +=1
                if d == -1:
                    n[w.environment.hidden_states[i, j]-1,0] +=1 
                if d == 4:
                    n[w.environment.hidden_states[i, j],1] +=1 
                if d == -4:
                    n[w.environment.hidden_states[i, j]-4,1] +=1 
                    
        un = np.zeros((ns, na))
        
        for i in unsuccessfull:
            
            for j in range(T-1):
                d = w.environment.hidden_states[i, j+1] - w.environment.hidden_states[i, j]
                if d not in [1,-1,L,-L,0]:
                    print("ERROR: beaming")
                if d == 1:
                    un[w.environment.hidden_states[i, j],0] +=1
                if d == -1:
                    un[w.environment.hidden_states[i, j]-1,0] +=1 
                if d == 4:
                    un[w.environment.hidden_states[i, j],1] +=1 
                if d == -4:
                    un[w.environment.hidden_states[i, j]-4,1] +=1 
        
        total_num = n.sum() + un.sum()
        
        if np.any(n > 0):
            n /= total_num
            
        if np.any(un > 0):
            un /= total_num
            
        #plotting
        for i in range(ns):
                
            x = [i%L + .5]
            y = [i//L + .5]            

            #plot unsuccessful paths    
            for j in range(2):
                
                if un[i,j]>0.0:
                    if j == 0:
                        xp = x + [x[0] + 1]
                        yp = y + [y[0] + 0]
                    if j == 1:
                        xp = x + [x[0] + 0]
                        yp = y + [y[0] + 1]
                        
                    plt.plot(xp,yp, '-', color='r', linewidth=factor*30*un[i,j],
                              zorder = 9, alpha=0.6)
        
        #set plot title
        plt.title(f"Alpha: {alpha}\n" + 
                  f"World index: {w_ind}\n" +
                  f"Planning success rate: {str(round(100*total/trials))}%", 
                  fontsize=factor*9)           
        
        #plot successful paths on top        
        for i in range(ns):       
            
            x = [i%L + .5]
            y = [i//L + .5]
            
            for j in range(2):
                 
                if n[i,j]>0.0:
                    if j == 0:
                        xp = x + [x[0] + 1]
                        yp = y + [y[0]]
                    if j == 1:
                        xp = x + [x[0] + 0]
                        yp = y + [y[0] + 1]
                    plt.plot(xp,yp, '-', color='c', linewidth=factor*30*n[i,j],
                             zorder = 10, alpha=0.6)   
        
        # save for figure in thesis
        if (alpha == 1 and w_ind == 52) or (alpha == 100 and w_ind == 13):    
            plt.savefig(folder_fig / f"paths_alpha_{alpha}_world_{w_ind}.svg")
            
        plt.show()
        
        prior = w.agent.prior_policies_all[49, :, 0] # prior in last episode
        prior /= prior.sum() # just to be sure
        plt.plot(policy_range, prior)
        plt.title("Policy prior in episode 50\n" + 
                  f"Alpha: {alpha}\n" +
                  f"World index: {w_ind}", 
                  fontsize = 10)
        plt.show()
        
        # unnormalized likelihood of 1st time step in last episode
        # (likelihood is the same for all episodes and time steps anyway)
        lik = w.agent.likelihood[49, 0, :, 0]
        norm_lik = lik / lik.sum()
        plt.plot(policy_range, norm_lik)
        plt.title("Normalized policy likelihood in episode 50\n"+
                  f"Alpha: {alpha}\n" +
                  f"World index: {w_ind}", 
                  fontsize = 10)
        plt.show()       
        
        # extract data for figures in thesis 
        if (alpha == 1 and w_ind == 52) or (alpha == 100 and w_ind == 13):
            df_prior_lik = pd.DataFrame({"policy_ind": policy_range,
                                         "prior_prob": prior,
                                         "norm_lik": norm_lik})
            df_prior_lik.to_csv(folder_data / 
                                f"prior_lik_episode_50_alpha_{alpha}_world_{w_ind}.csv",
                                index = False)
        
        for episode in range(trials):        
            prior = w.agent.prior_policies_all[episode, :, 0]
            prior /= prior.sum() # just to be sure
            lik = w.agent.likelihood[episode, 0, :, 0]
            norm_lik = lik / lik.sum()
            kl_div_array[w_ind, episode] = kl_div(prior, norm_lik)           
   
    perc_succ = 100 * n_succ/(trials*len(worlds)) 
    print(f"Average success rate for alpha = {alpha}: {perc_succ}%")   
    print(f"Average KL divergence for alpha = {alpha}: {kl_div_array.mean()}")

    df_kl_div = df_kl_div.append(pd.DataFrame({"habitual_tendency": 1.0 / alpha,
                                               "episode": np.arange(1, trials + 1),
                                               "kl_div": kl_div_array.mean(0)})) # average over repetitions
    
df_kl_div.to_csv(folder_data /
                 f"average_kl_div_alpha_{'_'.join(map(str, alpha_values))}.csv",
                 index = False)