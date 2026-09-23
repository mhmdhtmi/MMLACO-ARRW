"""MMLACO-ARRW reference implementation aligned with the manuscript protocol."""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

def compute_mutual_information(data: pd.DataFrame, labels: pd.DataFrame):
    lsize, fsize = labels.shape[1], data.shape[1]
    mif = np.zeros((fsize, fsize), dtype=float)
    mil = np.zeros((fsize, lsize), dtype=float)
    for i in range(fsize):
        for j in range(fsize):
            mif[i, j] = normalized_mutual_info_score(data.iloc[:, i].values, data.iloc[:, j].values)
    for i in range(fsize):
        for j in range(lsize):
            mil[i, j] = normalized_mutual_info_score(data.iloc[:, i].values, labels.iloc[:, j].values)
    return mif, mil

def create_graph(mif):
    return 1.0 / np.maximum(mif, 1e-12)

def relevance(mil):
    return np.sum(mil, axis=1)

def redundancy(feature, selected, mif):
    if len(selected) == 0:
        return 0.0
    return float(np.sum(mif[feature, selected]))

def alpha_gamma(iteration, n_iter, alpha_max=0.9, alpha_min=0.1, mode="arrw"):
    if mode == "fixed":
        return alpha_max, 1.0-alpha_max
    frac = iteration / float(max(n_iter-1,1))
    alpha_t = alpha_max - (alpha_max-alpha_min)*frac
    gamma_t = 1.0-alpha_t
    return alpha_t, gamma_t

def mmlaco(data, mif, mil, *, beta=1, rho=0.1, n_iter=50, n_ants=100,
           n_selected=40, q0=0.6, initial_pheromone=0.1,
           mode="arrw", random_state=0):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    fsize=data.shape[1]
    graph=create_graph(mif)
    rel=relevance(mil)
    tau=np.ones(fsize,dtype=float)*initial_pheromone
    history=[]
    for iteration in range(n_iter):
        alpha_t,gamma_t=alpha_gamma(iteration,n_iter,mode=mode)
        paths=[[] for _ in range(n_ants)]
        for k in range(n_ants):
            first=np.random.randint(0,fsize)
            paths[k].append(first)
            for _ in range(n_selected-1):
                candidates=[i for i in range(fsize) if i not in set(paths[k])]
                heuristic=np.array([max(0.0,tau[d]*(graph[paths[k][-1],d]**beta)) for d in candidates])
                q=random.random()
                if q<=q0:
                    nxt=candidates[int(np.argmax(heuristic))]
                else:
                    total=np.sum(heuristic)
                    if total<=0 or not np.isfinite(total):
                        nxt=candidates[int(np.random.randint(len(candidates)))]
                    else:
                        nxt=candidates[int(np.random.choice(len(candidates),p=heuristic/total))]
                paths[k].append(nxt)
        delta_ant=np.array([sum(alpha_t*rel[f]-gamma_t*redundancy(f,path,mif) for f in path) for path in paths])
        delta_feature=np.zeros(fsize,dtype=float)
        for ant,path in enumerate(paths):
            for feature in path:
                delta_feature[feature]+=delta_ant[ant]
        tau=(1.0-rho)*tau+delta_feature
        history.append({"iteration":iteration+1,"alpha":alpha_t,"gamma":gamma_t,
                        "best_ant_score":float(np.max(delta_ant)),
                        "mean_ant_score":float(np.mean(delta_ant)),
                        "max_pheromone":float(np.max(tau)),
                        "mean_pheromone":float(np.mean(tau))})
    selected=np.argsort(tau)[::-1][:n_selected].astype(int)
    return selected,tau,pd.DataFrame(history)
