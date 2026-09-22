"""MMLACO-ARRW reference implementation.

Core feature-selection routines used by the reproducibility materials.
"""
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

def alpha_gamma(iteration, n_iter, alpha_max=0.9, alpha_min=0.1,
                gamma_min=0.1, gamma_max=0.9, mode="arrw"):
    if mode == "fixed":
        return alpha_max, gamma_min
    frac = iteration / float(max(n_iter - 1, 1))
    alpha_t = alpha_max - (alpha_max - alpha_min) * frac
    gamma_t = gamma_min + (gamma_max - gamma_min) * frac
    return alpha_t, gamma_t

def _delta_tau(feature, selected, rel, mif, iteration, n_iter, mode,
               alpha_max=0.9, alpha_min=0.1):
    alpha_t, gamma_t = alpha_gamma(
        iteration, n_iter, alpha_max=alpha_max, alpha_min=alpha_min,
        gamma_min=1-alpha_max, gamma_max=1-alpha_min, mode=mode)
    return alpha_t * rel[feature] - gamma_t * redundancy(feature, selected, mif)

def _candidate_score(path, rel, mif, iteration, n_iter, mode):
    return sum(_delta_tau(f, path, rel, mif, iteration, n_iter, mode) for f in path)

def _remaining(fsize, selected):
    used = set(selected)
    return [i for i in range(fsize) if i not in used]

def _roulette(probabilities, rng):
    total = np.sum(probabilities)
    if total <= 0 or not np.isfinite(total):
        return int(rng.integers(len(probabilities)))
    return int(rng.choice(len(probabilities), p=probabilities / total))

def _choose_next(q, q0, candidates, tau, graph, current, beta, rng):
    heuristic = np.array([max(0.0, tau[d] * (graph[current, d] ** beta)) for d in candidates])
    if q <= q0:
        return candidates[int(np.argmax(heuristic))]
    return candidates[_roulette(heuristic, rng)]

def mmlaco(data, mif, mil, *, beta=1, rho=0.1, n_iter=50, n_ants=100,
           n_selected=40, q0=0.6, initial_pheromone=0.1, mode="arrw",
           random_state=0):
    rng = np.random.default_rng(random_state)
    py_rng = random.Random(random_state)
    fsize = data.shape[1]
    graph, rel = create_graph(mif), relevance(mil)
    tau = np.ones(fsize, dtype=float) * initial_pheromone
    history = []
    for iteration in range(n_iter):
        alpha_t, gamma_t = alpha_gamma(iteration, n_iter, mode=mode)
        paths = []
        for _ in range(n_ants):
            path = [int(rng.integers(0, fsize))]
            for _ in range(n_selected - 1):
                candidates = _remaining(fsize, path)
                path.append(_choose_next(py_rng.random(), q0, candidates, tau, graph, path[-1], beta, rng))
            paths.append(path)
        delta_ant = np.array([_candidate_score(path, rel, mif, iteration, n_iter, mode) for path in paths])
        delta_feature = np.zeros(fsize, dtype=float)
        for ant, path in enumerate(paths):
            delta_feature[path] += delta_ant[ant]
        tau = (1.0 - rho) * tau + delta_feature
        history.append({"iteration": iteration + 1, "alpha": alpha_t, "gamma": gamma_t,
                        "best_ant_score": float(np.max(delta_ant)),
                        "mean_ant_score": float(np.mean(delta_ant)),
                        "max_pheromone": float(np.max(tau)),
                        "mean_pheromone": float(np.mean(tau))})
    return np.argsort(tau)[::-1][:n_selected].astype(int), tau, pd.DataFrame(history)
