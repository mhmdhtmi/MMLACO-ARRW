# Reproducibility protocol

Use Python 3.10 with the pinned dependencies. Prepare the same benchmark files and preprocessing used for the manuscript.

For each dataset:
1. compute feature-feature and feature-label mutual information;
2. construct the inverse-MI transition heuristic;
3. initialize pheromone to 0.1;
4. run 50 ACO iterations;
5. use d ants when d < 100, otherwise 100 ants;
6. use q0 = 0.6, beta = 1, rho = 0.1;
7. construct candidate subsets using the required feature budget;
8. evaluate subsets with the relevance-redundancy criterion;
9. reinforce pheromone from candidate scores;
10. rank features by final pheromone.

ARRW uses alpha_max=0.9, alpha_min=0.1 and gamma_t=1-alpha_t. The first and final iterations therefore use (0.9,0.1) and (0.1,0.9), respectively.

Evaluation uses ML-kNN with k=10 and a 70/30 train-test split, with ten independent runs. Record the seed for every run.

For the fixed-vs-ARRW ablation, keep dataset, feature budget, ACO settings, train/test protocol, and paired seeds identical; change only the weighting mode.

Archive seeds, dataset dimensions, selected feature indices, pheromone values, runtime, metrics, and environment/package versions.

The repository contains the supplied proposed-method implementation and ARRW ablation. It does not include every literature baseline or third-party dataset.
