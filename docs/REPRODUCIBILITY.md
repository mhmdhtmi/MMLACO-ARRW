# Reproducibility protocol

The repository is aligned with the final manuscript protocol.

1. Prepare the same benchmark files and preprocessing used for the manuscript.
2. Compute feature-feature and feature-label mutual information.
3. Construct the inverse-MI transition heuristic.
4. Initialize pheromone to 0.1.
5. Run 50 ACO iterations.
6. Use d ants when d < 100 and 100 ants otherwise.
7. Use q0 = 0.6, beta = 1, rho = 0.1.
8. Construct candidate subsets using the required feature budget (40, or 8 for Flags).
9. Evaluate each subset using the relevance-redundancy criterion.
10. Reinforce and update pheromone values.
11. Rank features by final pheromone.

ARRW uses alpha_max=0.9, alpha_min=0.1 and gamma_t=1-alpha_t. The first and final iterations therefore use (0.9,0.1) and (0.1,0.9).

Evaluation uses ML-kNN with k=10 and a 70/30 train-test split. The manuscript reports ten independent runs. Record the seed for every run.

For the fixed-vs-ARRW ablation, keep dataset, feature budget, ACO settings, train/test protocol, and paired seeds identical; change only the weighting mode.

Third-party benchmark datasets are not redistributed. Their provenance and expected input format are documented in docs/DATA.md.

The repository provides the implementation and protocol needed to facilitate faithful reproduction; it does not fabricate or embed the manuscript's numerical tables.
