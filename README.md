# MMLACO-ARRW: Mutual Information-Based Multi-Label Feature Selection

Reference implementation and reproducibility materials for **Mutual Information-Based Multi-Label Feature Selection via Ant Colony Optimization**.

This repository provides the implementation of **MMLACO-ARRW**, extending the MMLACO formulation with **Adaptive Relevance–Redundancy Weighting (ARRW)**.

## Method
- Mutual information measures feature–label relevance and feature–feature redundancy.
- Ant Colony Optimization explores candidate feature subsets.
- Pheromone reinforcement accumulates information from candidate-subset scores.
- ARRW changes the relevance/redundancy balance over iterations.

The manuscript-aligned ARRW schedule starts at `(alpha, gamma) = (0.9, 0.1)` and ends at `(0.1, 0.9)`.

## Manuscript-aligned settings
| Setting | Value |
|---|---:|
| ACO iterations | 50 |
| Pheromone evaporation rho | 0.1 |
| Initial pheromone | 0.1 |
| q0 | 0.6 |
| beta | 1 |
| alpha_max | 0.9 |
| alpha_min | 0.1 |
| Selected features | 40 (8 for Flags) |
| Number of ants | d if d < 100; otherwise 100 |
| ML-kNN k | 10 |
| Train/test split | 70/30 |
| Main repeated experiments | 10 independent runs |

## Structure
`src/` contains the algorithm, `scripts/` contains command-line experiment runners, `configs/` contains example configuration, `tests/` contains basic tests, and `docs/` contains reproducibility and dataset notes.

## Installation
```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducibility
Third-party datasets are **not redistributed** in this repository. Prepare the dataset locally according to `docs/DATA.md` and use the supplied scripts/configuration. Do not interpret the repository as a replacement for the manuscript's complete numerical tables; regenerate results from the final dataset files and preprocessing used for the paper.

## Example
```bash
python scripts/run_repeated_experiments.py --data data/scene.csv --features 294 --labels 6 --selected 40 --runs 10 --start-seed 0 --mode arrw --mlknn-k 10 --output results/scene_arrw_runs.csv
```

## Citation
Please cite the associated article: **Mutual Information-Based Multi-Label Feature Selection via Ant Colony Optimization**.

## License
Source code: MIT License. Third-party datasets and dependencies remain subject to their own terms.
