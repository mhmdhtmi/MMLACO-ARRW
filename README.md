# MMLACO-ARRW: Mutual Information-Based Multi-Label Feature Selection

Reference implementation and reproducibility materials for **Mutual Information-Based Multi-Label Feature Selection via Ant Colony Optimization**.

This repository provides the manuscript-aligned implementation of **MMLACO-ARRW**, extending the MMLACO formulation with **Adaptive Relevance–Redundancy Weighting (ARRW)**.

## Method
- Mutual information measures feature–label relevance and feature–feature redundancy.
- Ant Colony Optimization explores candidate feature subsets.
- Pheromone reinforcement accumulates information from candidate-subset scores.
- ARRW changes the relevance/redundancy balance over iterations.
- The ARRW schedule starts at **(alpha, gamma) = (0.9, 0.1)** and ends at **(0.1, 0.9)**.

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

## Reproducibility
The repository is designed to facilitate **faithful reproduction of the reported experimental protocol**. It provides the core implementation, the fixed-weight ablation variant, a manuscript-aligned reproducibility notebook, configuration files, repeated-run scripts, tests, and documentation.

The stochastic ACO procedure uses explicit random seeds. For the ARRW ablation, the fixed-weight and ARRW variants use the same experimental protocol and paired seeds, with the weighting strategy as the controlled difference.

The repository does **not** claim that the manuscript's numerical tables are automatically reproduced from a single command. Reproduction of the reported tables requires the same benchmark files, preprocessing, random seeds, and repeated-run protocol used in the manuscript.

## Data
Third-party benchmark datasets are not redistributed. The manuscript uses publicly available MULAN multi-label datasets. Dataset provenance and the expected local input format are documented in [docs/DATA.md](docs/DATA.md).

## Structure
- `src/`: MMLACO-ARRW implementation
- `scripts/`: single-run and repeated-run experiment scripts
- `notebooks/`: reproducibility notebook
- `configs/`: manuscript-aligned configuration
- `tests/`: basic validation tests
- `docs/`: data and reproducibility documentation
- `results/`: location for locally generated outputs

## Installation
```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Example
```bash
python scripts/run_repeated_experiments.py --data data/scene.csv --features 294 --labels 6 --selected 40 --runs 10 --start-seed 0 --mode arrw --mlknn-k 10 --output results/scene_arrw_runs.csv
```

## Citation
Please cite the associated article: **Mutual Information-Based Multi-Label Feature Selection via Ant Colony Optimization**.

## License
Source code: MIT License. Third-party datasets and dependencies remain subject to their own terms.
