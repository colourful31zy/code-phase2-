# CPM–Western Drug Co-administration: Two-Stage Interpretable ML Framework

Supporting code for the manuscript:

> **An interpretable two-stage machine learning framework for predicting herb–drug interaction safety in Chinese patent medicines**

This repository contains the Phase 2 modeling pipeline: compound-level multi-endpoint toxicity prediction (Stage 1) and herb–drug (compound–drug) interaction safety risk assessment (Stage 2), together with SHAP interpretability analysis, clinical case validation, and independent DDID external validation.

## Repository contents

```
code/phase2/
├── config.py                       # Global config: SEED=42, Optuna trials, 5-fold CV, AUPRC metric
├── utils.py                        # Data loading, metrics, model factory, NB ensemble helpers
├── 01_scaffold_split.py            # Murcko-scaffold 80/20 split (zero overlap, Tanimoto check)
├── 02_train_traditional_ml.py      # 5 algorithms × 26 endpoints, Optuna, threshold optimization
├── 02b_train_improved.py           # Plan D v1: SMOTE on CTD CYP group
├── 02c_train_improved.py           # Plan D v2: MI Top-500 feature selection
├── 03_train_attentivefp.py         # AttentiveFP multi-task GNN
├── 03_nb_ensemble_yrand.py         # NB ensemble + Y-randomization test
├── 03b_nb_ensemble_top3.py         # NB Ensemble v2 (RF/XGB/LGB only)
├── 04_threshold_optimization.py    # Per-endpoint threshold scan (maximize F1)
├── 04_generate_comparison.py       # Comparison: traditional ML × NB Ensemble × AttentiveFP
├── 05_stage2_ddi.py                # Stage 2 DDI risk model + GroupKFold + ablation
├── 06_shap_interpretability.py     # SHAP three-layer analysis (TreeExplainer / KernelExplainer)
├── 07_case_study.py                # Four clinical cases (3 positive + 1 negative control)
├── 08_ddid_external_validation.py  # Independent DDID external validation (Sec 3.5)
├── 09_cold_start_validation.py     # Additional cold-start robustness check
├── make_figures.py                 # Generates Figures 1–5 from result tables
├── ob_deploy.py                    # Deployment helper (OpenBayes)
├── requirements.txt
└── run.sh                          # End-to-end runner
```

## Data sources (all public)

The raw data are **not** redistributed in this repository. Download them directly from the official sources:

| Database | Used for | URL |
|---|---|---|
| DrugBank 5.0 | Compound space + Stage 2 DDI positives | https://go.drugbank.com/ |
| TCMBank | CPM ingredients / prescriptions | https://www.tcmbank.com/ |
| Tox21 | 12 in-vitro qHTS toxicity endpoints | https://ntp.niehs.nih.gov/whatwestudy/tox21 |
| CTD (Comparative Toxicogenomics Database) | 14 literature-mined toxicity endpoints | http://ctdbase.org/ |
| BioSNAP | DDI supplement | https://snap.stanford.edu/biodata/datasets/10015/10015-ChChSeDecagon.html |
| DDID (Herb–Drug Interaction Database) | Independent external validation | http://www.ddidatabase.com/ |

Phase 1 (data preparation: SMILES unification, deduplication to 47,724 unique compounds, 1,210-dim feature matrix) is described in the manuscript (Section 2.1). Its scripts live under `data/02_processed/phase1/` and are run once to produce the feature matrix and the Tox21/CTD label files consumed by Phase 2.

### Where to put the data

All paths are resolved relative to the project root via `config.py` (`PROJECT_ROOT = code/phase2/../..`). To reproduce Phase 2, place the Phase 1 outputs under:

```
<project_root>/data/02_processed/phase2/split/
├── train_features.csv          # 38,179 × 1,210
├── test_features.csv           # 9,545 × 1,210
├── tox21_train_labels.csv      # 7,831 × 12 (tox21 subset of train)
├── tox21_test_labels.csv
├── ctd_train_labels.csv        # 3,675 × 14 (ctd subset of train)
├── ctd_test_labels.csv
└── split_meta.json             # feature column list + split stats
```

Generated models, result tables, and figures are written back under `data/02_processed/phase2/models/` and `data/02_processed/phase2/results/`.

## Environment

- **Python**: 3.10+ (developed on Python 3.11, Windows)
- **OS**: Windows / Linux (paths use `os.path.join`, no hard-coded separators)
- **GPU**: one NVIDIA GPU for `03_train_attentivefp.py` (711 min); all other scripts run on CPU

Install dependencies:

```bash
pip install -r code/phase2/requirements.txt
```

> RDKit and PyTorch Geometric have platform-specific installation steps. If `pip install rdkit` fails, use `conda install -c conda-forge rdkit`. For PyTorch Geometric, follow the wheel index at https://data.pyg.org/wheel/ matching your torch + CUDA version.

## Reproduction

All experiments use a fixed random seed of **42** (`config.py`). Reproduce the full pipeline end-to-end:

```bash
bash code/phase2/run.sh
```

Or run the stages individually (in this order):

```bash
python code/phase2/01_scaffold_split.py            # produces train/test split files
python code/phase2/02_train_traditional_ml.py      # ~17 h on CPU (120 models)
python code/phase2/03_train_attentivefp.py         # ~711 min on GPU
python code/phase2/03_nb_ensemble_yrand.py         # NB ensemble + Y-randomization
python code/phase2/05_stage2_ddi.py                # Stage 2 GroupKFold + ablation
python code/phase2/06_shap_interpretability.py     # SHAP three-layer analysis
python code/phase2/07_case_study.py                # Four clinical case studies
python code/phase2/08_ddid_external_validation.py  # DDID external validation
python code/phase2/make_figures.py                 # regenerate Figures 1–5
```

The improvement experiments (Section 3.1.4) are optional:

```bash
python code/phase2/02b_train_improved.py           # SMOTE on CTD CYP
python code/phase2/02c_train_improved.py           # MI Top-500 feature selection
```

## Expected results

Running the pipeline reproduces the key numbers reported in the manuscript:

| Result | Value |
|---|---|
| Compounds / endpoints | 47,724 × 21 |
| Train / test split (Murcko scaffold, 80/20) | 38,179 / 9,545 (zero scaffold overlap) |
| Stage 1 RF mean AUC on Tox21 | 0.768 |
| Stage 1 RF best on 13 / 21 endpoints | (XGBoost 5, LightGBM 3) |
| Stage 1 No_Stage1 vs Full (LightGBM) | 0.9819 vs 0.9853 (ΔAUC +0.0034) |
| Stage 2 LightGBM 5-fold GroupKFold AUC | 0.9853 ± 0.0014 |
| Stage 2 No_MolFeat (42-dim toxicity only) AUC | 0.9827 |
| DDID external validation AUC | 0.745 (p < 1e-300) |

## Citation

If you use this code, please cite the published article (Frontiers in Pharmacology, Ethnopharmacology).

DOI: *to be added upon publication.*

## License

The code in this repository is released under the MIT License. Underlying data are governed by the licenses of their respective public databases (see the Data sources table above).
