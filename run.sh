#!/bin/bash
# End-to-end Phase 2 runner.
# Reproduces Stage 1, Stage 2, SHAP, case studies, and DDID external validation.
# All experiments use a fixed random seed of 42 (see config.py).
set -e

# ------------------------------------------------------------------
# 0. Install dependencies
# ------------------------------------------------------------------
pip install -r code/phase2/requirements.txt -q

# ------------------------------------------------------------------
# 1. Stage 1 — compound-level toxicity prediction (Sec 2.3 / 3.1)
# ------------------------------------------------------------------
echo "=== 01. Murcko scaffold 80/20 split ==="
python code/phase2/01_scaffold_split.py

echo "=== 02. Traditional ML training (5 algos x 26 endpoints, Optuna, ~17 h) ==="
python code/phase2/02_train_traditional_ml.py

echo "=== 03. AttentiveFP GNN (~711 min on GPU) ==="
python code/phase2/03_train_attentivefp.py

echo "=== 03b. NB ensemble + Y-randomization ==="
python code/phase2/03_nb_ensemble_yrand.py

# Optional improvement experiments (Sec 3.1.4)
# python code/phase2/02b_train_improved.py     # SMOTE on CTD CYP
# python code/phase2/02c_train_improved.py     # MI Top-500 feature selection

# ------------------------------------------------------------------
# 2. Stage 2 — herb-drug interaction risk assessment (Sec 2.4 / 3.2)
# ------------------------------------------------------------------
echo "=== 05. Stage 2 LightGBM + GroupKFold + ablation ==="
python code/phase2/05_stage2_ddi.py

# ------------------------------------------------------------------
# 3. Interpretability + validation (Sec 2.5, 2.6 / 3.3, 3.4, 3.5)
# ------------------------------------------------------------------
echo "=== 06. SHAP three-layer interpretability ==="
python code/phase2/06_shap_interpretability.py

echo "=== 07. Clinical case validation ==="
python code/phase2/07_case_study.py

echo "=== 08. DDID external validation ==="
python code/phase2/08_ddid_external_validation.py

echo "=== make_figures ==="
python code/phase2/make_figures.py

echo "=========================================="
echo "Phase 2 pipeline complete."
echo "=========================================="
