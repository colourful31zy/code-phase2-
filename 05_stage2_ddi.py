"""
05_stage2_ddi.py — Stage 2: Drug-Pair Interaction Safety Risk Assessment

Pipeline:
  1. Load best per-endpoint model config → (endpoint → best algorithm) mapping
  2. Re-train Stage 1 models (RF/XGBoost/LightGBM) on full train set,
     predict on ALL 47,724 compounds → 21-dim toxicity profile per compound
  3. Build DDI pairs from DrugBank (positive) + random negative sampling
  4. Feature engineering:
     ToxA(21) | ToxB(21) | |MolFeat_A - MolFeat_B|(1210) | MolFeat_A * MolFeat_B(1210)
     = 2462 features per pair
  5. GroupKFold CV (5-fold, grouped by compound) × 3 algorithms (RF/XGBoost/LightGBM),
     Optuna 20 trials
  6. Ablation: Full / No_Stage1 / No_MolFeat
  7. Save results

Output:
  - results/evaluation/stage1_toxicity_profiles.csv  — Stage 1 predictions for all compounds
  - results/evaluation/stage2_results.csv            — per-algorithm CV metrics
  - results/evaluation/ablation_results.csv          — ablation experiments
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Constants ────────────────────────────────────────────────────────
SEED = 42
N_TRIALS = 20            # Optuna trials for Stage 2
N_FOLDS = 5
MAX_POS_PAIRS = 50000    # cap to keep memory manageable
NEG_POS_RATIO = 1.0      # 1:1 pos:neg
MAX_NEG_ATTEMPTS = 2000000

# Paths
PHASE1 = config.PHASE1_DIR
SPLIT_DIR = config.SPLIT_DIR
RESULTS_DIR = config.STAGE2_RESULTS
MODEL_DIR = config.STAGE2_MODELS
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(PHASE1, "step3_features", "all_features.csv")
DRUGBANK_DDI_FILE = os.path.join(PHASE1, "step4_ddi_integrated", "ddi_drugbank.csv")
BEST_ENDPOINT_FILE = os.path.join(config.TRADITIONAL_ML_RESULTS, "best_per_endpoint.csv")
META_FILE = config.META_FILE

# 26 endpoints, 21 valid (skip Tox21_10/11 zero-pos, SCN5A/KCNQ1/KCNE1 <1%)
TOX21_EPS = [f"Tox21_{i}" for i in range(12)]
CTD_GENE_EPS = ["gene_CYP1A2", "gene_CYP2C9", "gene_CYP2C19",
                "gene_CYP2D6", "gene_CYP3A4", "gene_CYP3A5",
                "gene_KCNH2", "gene_SCN5A", "gene_KCNQ1", "gene_KCNE1"]
CTD_TOX_EPS = ["tox_hepatotoxicity", "tox_cardiotoxicity",
               "tox_nephrotoxicity", "tox_neurotoxicity"]
ALL_EPS = TOX21_EPS + CTD_GENE_EPS + CTD_TOX_EPS
SKIP_EPS = {"Tox21_10", "Tox21_11", "gene_SCN5A", "gene_KCNQ1", "gene_KCNE1"}
VALID_EPS = [ep for ep in ALL_EPS if ep not in SKIP_EPS]  # 21

EP_DATASET = {ep: "tox21" for ep in TOX21_EPS}
for ep in CTD_GENE_EPS + CTD_TOX_EPS:
    EP_DATASET[ep] = "ctd"


# ═══════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════

def log(msg):
    print(f"  {msg}")


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Load per-endpoint best-algorithm config
# ═══════════════════════════════════════════════════════════════════════

def load_endpoint_config():
    df = pd.read_csv(BEST_ENDPOINT_FILE)
    cfg = {}
    for _, r in df.iterrows():
        cfg[r["endpoint"]] = {"algorithm": r["algorithm"], "dataset": r["dataset"]}
    log(f"Loaded {len(cfg)} endpoint configs")
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Generate Stage 1 toxicity profiles for all 47,724 compounds
# ═══════════════════════════════════════════════════════════════════════

def build_model(algorithm, n_pos, n_neg):
    if algorithm == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif algorithm == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(scale_pos_weight=n_neg / max(n_pos, 1),
                             eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=SEED)
    elif algorithm == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(scale_pos_weight=n_neg / max(n_pos, 1),
                              verbose=-1, n_jobs=-1, random_state=SEED)
    raise ValueError(f"Unsupported: {algorithm}")


def generate_toxicity_profiles(feature_cols, endpoint_cfg):
    """
    Train 21 single-endpoint models on the full training set,
    predict probabilities for ALL 47,724 compounds.

    Tox21 matched by SMILES (exact).
    CTD matched by canonicalized SMILES (RDKit) to handle format inconsistencies.

    Returns (tox_df, tox_cols) where tox_df has columns [ID] + 21 tox endpoints.
    """
    print_header("Step 2: Generating Stage 1 toxicity profiles")

    # NOTE: all_features.csv has only 1044 cols (19 desc + ID + 1024 Morgan — NO MACCS).
    # split_meta.json feature_columns lists 1210 cols including MACCS, so we must load
    # from split files (train_features.csv + test_features.csv, which have all 1210 features).
    log("Loading features from split files (train + test, covers all 47724 compounds)...")
    train_f = pd.read_csv(os.path.join(SPLIT_DIR, "train_features.csv"),
                          usecols=["ID", "SMILES"] + feature_cols)
    test_f = pd.read_csv(os.path.join(SPLIT_DIR, "test_features.csv"),
                         usecols=["ID", "SMILES"] + feature_cols)
    df_all = pd.concat([train_f, test_f], ignore_index=True)
    all_ids = df_all["ID"].values
    X_all = df_all[feature_cols].values.astype(np.float32)
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
    n_all = len(all_ids)
    log(f"  {n_all} compounds x {len(feature_cols)} features")

    # Build SMILES->ID mapping
    smi_map = df_all[["SMILES", "ID"]].drop_duplicates(subset="SMILES")
    smiles_to_id = dict(zip(smi_map["SMILES"], smi_map["ID"]))
    smiles_to_idx = {smi: id_to_idx[cid] for smi, cid in smiles_to_id.items()
                     if cid in id_to_idx}
    log(f"  Mapped {len(smiles_to_idx)} SMILES -> IDs")

    # Load labels
    tox21_train = pd.read_csv(os.path.join(SPLIT_DIR, "tox21_train_labels.csv"))
    ctd_train = pd.read_csv(os.path.join(SPLIT_DIR, "ctd_train_labels.csv"))

    # Build canonical SMILES -> ID map for CTD matching
    from rdkit import Chem
    canon_to_id = {}
    for orig_smi, feat_id in smiles_to_id.items():
        mol = Chem.MolFromSmiles(orig_smi)
        if mol:
            canon_to_id[Chem.MolToSmiles(mol)] = feat_id
    log(f"  Built canonical SMILES map: {len(canon_to_id)} entries")

    # Predict for each endpoint
    tox_matrix = np.zeros((n_all, len(VALID_EPS)), dtype=np.float32)
    trained, skipped = 0, 0

    for j, endpoint in enumerate(VALID_EPS):
        ds = EP_DATASET[endpoint]

        if ds == "tox21":
            tr = tox21_train[["SMILES", endpoint]].dropna()
            train_idx, train_lbl = [], []
            for _, row in tr.iterrows():
                idx = smiles_to_idx.get(row["SMILES"])
                if idx is not None:
                    train_idx.append(idx)
                    train_lbl.append(row[endpoint])
        else:
            tr = ctd_train[["smiles", endpoint]].dropna()
            train_idx, train_lbl = [], []
            for _, row in tr.iterrows():
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol:
                    cid = canon_to_id.get(Chem.MolToSmiles(mol))
                    if cid and cid in id_to_idx:
                        train_idx.append(id_to_idx[cid])
                        train_lbl.append(row[endpoint])

        train_idx = np.array(train_idx)
        train_lbl = np.array(train_lbl, dtype=np.float32)

        if len(train_idx) < 10:
            tox_matrix[:, j] = 0.0
            skipped += 1
            log(f"  [{j+1}/{len(VALID_EPS)}] {endpoint}: skipped (<10 samples)")
            continue

        n_pos = int(np.sum(train_lbl == 1))
        n_neg = int(np.sum(train_lbl == 0))
        algorithm = endpoint_cfg.get(endpoint, {}).get("algorithm", "RF")

        model = build_model(algorithm, n_pos, n_neg)
        model.fit(X_all[train_idx], train_lbl)
        tox_matrix[:, j] = model.predict_proba(X_all)[:, 1]
        trained += 1
        log(f"  [{j+1}/{len(VALID_EPS)}] {endpoint} ({algorithm}): "
            f"{len(train_idx)} train, {n_pos}P/{n_neg}N")

    log(f"Done: {trained} trained, {skipped} skipped")

    tox_cols = [f"tox_{ep}" for ep in VALID_EPS]
    tox_df = pd.DataFrame(tox_matrix, columns=tox_cols)
    tox_df.insert(0, "ID", all_ids)
    out_path = os.path.join(RESULTS_DIR, "stage1_toxicity_profiles.csv")
    tox_df.to_csv(out_path, index=False)
    log(f"Saved: {out_path} ({tox_df.shape})")
    return tox_df, tox_cols


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Build DDI pairs (DrugBank positive + random negative)
# ═══════════════════════════════════════════════════════════════════════

def build_ddi_pairs():
    """
    Positive: DrugBank DDI pairs where both drugs are in our feature space (capped).
    Negative: random non-interacting pairs (1:1 ratio).
    """
    print_header("Step 3: Building DDI pairs")

    df_feat = pd.read_csv(FEATURES_FILE, usecols=["ID"])
    all_ids = df_feat["ID"].values
    all_id_set = set(all_ids)
    n_compounds = len(all_ids)
    log(f"Feature space: {n_compounds} compounds")

    log("Loading DrugBank DDI...")
    ddi = pd.read_csv(DRUGBANK_DDI_FILE)
    log(f"Raw entries: {len(ddi)}")

    in_a = ddi["Drug1_ID"].isin(all_id_set)
    in_b = ddi["Drug2_ID"].isin(all_id_set)
    valid = ddi[in_a & in_b].copy()
    log(f"Both drugs in feature space: {len(valid)}")

    # Dedup (A,B) == (B,A)
    valid["_pair_key"] = valid.apply(
        lambda r: tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])), axis=1)
    valid = valid.drop_duplicates(subset="_pair_key").reset_index(drop=True)
    log(f"After dedup: {len(valid)} unique pairs")

    n_pos = min(len(valid), MAX_POS_PAIRS)
    if len(valid) > MAX_POS_PAIRS:
        valid = valid.sample(n=MAX_POS_PAIRS, random_state=SEED)
    log(f"Positive pairs: {n_pos}")

    pos_set = set()
    for _, r in valid.iterrows():
        pos_set.add(tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])))

    pos_df = pd.DataFrame({
        "DrugA_ID": valid["Drug1_ID"].values,
        "DrugB_ID": valid["Drug2_ID"].values,
        "label": np.ones(n_pos, dtype=np.int32),
    })
    # Ensure DrugA < DrugB
    for col_a, col_b in [("DrugA_ID", "DrugB_ID")]:
        a, b = pos_df[col_a].values, pos_df[col_b].values
        swap = a > b
        if swap.any():
            pos_df[col_a] = np.where(swap, b, a)
            pos_df[col_b] = np.where(swap, a, b)

    # Sample negative pairs
    n_neg = int(n_pos * NEG_POS_RATIO)
    log(f"Sampling {n_neg} negative pairs...")
    np.random.seed(SEED)
    neg_list = []
    attempt = 0
    while len(neg_list) < n_neg and attempt < MAX_NEG_ATTEMPTS:
        batch = min(MAX_NEG_ATTEMPTS, (n_neg - len(neg_list)) * 10)
        idx_a = np.random.randint(0, n_compounds, batch)
        idx_b = np.random.randint(0, n_compounds, batch)
        for a, b in zip(idx_a, idx_b):
            if a == b:
                continue
            key = tuple(sorted([all_ids[a], all_ids[b]]))
            if key not in pos_set:
                neg_list.append(key)
                attempt += 1
                if len(neg_list) >= n_neg:
                    break

    neg_arr = np.array(neg_list[:n_neg])
    neg_df = pd.DataFrame({
        "DrugA_ID": neg_arr[:, 0],
        "DrugB_ID": neg_arr[:, 1],
        "label": np.zeros(n_neg, dtype=np.int32),
    })
    a, b = neg_df["DrugA_ID"].values, neg_df["DrugB_ID"].values
    swap = a > b
    if swap.any():
        neg_df["DrugA_ID"] = np.where(swap, b, a)
        neg_df["DrugB_ID"] = np.where(swap, a, b)

    df_pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    df_pairs = df_pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
    log(f"Total: {len(df_pairs)} pairs ({n_pos} pos, {n_neg} neg)")
    return df_pairs


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Build DDI feature matrix
# ═══════════════════════════════════════════════════════════════════════

def build_feature_matrix(df_pairs, tox_df, tox_cols, feature_cols):
    """
    Per pair (A, B): ToxA(21) | ToxB(21) | |Feat_A-Feat_B|(1210) | Feat_A*Feat_B(1210)
    Vectorized via two separate merges (A and B), then concatenated.
    """
    print_header("Step 4: Building DDI feature matrix (vectorized)")

    log("Loading molecular features from split files...")
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train_features.csv"),
                     usecols=["ID"] + feature_cols)
    te = pd.read_csv(os.path.join(SPLIT_DIR, "test_features.csv"),
                     usecols=["ID"] + feature_cols)
    df_mol = pd.concat([tr, te], ignore_index=True)
    tox_flat = tox_df.reset_index()

    n_pairs = len(df_pairs)
    y = df_pairs["label"].values.astype(np.float32)
    log(f"Processing {n_pairs} DDI pairs...")

    # DrugA: merge mol features + toxicity
    m_a = df_pairs.merge(df_mol, left_on="DrugA_ID", right_on="ID", how="left")
    m_a = m_a.merge(tox_flat, on="ID", how="left")

    # DrugB: merge mol features + toxicity
    m_b = df_pairs.merge(df_mol, left_on="DrugB_ID", right_on="ID", how="left")
    m_b = m_b.merge(tox_flat, on="ID", how="left")

    mol_a = m_a[feature_cols].fillna(0).values.astype(np.float32)
    mol_b = m_b[feature_cols].fillna(0).values.astype(np.float32)
    tox_a = m_a[tox_cols].fillna(0).values.astype(np.float32)
    tox_b = m_b[tox_cols].fillna(0).values.astype(np.float32)

    n_tox = len(tox_cols); n_mol = len(feature_cols)
    n_features = 2 * n_tox + 2 * n_mol
    log(f"Assembling {n_features}-dim matrix...")

    X = np.concatenate([
        tox_a, tox_b,
        np.abs(mol_a - mol_b).astype(np.float32),
        (mol_a * mol_b).astype(np.float32),
    ], axis=1)

    groups = df_pairs["DrugA_ID"].values.tolist()
    fnames = ([f"ToxA_{c}" for c in tox_cols] +
              [f"ToxB_{c}" for c in tox_cols] +
              [f"Diff_{c}" for c in feature_cols] +
              [f"Prod_{c}" for c in feature_cols])

    log(f"X: {X.shape}, pos rate: {y.mean():.4f}")
    return X, y, groups, fnames


# ═══════════════════════════════════════════════════════════════════════
# Step 5: GroupKFold + Optuna training
# ═══════════════════════════════════════════════════════════════════════

def train_stage2_model(X_train, y_train, X_val, y_val, algorithm):
    """Train one fold with Optuna tuning."""
    import optuna

    def objective(trial):
        if algorithm == "RF":
            from sklearn.ensemble import RandomForestClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = RandomForestClassifier(**params, class_weight="balanced",
                                           random_state=SEED, n_jobs=-1)
        elif algorithm == "XGBoost":
            from xgboost import XGBClassifier
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            }
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            model = XGBClassifier(**params, scale_pos_weight=n_neg / max(n_pos, 1),
                                  eval_metric="logloss", verbosity=0,
                                  n_jobs=-1, random_state=SEED)
        elif algorithm == "LightGBM":
            from lightgbm import LGBMClassifier
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            }
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            model = LGBMClassifier(**params, scale_pos_weight=n_neg / max(n_pos, 1),
                                   verbose=-1, n_jobs=-1, random_state=SEED)
        else:
            raise ValueError(f"Unknown: {algorithm}")

        model.fit(X_train, y_train)
        try:
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        except Exception:
            return 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    bp = study.best_params
    if algorithm == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**bp, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1)
    elif algorithm == "XGBoost":
        from xgboost import XGBClassifier
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        model = XGBClassifier(**bp, scale_pos_weight=n_neg / max(n_pos, 1),
                              eval_metric="logloss", verbosity=0,
                              n_jobs=-1, random_state=SEED)
    elif algorithm == "LightGBM":
        from lightgbm import LGBMClassifier
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        model = LGBMClassifier(**bp, scale_pos_weight=n_neg / max(n_pos, 1),
                               verbose=-1, n_jobs=-1, random_state=SEED)

    model.fit(X_train, y_train)
    return model, model.predict_proba(X_val)[:, 1], study.best_value


def run_groupkfold(X, y, groups, algorithm):
    """GroupKFold 5-fold CV with Optuna per fold."""
    from sklearn.model_selection import GroupKFold

    log(f"\n--- GroupKFold ({N_FOLDS}-fold), {algorithm} ---")
    gkf = GroupKFold(n_splits=N_FOLDS)

    ug = sorted(set(groups))
    gm = {g: i for i, g in enumerate(ug)}
    gids = np.array([gm[g] for g in groups])

    fold_results = []
    all_yt, all_yp = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=gids)):
        X_t, X_v = X[tr_idx], X[va_idx]
        y_t, y_v = y[tr_idx], y[va_idx]
        n_pt, n_nt = int(np.sum(y_t == 1)), int(np.sum(y_t == 0))
        log(f"  Fold {fold+1}: train {len(y_t)} ({n_pt}P/{n_nt}N), val {len(y_v)}")

        model, y_prob, best_auc = train_stage2_model(X_t, y_t, X_v, y_v, algorithm)
        auc = roc_auc_score(y_v, y_prob)
        auprc = average_precision_score(y_v, y_prob)
        log(f"    Val AUC={auc:.4f} AUPRC={auprc:.4f} (Optuna best={best_auc:.4f})")

        fold_results.append({"fold": fold + 1, "auc_roc": auc, "auprc": auprc})
        all_yt.extend(y_v)
        all_yp.extend(y_prob)

    mean_auc = np.mean([r["auc_roc"] for r in fold_results])
    std_auc = np.std([r["auc_roc"] for r in fold_results])
    mean_auprc = np.mean([r["auprc"] for r in fold_results])
    log(f"  Overall: AUC={mean_auc:.4f} +/- {std_auc:.4f}, AUPRC={mean_auprc:.4f}")
    return fold_results


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Ablation experiments
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(X_full, y, groups, feature_names, algorithm):
    """Ablation: Full, No_Stage1 (mol only), No_MolFeat (tox only)."""
    print_header("Step 6: Ablation Experiments")

    tox_c = [i for i, n in enumerate(feature_names)
             if n.startswith("ToxA_") or n.startswith("ToxB_")]
    mol_c = [i for i, n in enumerate(feature_names)
             if n.startswith("Diff_") or n.startswith("Prod_")]

    exps = {
        "Full": X_full,
        "No_Stage1": X_full[:, mol_c],
        "No_MolFeat": X_full[:, tox_c],
    }
    results = []
    for name, X_e in exps.items():
        log(f"\n--- Ablation: {name} ({X_e.shape[1]} features) ---")
        fr = run_groupkfold(X_e, y, groups, algorithm)
        row = {"experiment": name, "n_features": X_e.shape[1],
               "auc_roc": np.mean([r["auc_roc"] for r in fr]),
               "auc_roc_std": np.std([r["auc_roc"] for r in fr]),
               "auprc": np.mean([r["auprc"] for r in fr])}
        for r in fr:
            row[f"fold_{r['fold']}_auc"] = r["auc_roc"]
            row[f"fold_{r['fold']}_auprc"] = r["auprc"]
        results.append(row)
        log(f"  {name}: AUC={row['auc_roc']:.4f} +/- {row['auc_roc_std']:.4f}, "
            f"AUPRC={row['auprc']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 60)
    print("  Stage 2: Drug-Pair Interaction Safety Risk Assessment")
    print("=" * 60)

    # Load meta
    with open(META_FILE) as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]
    log(f"Molecular features: {len(feature_cols)} dims")

    # Step 1: Load endpoint config
    endpoint_cfg = load_endpoint_config()

    # Step 2: Generate toxicity profiles (skip if already computed)
    tox_profile_path = os.path.join(RESULTS_DIR, "stage1_toxicity_profiles.csv")
    if os.path.exists(tox_profile_path):
        log(f"Toxicity profile exists at {tox_profile_path}, loading...")
        tox_df = pd.read_csv(tox_profile_path)
        tox_cols = [c for c in tox_df.columns if c.startswith("tox_")]
        log(f"  Loaded {tox_df.shape[0]} compounds x {len(tox_cols)} endpoints")
    else:
        tox_df, tox_cols = generate_toxicity_profiles(feature_cols, endpoint_cfg)

    # Step 3: Build DDI pairs
    df_pairs = build_ddi_pairs()

    # Step 4: Build feature matrix
    X, y, groups, feature_names = build_feature_matrix(df_pairs, tox_df, tox_cols, feature_cols)

    # Step 5: Algorithm comparison
    print_header("Step 5: Algorithm Comparison")
    alg_results = []
    for alg in ["RF", "XGBoost", "LightGBM"]:
        fr = run_groupkfold(X, y, groups, alg)
        row = {"algorithm": alg,
               "auc_roc": np.mean([r["auc_roc"] for r in fr]),
               "auc_roc_std": np.std([r["auc_roc"] for r in fr]),
               "auprc": np.mean([r["auprc"] for r in fr])}
        for r in fr:
            row[f"fold_{r['fold']}_auc"] = r["auc_roc"]
        alg_results.append(row)

    pd.DataFrame(alg_results).to_csv(
        os.path.join(RESULTS_DIR, "stage2_results.csv"), index=False)
    log("Saved: stage2_results.csv")

    best_alg = max(alg_results, key=lambda r: r["auc_roc"])
    log(f"Best algorithm: {best_alg['algorithm']} (AUC={best_alg['auc_roc']:.4f})")

    # Step 6: Ablation
    ablation_results = run_ablation(X, y, groups, feature_names, best_alg["algorithm"])
    pd.DataFrame(ablation_results).to_csv(
        os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    log("Saved: ablation_results.csv")

    # Summary
    print_header("Summary")
    for r in alg_results:
        log(f"  {r['algorithm']:12s}: AUC={r['auc_roc']:.4f}+/-{r['auc_roc_std']:.4f}, "
            f"AUPRC={r['auprc']:.4f}")
    log("")
    for r in ablation_results:
        log(f"  {r['experiment']:15s}: AUC={r['auc_roc']:.4f}+/-{r['auc_roc_std']:.4f}, "
            f"AUPRC={r['auprc']:.4f}")

    log(f"\nTotal: {(time.time() - t0) / 60:.1f} min")
    log("Done!")


if __name__ == "__main__":
    main()