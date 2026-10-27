"""
06_shap_interpretability.py — SHAP Three-Layer Interpretability Analysis

Layer 1: Global Feature Importance — SHAP summary plot across all endpoints
Layer 2: Endpoint-Level Dependence — SHAP dependence plots for top endpoints
Layer 3: Case-Level Waterfall — Individual prediction explanation

Also: Stage 2 SHAP analysis for DDI risk assessment

Output (saved to /openbayes/home/  or local results/shap/):
  - global_shap_summary.png
  - shap_dependence_{endpoint}.png
  - shap_waterfall_{case_name}.png
  - stage2_shap_summary.png
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Paths (use /openbayes/home/ on server, local otherwise) ────────
HOME_DIR = '/openbayes/home/project'
if not os.path.exists(HOME_DIR):
    HOME_DIR = config.PROJECT_ROOT  # fallback to local

SPLIT_DIR = os.path.join(HOME_DIR, 'data', '02_processed', 'phase2', 'split')
RESULTS_DIR = os.path.join(HOME_DIR, 'data', '02_processed', 'phase2', 'results', 'evaluation')
SHAP_DIR = os.path.join(HOME_DIR, 'data', '02_processed', 'phase2', 'results', 'shap')
os.makedirs(SHAP_DIR, exist_ok=True)

SEED = 42
N_SAMPLES_SHAP = 500  # SHAP background samples (speed vs accuracy)
N_SAMPLES_PREDICT = 1000  # SHAP prediction samples

# ─── Endpoints ──────────────────────────────────────────────────────
TOX21_EPS = [f"Tox21_{i}" for i in range(12)]
CTD_GENE_EPS = ["gene_CYP1A2", "gene_CYP2C9", "gene_CYP2C19",
                "gene_CYP2D6", "gene_CYP3A4", "gene_CYP3A5", "gene_KCNH2"]
CTD_TOX_EPS = ["tox_hepatotoxicity", "tox_cardiotoxicity",
               "tox_nephrotoxicity", "tox_neurotoxicity"]
ALL_EPS = TOX21_EPS + CTD_GENE_EPS + CTD_TOX_EPS
SKIP_EPS = {"Tox21_10", "Tox21_11", "gene_SCN5A", "gene_KCNQ1", "gene_KCNE1"}
VALID_EPS = [ep for ep in ALL_EPS if ep not in SKIP_EPS]

EP_DATASET = {}
for ep in TOX21_EPS: EP_DATASET[ep] = 'tox21'
for ep in CTD_GENE_EPS + CTD_TOX_EPS: EP_DATASET[ep] = 'ctd'

# Load feature columns
with open(os.path.join(SPLIT_DIR, 'split_meta.json')) as f:
    META = json.load(f)
FEATURE_COLS = META['feature_columns']

# Endpoint groups for analysis
ENDPOINT_GROUPS = {
    'tox21': [ep for ep in VALID_EPS if ep.startswith('Tox21')],
    'ctd_tox': [ep for ep in VALID_EPS if ep.startswith('tox_')],
    'ctd_cyp': [ep for ep in VALID_EPS if ep.startswith('gene_')],
}

# Feature groups
FEATURE_GROUPS = {
    'Descriptors': ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
                    'NumRotatableBonds', 'NumAromaticRings', 'RingCount',
                    'FractionCSP3', 'HeavyAtomCount'],
    'Morgan Fingerprints': [c for c in FEATURE_COLS if c.startswith('Morgan_')],
    'MACCS Keys': [c for c in FEATURE_COLS if c.startswith('MACCS_')],
}

TOP_ENDPOINTS_SHAP = 6  # number of endpoints to do full SHAP on

# ─── Utility ─────────────────────────────────────────────────────────

def log(msg):
    print(f"  {msg}")

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─── Data Loading ────────────────────────────────────────────────────

def load_data():
    """Load train features + labels, return merged DataFrame."""
    log("Loading training data...")
    tr = pd.read_csv(os.path.join(SPLIT_DIR, 'train_features.csv'))
    te = pd.read_csv(os.path.join(SPLIT_DIR, 'test_features.csv'))
    df_feat = pd.concat([tr, te], ignore_index=True)

    tox21_tr = pd.read_csv(os.path.join(SPLIT_DIR, 'tox21_train_labels.csv'))
    tox21_te = pd.read_csv(os.path.join(SPLIT_DIR, 'tox21_test_labels.csv'))
    ctd_tr = pd.read_csv(os.path.join(SPLIT_DIR, 'ctd_train_labels.csv'))
    ctd_te = pd.read_csv(os.path.join(SPLIT_DIR, 'ctd_test_labels.csv'))

    return df_feat, tox21_tr, tox21_te, ctd_tr, ctd_te

def get_train_test_idx(df_feat, ids_train, ids_test, id_col='ID'):
    train_idx = np.where(df_feat[id_col].isin(ids_train))[0]
    test_idx = np.where(df_feat[id_col].isin(ids_test))[0]
    return train_idx, test_idx

# ─── Model Training (lightweight, no Optuna) ─────────────────────────

def build_model(algorithm, n_pos, n_neg):
    if algorithm == 'RF':
        return RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1)
    elif algorithm == 'XGBoost':
        spw = n_neg / max(n_pos, 1)
        return XGBClassifier(scale_pos_weight=spw, eval_metric='logloss',
                             verbosity=0, n_jobs=-1, random_state=SEED)
    elif algorithm == 'LightGBM':
        spw = n_neg / max(n_pos, 1)
        return LGBMClassifier(scale_pos_weight=spw, verbose=-1,
                              n_jobs=-1, random_state=SEED)

def train_endpoint_model(algorithm, X, y):
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    model = build_model(algorithm, n_pos, n_neg)
    model.fit(X, y)
    return model

# ─── Layer 1: Global SHAP Summary ────────────────────────────────────

def layer1_global_summary(models, X_test_dict, endpoint_names):
    """Layer 1: Global feature importance across endpoints."""
    print_header("Layer 1: Global SHAP Feature Importance")

    try:
        import shap
    except ImportError:
        log("shap not installed, installing...")
        os.system(f"{sys.executable} -m pip install shap -q")
        import shap

    # Compute SHAP values for each endpoint model
    all_importances = {}
    for ep_name in endpoint_names[:TOP_ENDPOINTS_SHAP]:
        if ep_name not in models:
            continue
        log(f"  Computing SHAP for {ep_name}...")
        model = models[ep_name]
        X_test = X_test_dict[ep_name]

        # Sample for SHAP (speed)
        if len(X_test) > N_SAMPLES_PREDICT:
            X_sample = X_test.sample(n=N_SAMPLES_PREDICT, random_state=SEED)
        else:
            X_sample = X_test

        if len(X_train_background := X_test.sample(n=min(N_SAMPLES_SHAP, len(X_test)),
                                                     random_state=SEED)) < 10:
            continue

        # TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # shap_values could be list (for RF) or array (for XGB/LGB)
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class
        else:
            sv = shap_values

        # Mean |SHAP| per feature
        mean_shap = np.abs(sv).mean(axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=-1)
        all_importances[ep_name] = mean_shap

    if not all_importances:
        log("No SHAP values computed.")
        return

    # Aggregate: top features across endpoints
    imp_df = pd.DataFrame(all_importances, index=FEATURE_COLS)
    imp_df['mean'] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values('mean', ascending=False)

    # Plot: Top 20 features
    top_n = 20
    top_features = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_features['mean'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Mean |SHAP| (across endpoints)')
    ax.set_title('Global SHAP Feature Importance (Top 20 Features)')
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(SHAP_DIR, 'global_shap_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {out_path}")

    # Print top features
    log(f"\nTop {top_n} features across {len(all_importances)} endpoints:")
    for i, (feat, row) in enumerate(top_features.iterrows()):
        log(f"  {i+1:2d}. {feat:<30s} Mean |SHAP| = {row['mean']:.6f}")

    return imp_df

# ─── Layer 2: Endpoint-Level Dependence ──────────────────────────────

def layer2_endpoint_dependence(models, X_test_dict, endpoint_names):
    """Layer 2: SHAP dependence plots for top endpoints."""
    print_header("Layer 2: Endpoint-Level SHAP Dependence")

    try:
        import shap
    except ImportError:
        import shap

    for ep_name in endpoint_names[:TOP_ENDPOINTS_SHAP]:
        if ep_name not in models:
            continue
        log(f"  Dependence plot for {ep_name}...")
        model = models[ep_name]
        X_test = X_test_dict[ep_name]

        if len(X_test) > N_SAMPLES_PREDICT:
            X_sample = X_test.sample(n=N_SAMPLES_PREDICT, random_state=SEED)
        else:
            X_sample = X_test

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # Find top 3 features for this endpoint
        mean_abs_shap = np.abs(sv).mean(axis=0)
        top3_idx = np.argsort(mean_abs_shap)[-3:][::-1].ravel()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax_idx, feat_idx in enumerate(top3_idx):
            feat_name = FEATURE_COLS[feat_idx]
            shap.dependence_plot(
                feat_idx, sv, X_sample,
                feature_names=FEATURE_COLS,
                ax=axes[ax_idx], show=False
            )
            axes[ax_idx].set_title(f'{ep_name}\\n{feat_name}', fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(SHAP_DIR, f'shap_dependence_{ep_name}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"    Saved: {out_path}")

# ─── Layer 3: Case-Level Waterfall ───────────────────────────────────

def layer3_case_waterfall(models, X_test_dict, endpoint_names):
    """Layer 3: Case-level SHAP waterfall plots."""
    print_header("Layer 3: Case-Level SHAP Waterfall")

    try:
        import shap
    except ImportError:
        import shap

    # Select a few representative test samples (high/low prediction)
    for ep_name in endpoint_names[:4]:
        if ep_name not in models:
            continue
        model = models[ep_name]
        X_test = X_test_dict[ep_name]

        if len(X_test) < 5:
            continue

        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]

        # Pick 2 cases: high risk and low risk
        high_idx = np.argmax(y_prob)
        low_idx = np.argmin(y_prob)

        explainer = shap.TreeExplainer(model)

        for case_name, case_idx in [('high_risk', high_idx), ('low_risk', low_idx)]:
            X_case = X_test.iloc[[case_idx]]
            shap_values_case = explainer.shap_values(X_case, check_additivity=False)
            if isinstance(shap_values_case, list):
                sv = shap_values_case[1][0]
            else:
                sv = shap_values_case[0]

            # Waterfall plot
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(values=sv,
                                 base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                                 else explainer.expected_value[1],
                                 data=X_case.values[0],
                                 feature_names=FEATURE_COLS),
                show=False, max_display=15
            )
            plt.title(f'{ep_name} - {case_name} (p={y_prob[case_idx]:.3f})')
            plt.tight_layout()
            out_path = os.path.join(SHAP_DIR, f'shap_waterfall_{ep_name}_{case_name}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            log(f"    Saved: {out_path}")

# ─── Stage 2 SHAP Analysis ───────────────────────────────────────────

def stage2_shap_analysis():
    """SHAP analysis for Stage 2 LightGBM model (drug-pair features)."""
    print_header("Stage 2 SHAP Analysis")

    try:
        import shap
    except ImportError:
        import shap

    stage2_file = os.path.join(RESULTS_DIR, 'stage2_results.csv')
    tox_profile_file = os.path.join(RESULTS_DIR, 'stage1_toxicity_profiles.csv')
    if not os.path.exists(stage2_file) or not os.path.exists(tox_profile_file):
        log("Stage 2 results or toxicity profile not found, skipping.")
        return

    log("Loading Stage 2 data and re-training LightGBM...")

    # Reload DDI data and train LightGBM
    # (Using 05_stage2_ddi functions — recreate the feature matrix)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Quick retrain: load DDI pairs + features
    df_feat = pd.read_csv(os.path.join(HOME_DIR, 'data', '02_processed', 'phase1',
                                        'step3_features', 'all_features.csv'),
                          usecols=['ID'])

    # Build feature matrix (simplified: use the last fold's approach)
    from sklearn.model_selection import GroupKFold

    # Re-create DDI pairs from DrugBank
    ddi_file = os.path.join(HOME_DIR, 'data', '02_processed', 'phase1',
                            'step4_ddi_integrated', 'ddi_drugbank.csv')
    if not os.path.exists(ddi_file):
        log("DDI file not found, skipping Stage 2 SHAP.")
        return

    ddi = pd.read_csv(ddi_file)
    all_ids = set(df_feat['ID'])
    in_a = ddi['Drug1_ID'].isin(all_ids)
    in_b = ddi['Drug2_ID'].isin(all_ids)
    valid = ddi[in_a & in_b].copy()
    valid['_pair_key'] = valid.apply(
        lambda r: tuple(sorted([r['Drug1_ID'], r['Drug2_ID']])), axis=1)
    valid = valid.drop_duplicates(subset='_pair_key').reset_index(drop=True)

    n_pos = min(len(valid), 50000)
    if len(valid) > 50000:
        valid = valid.sample(n=50000, random_state=SEED)

    pos_set = set()
    for _, r in valid.iterrows():
        pos_set.add(tuple(sorted([r['Drug1_ID'], r['Drug2_ID']])))

    # Sample negatives
    np.random.seed(SEED)
    neg_list = []
    all_ids_arr = np.array(list(all_ids))
    n_compounds = len(all_ids_arr)
    while len(neg_list) < n_pos:
        a, b = np.random.randint(0, n_compounds, 2)
        key = tuple(sorted([all_ids_arr[a], all_ids_arr[b]]))
        if key not in pos_set and a != b:
            neg_list.append(key)

    # Build DFs
    pos_df = pd.DataFrame({'DrugA_ID': valid['Drug1_ID'].values,
                           'DrugB_ID': valid['Drug2_ID'].values,
                           'label': 1})
    neg_arr = np.array(neg_list)
    neg_df = pd.DataFrame({'DrugA_ID': neg_arr[:, 0],
                           'DrugB_ID': neg_arr[:, 1],
                           'label': 0})
    df_pairs = pd.concat([pos_df, neg_df], ignore_index=True)

    # Feature engineering (same as 05_stage2_ddi.py)
    tox_df = pd.read_csv(tox_profile_file)
    tox_cols = [c for c in tox_df.columns if c.startswith('tox_')]

    tr_f = pd.read_csv(os.path.join(SPLIT_DIR, 'train_features.csv'),
                       usecols=['ID'] + FEATURE_COLS)
    te_f = pd.read_csv(os.path.join(SPLIT_DIR, 'test_features.csv'),
                       usecols=['ID'] + FEATURE_COLS)
    df_mol = pd.concat([tr_f, te_f], ignore_index=True)

    m_a = df_pairs.merge(df_mol, left_on='DrugA_ID', right_on='ID', how='left')
    m_b = df_pairs.merge(df_mol, left_on='DrugB_ID', right_on='ID', how='left')
    tox_flat = tox_df.reset_index()
    m_a = m_a.merge(tox_flat, on='ID', how='left')
    m_b = m_b.merge(tox_flat, on='ID', how='left')

    mol_a = m_a[FEATURE_COLS].fillna(0).values.astype(np.float32)
    mol_b = m_b[FEATURE_COLS].fillna(0).values.astype(np.float32)
    tox_a = m_a[tox_cols].fillna(0).values.astype(np.float32)
    tox_b = m_b[tox_cols].fillna(0).values.astype(np.float32)

    X = np.concatenate([tox_a, tox_b,
                        np.abs(mol_a - mol_b).astype(np.float32),
                        (mol_a * mol_b).astype(np.float32)], axis=1)

    y = df_pairs['label'].values.astype(np.float32)

    # Feature names
    fnames = ([f'ToxA_{c}' for c in tox_cols] + [f'ToxB_{c}' for c in tox_cols] +
              [f'Diff_{c}' for c in FEATURE_COLS] + [f'Prod_{c}' for c in FEATURE_COLS])

    # Train LightGBM
    n_pos_t = int(np.sum(y == 1))
    n_neg_t = int(np.sum(y == 0))
    model = LGBMClassifier(scale_pos_weight=n_neg_t / max(n_pos_t, 1),
                           verbose=-1, n_jobs=-1, random_state=SEED)
    model.fit(X, y)
    from sklearn.metrics import roc_auc_score
    log(f"Stage 2 LightGBM trained, AUC={roc_auc_score(y, model.predict_proba(X)[:,1]):.4f}")

    # SHAP analysis
    log("Computing Stage 2 SHAP values...")
    X_df = pd.DataFrame(X, columns=fnames)
    X_sample = X_df.sample(n=min(N_SAMPLES_PREDICT, len(X_df)), random_state=SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Mean |SHAP| per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-20:][::-1]

    # Feature group analysis
    log("\nStage 2 SHAP Top 20 Features:")
    for i, idx in enumerate(top_idx):
        log(f"  {i+1:2d}. {fnames[idx]:<30s} Mean |SHAP| = {mean_shap[idx]:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_names = [fnames[i] for i in top_idx]
    top_values = [mean_shap[i] for i in top_idx]
    colors = []
    for n in top_names:
        if n.startswith('Tox'): colors.append('#ff7f0e')
        elif n.startswith('Diff'): colors.append('#1f77b4')
        else: colors.append('#2ca02c')

    ax.barh(range(len(top_names)), top_values, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('Stage 2 SHAP Feature Importance (LightGBM)')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f0e', label='Toxicity Profile'),
                       Patch(facecolor='#1f77b4', label='Mol Diff/Prod'),
                       Patch(facecolor='#2ca02c', label='Mol Prod')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(SHAP_DIR, 'stage2_shap_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {out_path}")

# ─── Feature Group Analysis ──────────────────────────────────────────

def feature_group_analysis(imp_df):
    """Analyze SHAP importance by feature group (descriptors vs Morgan vs MACCS)."""
    if imp_df is None:
        return

    print_header("Feature Group Analysis")

    group_importance = {}
    for group_name, cols in FEATURE_GROUPS.items():
        common_cols = [c for c in cols if c in imp_df.index]
        if common_cols:
            group_importance[group_name] = imp_df.loc[common_cols, 'mean'].mean()

    log("Mean |SHAP| by feature group:")
    for name, val in sorted(group_importance.items(), key=lambda x: -x[1]):
        log(f"  {name:<25s} {val:.6f}")

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(group_importance.keys())
    sizes = list(group_importance.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('SHAP Importance by Feature Group')

    plt.tight_layout()
    out_path = os.path.join(SHAP_DIR, 'feature_group_importance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("  SHAP Interpretability Analysis (Phase 2)")
    print("=" * 60)

    log(f"Home dir: {HOME_DIR}")
    log(f"SHAP output: {SHAP_DIR}")

    # ── Load best endpoint config ────────────────────────────────────
    best_file = os.path.join(RESULTS_DIR, 'best_per_endpoint.csv')
    best_cfg = {}
    if os.path.exists(best_file):
        bdf = pd.read_csv(best_file)
        for _, r in bdf.iterrows():
            best_cfg[r['endpoint']] = {'algorithm': r['algorithm'], 'dataset': r['dataset']}
    log(f"Loaded {len(best_cfg)} endpoint configs")

    # ── Load data ────────────────────────────────────────────────────
    df_feat, tox21_tr, tox21_te, ctd_tr, ctd_te = load_data()
    X_all = df_feat[FEATURE_COLS].values.astype(np.float32)
    all_ids = df_feat['ID'].values
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}

    # SMILES mapping for Tox21
    smiles_to_idx = {}
    for _, row in df_feat.iterrows():
        smiles_to_idx[row.get('SMILES', row['ID'])] = id_to_idx[row['ID']]

    from rdkit import Chem
    canon_to_id = {}
    for _, row in df_feat.iterrows():
        mol = Chem.MolFromSmiles(row.get('SMILES', ''))
        if mol:
            canon_to_id[Chem.MolToSmiles(mol)] = row['ID']

    # ── Train models + collect test sets ─────────────────────────────
    models = {}
    X_test_dict = {}
    endpoint_names = []

    for endpoint in VALID_EPS:
        ds = EP_DATASET[endpoint]
        cfg = best_cfg.get(endpoint, {'algorithm': 'RF'})
        algorithm = cfg['algorithm']

        # Load labels
        if ds == 'tox21':
            tr = tox21_tr[['SMILES', endpoint]].dropna()
            te = tox21_te[['SMILES', endpoint]].dropna()
            tr_idx, tr_lbl = [], []
            for _, row in tr.iterrows():
                idx = smiles_to_idx.get(row['SMILES'])
                if idx is not None:
                    tr_idx.append(idx)
                    tr_lbl.append(row[endpoint])
            te_idx, te_lbl = [], []
            for _, row in te.iterrows():
                idx = smiles_to_idx.get(row['SMILES'])
                if idx is not None:
                    te_idx.append(idx)
                    te_lbl.append(row[endpoint])
        else:
            tr = ctd_tr[['smiles', endpoint]].dropna()
            te = ctd_te[['smiles', endpoint]].dropna()
            tr_idx, tr_lbl = [], []
            for _, row in tr.iterrows():
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    cid = canon_to_id.get(Chem.MolToSmiles(mol))
                    if cid and cid in id_to_idx:
                        tr_idx.append(id_to_idx[cid])
                        tr_lbl.append(row[endpoint])
            te_idx, te_lbl = [], []
            for _, row in te.iterrows():
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    cid = canon_to_id.get(Chem.MolToSmiles(mol))
                    if cid and cid in id_to_idx:
                        te_idx.append(id_to_idx[cid])
                        te_lbl.append(row[endpoint])

        if len(tr_idx) < 10 or len(te_idx) < 5:
            continue

        X_tr = X_all[tr_idx]
        X_te = X_all[te_idx]
        y_tr = np.array(tr_lbl, dtype=np.float32)
        y_te = np.array(te_lbl, dtype=np.float32)

        model = train_endpoint_model(algorithm, X_tr, y_tr)
        models[endpoint] = model

        X_test_df = pd.DataFrame(X_te, columns=FEATURE_COLS)
        X_test_dict[endpoint] = X_test_df
        endpoint_names.append(endpoint)

        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        log(f"  [{len(endpoint_names):2d}/{len(VALID_EPS)}] {endpoint} ({algorithm}): AUC={auc:.4f}")

    # ── SHAP Layers ──────────────────────────────────────────────────
    imp_df = layer1_global_summary(models, X_test_dict, endpoint_names)
    # Layer 2 skipped (SHAP compat)
    # Layer 3 skipped (SHAP compat)
    log("Skipping Layer 2/3 due to SHAP version compatibility")
    feature_group_analysis(imp_df)

    # ── Stage 2 SHAP ─────────────────────────────────────────────────
    stage2_shap_analysis()

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed/60:.1f} minutes")
    log(f"All SHAP outputs saved to: {SHAP_DIR}")
    log("Done!")

if __name__ == '__main__':
    main()