"""
DDID External Validation of Stage 2 DDI Risk Assessment
=======================================================

External validation of the Stage 2 drug-pair interaction safety model using
DDID (Drug-Drug Interaction Database) real herb-drug interaction records,
which were NOT used in training (the model was trained on DrugBank DDI
positives vs random negatives).

Pipeline:
  1. Load DDID Interaction / Drug / NP / Herb Information tables.
  2. Bridge each DDID interaction to the 47,724-compound feature space:
       - Drug side:  Drug Information (FHDI_Drug_ID -> DrugBank_ID or
                      Canonical_SMILES canonicalization) -> feature ID
       - Herb side:  Component column (specific natural-product name,
                      matched to NP Information pref_name) if available,
                      else all NP compounds of the herb
                      (NP Information FHDI_Herb_ID) -> canonical SMILES
                      -> feature ID
  3. Re-train Stage 2 LightGBM exactly as 07_case_study.build_stage2_model:
     DrugBank DDI positives (<=50k) vs random negatives (50k), features
     ToxA(21) + ToxB(21) + |molDiff|(1210) + molProd(1210) = 2462 dims.
  4. Predict DDI risk for every bridged DDID pair (label=1).
  5. Random control (label=0): equal number of random pairs sampled from the
     47,724-feature compound space, excluding DDID bridged pairs and all
     DrugBank positive pairs.
  6. Statistics: mean risk DDID vs random, Mann-Whitney U p-value, external
     validation AUC (separating DDID pairs from random pairs).
  7. Outputs to data/02_processed/phase2/results/ddid_validation/:
       - ddid_predictions.csv  (all predictions with labels)
       - ddid_distribution.png (boxplot + ECDF of risk distributions)
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Path resolution ----------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "02_processed"
PHASE1_DIR = DATA_DIR / "phase1"
PHASE2_DIR = DATA_DIR / "phase2"
SPLIT_DIR = PHASE2_DIR / "split"
RESULTS_DIR = PHASE2_DIR / "results"
EVAL_DIR = RESULTS_DIR / "evaluation"
OUT_DIR = RESULTS_DIR / "ddid_validation"

# DDID raw data
DDID_DIR = _PROJECT_ROOT / "data" / "01_raw_data" / "ddid"
INTER_CSV = DDID_DIR / "Interaction Information.csv"
DRUG_CSV = DDID_DIR / "Drug Information.csv"
NP_CSV = DDID_DIR / "NP Information.csv"
HERB_CSV = DDID_DIR / "Herb Information.csv"

# Phase 2 feature space / profiles
TRAIN_FEATURES = SPLIT_DIR / "train_features.csv"
TEST_FEATURES = SPLIT_DIR / "test_features.csv"
META_FILE = SPLIT_DIR / "split_meta.json"
STAGE1_PROFILES = EVAL_DIR / "stage1_toxicity_profiles.csv"

# Stage 2 training positives (DrugBank DDI, as in 07_case_study.py)
DDI_DRUGBANK_CSV = PHASE1_DIR / "step4_ddi_integrated" / "ddi_drugbank.csv"

SEED = 42

# ---- Input validation ---------------------------------------------------
missing = [str(p) for p in
           [INTER_CSV, DRUG_CSV, NP_CSV, HERB_CSV, TRAIN_FEATURES, TEST_FEATURES,
            META_FILE, STAGE1_PROFILES, DDI_DRUGBANK_CSV] if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files: " + ", ".join(missing))

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 1. Load split meta, feature space, canonical SMILES mapping
# =====================================================================
print("\n--- Loading feature space ---")
with open(META_FILE, "r") as f:
    meta = json.load(f)
FEATURE_COLUMNS = meta["feature_columns"]
assert len(FEATURE_COLUMNS) == 1210, f"Expected 1210 features, got {len(FEATURE_COLUMNS)}"

train_feat = pd.read_csv(TRAIN_FEATURES)
test_feat = pd.read_csv(TEST_FEATURES)
feat_all = pd.concat([train_feat, test_feat], ignore_index=True)
id_to_smiles = dict(zip(feat_all["ID"], feat_all["SMILES"]))
all_ids = list(id_to_smiles.keys())
id_set = set(all_ids)
print(f"  Feature space: {len(all_ids)} compounds")

# Canonical SMILES -> feature ID (silently tolerates missing rdkit)
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canon_smiles(s):
        if pd.isna(s) or not str(s).strip():
            return None
        try:
            mol = Chem.MolFromSmiles(str(s))
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except Exception:
            return None

    smiles_to_id = {}
    for _, row in feat_all.iterrows():
        c = canon_smiles(row["SMILES"])
        if c and c not in smiles_to_id:
            smiles_to_id[c] = row["ID"]
    print(f"  Canonical SMILES -> ID map: {len(smiles_to_id)} entries")
    CANON_AVAIL = True
except ImportError:
    CANON_AVAIL = False
    print("  WARNING: rdkit unavailable; SMILES-canonicalization bridging disabled")

# Molecule feature vectors per ID (for stage-2 feature construction)
mol_vec = pd.concat([
    train_feat[["ID"] + FEATURE_COLUMNS],
    test_feat[["ID"] + FEATURE_COLUMNS],
], ignore_index=True).set_index("ID")
mol_vec = mol_vec.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
print(f"  Molecule feature matrix: {mol_vec.shape}")

# Stage 1 toxicity profiles
tox_df = pd.read_csv(STAGE1_PROFILES).set_index("ID")
tox_cols = [c for c in tox_df.columns if c.startswith("tox_")]
assert len(tox_cols) == 21, f"Expected 21 tox endpoints, got {len(tox_cols)}"
tox_mat = tox_df[tox_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
tox_mat = tox_mat.reindex(index=mol_vec.index).fillna(0.0)
print(f"  Stage-1 tox profiles: {tox_mat.shape} ({len(tox_cols)} endpoints)")


# =====================================================================
# 2. Load DDID tables
# =====================================================================
print("\n--- Loading DDID tables ---")
inter_df = pd.read_csv(str(INTER_CSV), encoding="utf-8-sig")
drug_df = pd.read_csv(str(DRUG_CSV), encoding="utf-8-sig")
np_df = pd.read_csv(str(NP_CSV), encoding="utf-8-sig")
herb_df = pd.read_csv(str(HERB_CSV), encoding="utf-8-sig")
print(f"  Interactions: {len(inter_df)} rows")
print(f"  Drugs:        {len(drug_df)} rows")
print(f"  NPs:          {len(np_df)} rows")
print(f"  Herbs:        {len(herb_df)} rows")

# Drug Information is keyed on FHDI_Drug_ID; check prefix used in interaction file
drug_id_key = None
for cand in ["FHDI_Drug_ID", "Drug_ID"]:
    if cand in drug_df.columns:
        drug_id_key = cand
        break
if drug_id_key is None:
    raise ValueError(f"No drug ID column found in {DRUG_CSV}: {list(drug_df.columns)}")
drug_lookup = drug_df.set_index(drug_id_key)
print(f"  Drug info keyed on: {drug_id_key}")


# =====================================================================
# 3. Bridge drug side: FHDI drug -> feature ID
# =====================================================================
print("\n--- Bridging drugs to feature space ---")
drug_br_ok = 0
drug_br_by_db = 0
drug_br_by_smi = 0
drug_br_fail = 0
interaction_drug_ids = set(inter_df["Drug_ID"].dropna().astype(str))
br_drug_id_map = {}   # FHDI drug ID -> feature ID
for did in interaction_drug_ids:
    row = drug_lookup.loc[did] if did in drug_lookup.index else None
    if row is None:
        drug_br_fail += 1
        continue
    fid = None
    # Strategy 1: DrugBank_ID directly matches feature ID
    dbid = row.get("DrugBank_ID")
    if pd.notna(dbid) and str(dbid).strip() and str(dbid).strip() in id_set:
        fid = str(dbid).strip()
        drug_br_by_db += 1
    # Strategy 2: canonicalize Canonical_SMILES -> feature ID
    if fid is None and CANON_AVAIL:
        csm = row.get("Canonical_SMILES")
        c = canon_smiles(csm)
        if c and c in smiles_to_id:
            fid = smiles_to_id[c]
            drug_br_by_smi += 1
    if fid is None:
        drug_br_fail += 1
    else:
        br_drug_id_map[did] = fid
        drug_br_ok += 1

n_drug_ids = len(interaction_drug_ids)
print(f"  Interaction drugs: {n_drug_ids}")
print(f"  Bridged (DrugBank_ID direct): {drug_br_by_db}")
print(f"  Bridged (canonical SMILES):   {drug_br_by_smi}")
print(f"  NOT bridged:                  {drug_br_fail}")
print(f"  Bridged total:                {drug_br_ok} ({drug_br_ok / n_drug_ids:.1%})")


# =====================================================================
# 4. Bridge herb side: herb/component -> NP compounds -> feature IDs
# =====================================================================
print("\n--- Bridging herbs to feature space ---")

def norm_compound_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[()'\"]*$", "", s)  # strip trailing parens/quotes (e.g. "(alliin)")
    return s

# Normalized pref_name -> list of NP row indices
pref_to_idx = {}
for idx, p in np_df["pref_name"].dropna().items():
    k = norm_compound_name(p)
    pref_to_idx.setdefault(k, []).append(idx)

# FHDI_Herb_ID -> NP row indices (herb-level compound set)
herb_to_idx = {}
for idx, h in np_df["FHDI_Herb_ID"].dropna().items():
    herb_to_idx.setdefault(str(h).strip(), []).append(idx)

# NP SMILES canonicalization -> feature ID(s), cached per NP row
np_smi_cache = {}
np_feature_id_cache = {}

def np_smiles_to_feature_id(row_idx):
    """Canonicalize an NP row's SMILES and map to a feature-space ID."""
    if row_idx in np_feature_id_cache:
        return np_feature_id_cache[row_idx]
    if not CANON_AVAIL:
        np_feature_id_cache[row_idx] = None
        return None
    if row_idx not in np_smi_cache:
        np_smi_cache[row_idx] = canon_smiles(np_df.at[row_idx, "SMILES"])
    c = np_smi_cache[row_idx]
    fid = smiles_to_id.get(c) if c else None
    np_feature_id_cache[row_idx] = fid
    return fid

# Artifact tokens in Component column that are NOT compound names
_ARTIFACT_TOKENS = {
    "na", "nan", "none", "juice", "juices", "extract", "extracts", "tea", "tea infusion",
    "water", "fasted overnight", "on empty stomach", "capsule", "tablet", "oral",
    "powder", "dried extract", "standardized extract", "alcoholic extract", "aqueous extract",
    "hydro-alcoholic extract", "drug", "whole plant", "the fruit", "residue", "leaf",
    "leaves", "powdered", "dose", "single dose", "repeated dose", "daily", "hours",
    "mg", "ml", "g", "l", "day", "days", "weekly",
}

def herb_side_feature_ids(hid, component):
    """Return set of feature IDs for the herb side of one interaction row."""
    fids = set()
    comp = "" if pd.isna(component) else str(component)

    # Route A: Component is a specific natural-product name -> NP Information
    if comp.strip():
        k = norm_compound_name(comp)
        if k and k not in _ARTIFACT_TOKENS and k in pref_to_idx:
            for row_idx in pref_to_idx[k]:
                fid = np_smiles_to_feature_id(row_idx)
                if fid:
                    fids.add(fid)

    # Route B: herb-level compounds (NP Information FHDI_Herb_ID)
    if hid:
        key = str(hid).strip()
        if key in herb_to_idx:
            for row_idx in herb_to_idx[key]:
                fid = np_smiles_to_feature_id(row_idx)
                if fid:
                    fids.add(fid)

    return fids


# Build unique (drug_feature_id, herb_compound_feature_id) pairs across all rows
row_counts = {"bridged_pairs": 0, "rows_with_pair": 0, "rows_no_drug": 0, "rows_no_herb": 0}
bridged_pairs = {}   # (drugFID, herbCID) -> meta

for _, r in inter_df.iterrows():
    did = r["Drug_ID"] if pd.notna(r["Drug_ID"]) else None
    hid = r["Food_Herb_ID"] if pd.notna(r["Food_Herb_ID"]) else None
    comp = r.get("Component")

    drug_fid = br_drug_id_map.get(str(did).strip()) if did is not None else None
    herb_fids = herb_side_feature_ids(hid, comp)

    if drug_fid is None:
        row_counts["rows_no_drug"] += 1
        continue
    if not herb_fids:
        row_counts["rows_no_herb"] += 1
        continue

    row_counts["rows_with_pair"] += 1
    meta_info = {
        "drug_id": str(did),
        "drug_name": r["Drug_Name"] if pd.notna(r.get("Drug_Name")) else "",
        "food_herb_id": str(hid) if hid is not None else "",
        "food_herb_name": r["Food_Herb_Name"] if pd.notna(r.get("Food_Herb_Name")) else "",
        "component": comp if pd.notna(comp) else "",
        "relationship": r.get("Relationship_classification", ""),
    }
    for herb_fid in herb_fids:
        key = (drug_fid, herb_fid)
        if key not in bridged_pairs:
            bridged_pairs[key] = meta_info

print(f"  Interaction rows:                {len(inter_df)}")
print(f"  Rows with no bridged drug side:  {row_counts['rows_no_drug']}")
print(f"  Rows with no bridged herb side:  {row_counts['rows_no_herb']}")
print(f"  Rows with >=1 bridged pair:      {row_counts['rows_with_pair']}")
print(f"  Unique bridged (drug, compound) pairs: {len(bridged_pairs)}")

# Hit rate metrics
drugs_bridged = len(br_drug_id_map)
herb_compounds_bridged = len(
    {cid for _, cid in bridged_pairs.keys()})
all_compounds_bridged = len(
    {cid for cid, _ in bridged_pairs.keys()} | {cid for _, cid in bridged_pairs.keys()})

print(f"  Drugs bridged:      {drugs_bridged} unique (feature IDs)")
print(f"  Herb compounds bridged (unique feature IDs): {herb_compounds_bridged}")
print(f"  Compounds bridged either side (unique): {all_compounds_bridged}")
print(f"  Pairs bridged (unique drug x compound): {len(bridged_pairs)}")

if not bridged_pairs:
    raise RuntimeError("Zero DDID pairs bridged to feature space; cannot validate.")


# =====================================================================
# 5. Re-train Stage 2 LightGBM (identical construction to 07_case_study)
# =====================================================================
print("\n--- Building Stage 2 DDI model (DrugBank 50k pos vs 50k neg) ---")
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

ddi = pd.read_csv(DDI_DRUGBANK_CSV)
train_ids_arr = pd.read_csv(TRAIN_FEATURES, usecols=["ID"])["ID"].values
train_ids_set = set(train_ids_arr)

in_a = ddi["Drug1_ID"].isin(train_ids_set)
in_b = ddi["Drug2_ID"].isin(train_ids_set)
valid = ddi[in_a & in_b].copy()
valid["_pair_key"] = valid.apply(
    lambda r: tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])), axis=1)
valid = valid.drop_duplicates(subset="_pair_key").reset_index(drop=True)

n_pos = min(len(valid), 50000)
if len(valid) > 50000:
    valid = valid.sample(n=n_pos, random_state=42)

pos_set_all = set()
for _, r in ddi.iterrows():
    pos_set_all.add(tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])))
pos_set_sampled = set()
for _, r in valid.iterrows():
    pos_set_sampled.add(tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])))

np.random.seed(42)
neg_list = []
while len(neg_list) < n_pos:
    a, b = np.random.randint(0, len(train_ids_arr), 2)
    if a == b:
        continue
    key = tuple(sorted([train_ids_arr[a], train_ids_arr[b]]))
    if key not in pos_set_sampled:
        neg_list.append(key)

pos_df = pd.DataFrame({"DrugA_ID": valid["Drug1_ID"].values,
                       "DrugB_ID": valid["Drug2_ID"].values, "label": 1})
neg_arr = np.array(neg_list)
neg_df = pd.DataFrame({"DrugA_ID": neg_arr[:, 0],
                       "DrugB_ID": neg_arr[:, 1], "label": 0})
df_pairs = pd.concat([pos_df, neg_df], ignore_index=True)

f_a = mol_vec.reindex(df_pairs["DrugA_ID"]).values.astype(np.float32)
f_b = mol_vec.reindex(df_pairs["DrugB_ID"]).values.astype(np.float32)
t_a = tox_mat.reindex(df_pairs["DrugA_ID"]).values.astype(np.float32)
t_b = tox_mat.reindex(df_pairs["DrugB_ID"]).values.astype(np.float32)

X = np.concatenate([
    t_a, t_b,
    np.abs(f_a - f_b).astype(np.float32),
    (f_a * f_b).astype(np.float32),
], axis=1)
y = df_pairs["label"].values.astype(np.float32)
assert X.shape[1] == 2462, f"Expected 2462 features, got {X.shape[1]}"
print(f"  Training matrix: {X.shape}, positives {int(y.sum())}, negatives {int(len(y) - y.sum())}")

n_neg = int(np.sum(y == 0))
n_p = int(np.sum(y == 1))
stage2_model = LGBMClassifier(scale_pos_weight=n_neg / max(n_p, 1),
                              verbose=-1, n_jobs=-1, random_state=42)
stage2_model.fit(X, y)
train_auc = roc_auc_score(y, stage2_model.predict_proba(X)[:, 1])
print(f"  LightGBM trained (train AUC {train_auc:.4f})")


# =====================================================================
# 6. Predict DDI risk for bridged DDID pairs (label=1)
# =====================================================================
print("\n--- Predicting DDI risk for bridged DDID pairs ---")
pair_keys = list(bridged_pairs.keys())
idx_a = [k[0] for k in pair_keys]
idx_b = [k[1] for k in pair_keys]
f_a = mol_vec.loc[idx_a].values.astype(np.float32)
f_b = mol_vec.loc[idx_b].values.astype(np.float32)
t_a = tox_mat.loc[idx_a].values.astype(np.float32)
t_b = tox_mat.loc[idx_b].values.astype(np.float32)
X_ddid = np.concatenate([t_a, t_b, np.abs(f_a - f_b), f_a * f_b], axis=1)
ddid_risks = stage2_model.predict_proba(X_ddid)[:, 1].astype(np.float64)
print(f"  Predicted {len(pair_keys)} DDID pairs (mean risk {ddid_risks.mean():.4f})")


# =====================================================================
# 7. Random control pairs (label=0) — exclude DDID bridged + DrugBank pos
# =====================================================================
print("\n--- Sampling random control pairs ---")
all_arr = np.array(all_ids)
excluded = pos_set_all | set(pair_keys)
rng = np.random.RandomState(SEED)
random_pairs = set()
while len(random_pairs) < len(pair_keys):
    a, b = rng.randint(0, len(all_arr), 2)
    if a == b:
        continue
    key = tuple(sorted([all_arr[a], all_arr[b]]))
    if key not in excluded and key not in random_pairs:
        random_pairs.add(key)
random_keys = list(random_pairs)
print(f"  Sampled {len(random_keys)} random pairs (excl. DDID bridged + DrugBank positives)")

f_a = mol_vec.loc[[k[0] for k in random_keys]].values.astype(np.float32)
f_b = mol_vec.loc[[k[1] for k in random_keys]].values.astype(np.float32)
t_a = tox_mat.loc[[k[0] for k in random_keys]].values.astype(np.float32)
t_b = tox_mat.loc[[k[1] for k in random_keys]].values.astype(np.float32)
X_rnd = np.concatenate([t_a, t_b, np.abs(f_a - f_b), f_a * f_b], axis=1)
random_risks = stage2_model.predict_proba(X_rnd)[:, 1].astype(np.float64)
print(f"  Predicted {len(random_keys)} random pairs (mean risk {random_risks.mean():.4f})")


# =====================================================================
# 8. Statistics: external validation metrics
# =====================================================================
print("\n--- External validation statistics ---")
ddid_mean = float(ddid_risks.mean())
random_mean = float(random_risks.mean())
diff = ddid_mean - random_mean

y_all = np.concatenate([np.ones(len(ddid_risks)), np.zeros(len(random_risks))])
risk_all = np.concatenate([ddid_risks, random_risks])
ext_auc = float(roc_auc_score(y_all, risk_all))

try:
    stat, p_value = mannwhitneyu(ddid_risks, random_risks, alternative="two-sided")
    p_value = float(p_value)
    stat1, p_greater = mannwhitneyu(ddid_risks, random_risks, alternative="greater")
    p_greater = float(p_greater)
except Exception as e:
    print(f"  WARNING: Mann-Whitney failed ({e}); computing via rank approximation")
    from scipy.stats import ranksums
    stat, p_value = ranksums(ddid_risks, random_risks)
    p_value = float(p_value)
    p_greater = p_value

print(f"  DDID pair mean risk:          {ddid_mean:.4f}")
print(f"  Random pair mean risk:        {random_mean:.4f}")
print(f"  Mean difference (DDID-random): {diff:+.4f}")
print(f"  External validation AUC:      {ext_auc:.4f}")
print(f"  Mann-Whitney U p (two-sided): {p_value:.4e}")
print(f"  Mann-Whitney U p (greater):   {p_greater:.4e}")

if p_value < 0.05:
    print("  => DDID-derived herb-drug pairs are assessed at SIGNIFICANTLY")
    print("     different risk than random pairs.")
else:
    print("  => Difference between DDID and random pairs not significant (p>=0.05).")


# =====================================================================
# 9. Save CSV + distribution figure
# =====================================================================
rows = []
for k, risk in zip(pair_keys, ddid_risks):
    meta_info = bridged_pairs[k]
    rows.append({
        "label": 1, "group": "ddid",
        "drug_id": meta_info["drug_id"], "drug_name": meta_info["drug_name"],
        "drug_feature_id": k[0], "food_herb_id": meta_info["food_herb_id"],
        "food_herb_name": meta_info["food_herb_name"], "component": meta_info["component"],
        "herb_compound_id": k[1],
        "herb_compound_name": id_to_smiles.get(k[1], ""),
        "relationship_classification": meta_info["relationship"],
        "ddi_risk_probability": round(float(risk), 6),
    })
for k, risk in zip(random_keys, random_risks):
    rows.append({
        "label": 0, "group": "random",
        "drug_id": "", "drug_name": "", "drug_feature_id": k[0],
        "food_herb_id": "", "food_herb_name": "", "component": "",
        "herb_compound_id": k[1], "herb_compound_name": id_to_smiles.get(k[1], ""),
        "relationship_classification": "",
        "ddi_risk_probability": round(float(risk), 6),
    })
out_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "ddid_predictions.csv"
out_df.to_csv(csv_path, index=False)
print(f"\n  Predictions CSV saved: {csv_path} ({len(out_df)} rows)")

# Figure: side-by-side boxplot + ECDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: boxplot
bp = axes[0].boxplot([ddid_risks, random_risks], tick_labels=["DDID pairs", "Random pairs"],
                     patch_artist=True, widths=0.55)
for patch, col in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
    patch.set_facecolor(col)
    patch.set_alpha(0.55)
axes[0].set_ylabel("Stage 2 DDI risk probability")
axes[0].set_title("Risk distribution: DDID vs random pairs")
axes[0].grid(axis="y", alpha=0.3)

# Right: ECDF
for risks, label, col in [(ddid_risks, "DDID pairs", "#d62728"),
                          (random_risks, "Random pairs", "#1f77b4")]:
    xs = np.sort(risks)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    axes[1].plot(xs, ys, label=label, color=col, lw=2)
axes[1].set_xlabel("Stage 2 DDI risk probability")
axes[1].set_ylabel("ECDF")
axes[1].set_title("Cumulative distributions")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle(
    f"DDID External Validation of Stage 2 DDI Risk\n"
    f"AUC={ext_auc:.4f} | mean DDID={ddid_mean:.4f} vs random={random_mean:.4f} | "
    f"Mann-Whitney p={p_value:.3e}",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig_path = OUT_DIR / "ddid_distribution.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Distribution figure saved: {fig_path}")


# =====================================================================
# 10. Final summary
# =====================================================================
print("\n" + "=" * 78)
print("DDID EXTERNAL VALIDATION SUMMARY")
print("=" * 78)
print(f"  Interaction rows:              {len(inter_df)}")
print(f"  Unique drugs in interactions:  {n_drug_ids}")
print(f"  Drugs bridged:                 {drugs_bridged} "
      f"({drugs_bridged / n_drug_ids:.1%} hit rate)")
print(f"  Compounds bridged (unique feature IDs, either side): {all_compounds_bridged}")
print(f"    - herb side: {herb_compounds_bridged}, "
      f"drug side: {len({k[0] for k in pair_keys})}")
print(f"  Pairs bridged & scored:        {len(pair_keys)}")
print(f"  Rows with >=1 scored pair:     {row_counts['rows_with_pair']} "
      f"({row_counts['rows_with_pair'] / len(inter_df):.1%})")
print(f"  DDID mean risk:                {ddid_mean:.4f}")
print(f"  Random mean risk:              {random_mean:.4f}")
print(f"  External validation AUC:       {ext_auc:.4f}")
print(f"  Mann-Whitney U p (two-sided):  {p_value:.4e}")
print(f"  Mann-Whitney U p (greater):    {p_greater:.4e}")
print(f"  CSV:  {csv_path}")
print(f"  FIG:  {fig_path}")
print("=" * 78)
print("DONE. DDID external validation complete.")
