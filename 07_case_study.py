"""
Case Study Analysis for Herb-Drug Interaction Safety Risk Assessment
====================================================================

Four clinical cases:
  1. St. John's Wort (贯叶连翘) + Warfarin    -> CYP induction risk
  2. Ginkgo biloba (白果叶) + Aspirin        -> Bleeding/coagulation risk
  3. Fuzi Lizhong Wan (附子理中丸) + Digoxin  -> Cardiotoxicity risk
  4. Liuwei Dihuang Wan (六味地黄丸) + safe drug -> Low risk (negative control)

Strategy:
  - Read TCMBank herb/ingredient tables to link herb names -> compounds
  - Multi-strategy search across herb name columns (latin, english, chinese, category)
  - DDID NP Information as secondary ingredient source
  - Map compounds via DrugBank_id -> feature 'ID', fallback to SMILES canonicalization
  - Load best per-endpoint trained models, predict toxicity profiles
  - Fallback to pre-computed stage1_toxicity_profiles.csv when models unavailable
  - Output CSV, radar charts, summary table
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---- Path resolution ------------------------------------------------
OPENBAYES_HOME = Path("/openbayes/home/project")
if OPENBAYES_HOME.exists():
    PROJECT_ROOT = OPENBAYES_HOME
else:
    _script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = _script_dir.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "02_processed"
PHASE1_DIR = DATA_DIR / "phase1"
PHASE2_DIR = DATA_DIR / "phase2"
SPLIT_DIR = PHASE2_DIR / "split"
MODELS_DIR = PHASE2_DIR / "models" / "traditional_ml"
RESULTS_DIR = PHASE2_DIR / "results"
EVAL_DIR = RESULTS_DIR / "evaluation"
SHAP_DIR = RESULTS_DIR / "shap"
CASE_STUDY_DIR = RESULTS_DIR / "case_study"

# TCMBank data
TCMBANK_HERBS = PHASE1_DIR / "integrated" / "tcmbank_herbs.csv"
TCMBANK_INGREDIENTS = PHASE1_DIR / "integrated" / "tcmbank_ingredients.csv"
DRUGBANK_SMILES = PHASE1_DIR / "integrated" / "drugbank_smiles.csv"

# Pre-computed toxicity profiles (fallback when models not available)
STAGE1_PROFILES = EVAL_DIR / "stage1_toxicity_profiles.csv"

# DDID NP Information (additional ingredient source, if available)
DDID_NP_INFO = PROJECT_ROOT / "data" / "01_raw_data" / "ddid" / "NP Information.csv"
DDID_HERB_INFO = PROJECT_ROOT / "data" / "01_raw_data" / "ddid" / "Herb Information.csv"

# Phase 2 split data
TRAIN_FEATURES = SPLIT_DIR / "train_features.csv"
TEST_FEATURES = SPLIT_DIR / "test_features.csv"
META_FILE = SPLIT_DIR / "split_meta.json"
BEST_MODEL_FILE = EVAL_DIR / "best_per_endpoint.csv"

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(CASE_STUDY_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir:    {DATA_DIR}")
print(f"SHAP dir:    {SHAP_DIR}")
print(f"Case study:  {CASE_STUDY_DIR}")


# =====================================================================
# 1. Load split metadata, feature columns, endpoints
# =====================================================================
def load_meta():
    with open(META_FILE, "r") as f:
        return json.load(f)


meta = load_meta()
FEATURE_COLUMNS = meta["feature_columns"]
TOX21_ENDPOINTS = meta["tox21"]["endpoints"]      # 12 endpoints
CTD_ENDPOINTS = meta["ctd"]["endpoints"]           # 14 endpoints
ALL_ENDPOINTS = TOX21_ENDPOINTS + CTD_ENDPOINTS     # 26 total

# Read feature files to build SMILES -> ID mapping
print("\n--- Loading feature data ---")
train_feat = pd.read_csv(TRAIN_FEATURES)
test_feat = pd.read_csv(TEST_FEATURES)
feat_all = pd.concat([train_feat, test_feat], ignore_index=True)

# Build ID -> SMILES mapping
id_to_smiles = dict(zip(feat_all["ID"], feat_all["SMILES"]))

# Build canonical SMILES -> ID mapping
try:
    from rdkit import Chem

    def canon_smiles(s):
        """Canonicalize a SMILES string."""
        if pd.isna(s) or not str(s).strip():
            return None
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    smiles_to_id = {}
    for _, row in feat_all.iterrows():
        sid = row["ID"]
        smi = row["SMILES"]
        if pd.notna(smi) and str(smi).strip():
            c = canon_smiles(str(smi))
            if c:
                smiles_to_id[c] = sid
    CANON_AVAIL = True
    print(f"  Built canonical SMILES mapping: {len(smiles_to_id)} entries")
except ImportError:
    CANON_AVAIL = False
    print("  RDKit not available; SMILES fallback disabled")


# =====================================================================
# 2. Load best per-endpoint models list
# =====================================================================
print("\n--- Loading best model assignments ---")
best_df = pd.read_csv(BEST_MODEL_FILE)
best_map = {}
for _, row in best_df.iterrows():
    ep = row["endpoint"]
    alg = row["algorithm"]
    ds = row["dataset"]
    best_map[ep] = {"algorithm": alg, "dataset": ds}

print(f"  Loaded {len(best_map)} endpoint -> best algorithm mappings")

VALID_ENDPOINTS = [ep for ep in ALL_ENDPOINTS if ep in best_map]
SKIPPED_ENDPOINTS = [ep for ep in ALL_ENDPOINTS if ep not in best_map]
if SKIPPED_ENDPOINTS:
    print(f"  Skipping endpoints with no best model: {SKIPPED_ENDPOINTS}")


# =====================================================================
# 3. Load models (lazy - only when needed) with fallback to profiles
# =====================================================================
_model_cache = {}
_tox_profile_cache = None


def load_tox_profile_fallback():
    """Load pre-computed toxicity profiles CSV as fallback when models missing."""
    global _tox_profile_cache
    if _tox_profile_cache is not None:
        return _tox_profile_cache
    if not STAGE1_PROFILES.exists():
        print("  WARNING: No stage1_toxicity_profiles.csv available")
        _tox_profile_cache = pd.DataFrame()
        return _tox_profile_cache
    try:
        df = pd.read_csv(STAGE1_PROFILES)
        df = df.set_index("ID")
        _tox_profile_cache = df
        print(f"  Loaded pre-computed profiles for {len(df)} compounds")
        return df
    except Exception as e:
        print(f"  ERROR loading profiles: {e}")
        _tox_profile_cache = pd.DataFrame()
        return _tox_profile_cache


def load_model(endpoint):
    """Load the best model for a given endpoint, with caching."""
    if endpoint in _model_cache:
        return _model_cache[endpoint]

    if endpoint not in best_map:
        return None

    info = best_map[endpoint]
    alg = info["algorithm"]
    ds = info["dataset"]

    model_path = MODELS_DIR / f"{alg}_{ds}_{endpoint}.pkl"
    if not model_path.exists():
        return None  # Silent fallback to profile

    import pickle
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        _model_cache[endpoint] = model
        return model
    except Exception as e:
        print(f"    ERROR loading model {model_path}: {e}")
        return None


# =====================================================================
# 4. Load TCMBank data and perform robust herb matching
# =====================================================================
print("\n--- Loading TCMBank data ---")
tcm_herbs = pd.read_csv(TCMBANK_HERBS)
tcm_ingredients = pd.read_csv(TCMBANK_INGREDIENTS, low_memory=False)
print(f"  TCMBank herbs:       {len(tcm_herbs)} rows")
print(f"  TCMBank ingredients: {len(tcm_ingredients)} rows")


def load_ddid_compounds_for_herb(search_terms):
    """
    Load herb-compound links from DDID NP Information as supplementary source.

    Searches DDID Herb Information for matching herbs, then finds NP compounds
    for those herbs. Returns DataFrame with SMILES, pref_name columns.
    """
    if not DDID_HERB_INFO.exists() or not DDID_NP_INFO.exists():
        return pd.DataFrame()

    try:
        ddid_herb = pd.read_csv(str(DDID_HERB_INFO), encoding="utf-8")
        np_info = pd.read_csv(str(DDID_NP_INFO))

        matches = set()
        herb_cols = ["Herb_Latin_Name", "Herb_English_Name",
                      "Herb_Pinyin_Name", "Herb_Chinese_Name"]
        for term in search_terms:
            if not term or not str(term).strip():
                continue
            term_l = str(term).strip().lower()
            for col in herb_cols:
                if col not in ddid_herb.columns:
                    continue
                mask = ddid_herb[col].fillna("").astype(str).str.lower().str.contains(term_l, na=False, regex=False)
                matches.update(ddid_herb.loc[mask, "FHDI_Herb_ID"].values)

        if not matches:
            return pd.DataFrame()

        compounds = np_info[np_info["FHDI_Herb_ID"].isin(matches)].copy()
        compounds = compounds[compounds["SMILES"].notna() & (compounds["SMILES"] != "")]
        # Rename for consistency with TCMBank
        compounds = compounds.rename(columns={"pref_name": "name", "np_id": "TCMBank_ID"})
        compounds["DrugBank_id"] = ""
        print(f"    DDID supplement: {len(compounds)} compounds from {len(matches)} herbs")
        return compounds
    except Exception as e:
        print(f"    DDID lookup failed: {e}")
        return pd.DataFrame()


def find_ingredients_by_herb_name(search_terms):
    """
    Search TCMBank ingredients table directly for rows matching herb names.

    The ingredients table has herb metadata embedded in each row:
      - TCM_name (Chinese name, e.g. '贯叶连翘')
      - TCM_name_en (English name, e.g. 'Common St. John's Wort')
      - level1_name_en (category)
      - level2_name_en (subcategory)

    We search across these columns for multiple search terms.

    Returns: DataFrame of matching ingredient rows.
    """
    results = []
    text_cols = ["TCM_name", "TCM_name_en", "level1_name_en", "level2_name_en",
                  "name", "TCM_name2"]

    for term in search_terms:
        if not term or not str(term).strip():
            continue
        term_lower = str(term).strip().lower()
        for col in text_cols:
            if col not in tcm_ingredients.columns:
                continue
            mask = (
                tcm_ingredients[col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(term_lower, na=False, regex=False)
            )
            matched = tcm_ingredients[mask]
            if len(matched) > 0:
                results.append(matched)

    if len(results) == 0:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.drop_duplicates(subset=["TCMBank_ID", "Smiles"])
    return combined


def map_compound_to_feature(row):
    """
    Map an ingredient row to its feature-space ID.

    Strategy:
      1. DrugBank_id -> feature 'ID' column
      2. Canonicalize ingredient SMILES -> match feature SMILES -> ID

    Returns: feature ID string or None
    """
    # Strategy 1: DrugBank_id
    db_id = row.get("DrugBank_id", None)
    if pd.notna(db_id) and str(db_id).strip():
        db_id_str = str(db_id).strip()
        if db_id_str in id_to_smiles:
            return db_id_str

    # Strategy 2: SMILES canonicalization
    ing_smiles = row.get("Smiles", row.get("SMILES", None))
    if pd.notna(ing_smiles) and str(ing_smiles).strip():
        ing_smi = str(ing_smiles).strip()
        if CANON_AVAIL:
            c = canon_smiles(ing_smi)
            if c and c in smiles_to_id:
                return smiles_to_id[c]

    return None


def resolve_herb_compounds(herb_name_label, search_terms):
    """
    Full pipeline: search ingredients table by herb name -> map to feature IDs.

    Returns: list of dicts {compound_id, compound_name, ingredient_smiles, herb_name}
    """
    print(f"\n  Resolving herb: {herb_name_label}")
    ingredients = find_ingredients_by_herb_name(search_terms)
    print(f"    Total ingredient rows found: {len(ingredients)}")

    if len(ingredients) > 0:
        has_db = ingredients["DrugBank_id"].notna() & (ingredients["DrugBank_id"] != "")
        has_smi = ingredients["Smiles"].notna() & (ingredients["Smiles"] != "")
        print(f"    With DrugBank_id: {has_db.sum()}, with SMILES: {has_smi.sum()}")
    else:
        print(f"    WARNING: No ingredients found matching '{herb_name_label}'")

    results = []
    mapped_ids = set()

    if len(ingredients) > 0:
        for _, ing_row in ingredients.iterrows():
            feat_id = map_compound_to_feature(ing_row)
            if feat_id and feat_id not in mapped_ids:
                mapped_ids.add(feat_id)
                ing_name = ing_row.get("name", ing_row.get("pref_name", ""))
                ing_smi = ing_row.get("Smiles", ing_row.get("SMILES", ""))
                results.append({
                    "compound_id": feat_id,
                    "compound_name": str(ing_name),
                    "ingredient_smiles": str(ing_smi),
                    "herb_name": herb_name_label,
                })
        print(f"    Mapped to feature compounds (TCMBank): {len(results)}")

    # If TCMBank yielded no mapped compounds, try DDID supplement
    if len(results) == 0:
        ddid_compounds = load_ddid_compounds_for_herb(search_terms)
        if len(ddid_compounds) > 0:
            print(f"    Trying DDID supplement ({len(ddid_compounds)} compounds)")
            for _, ing_row in ddid_compounds.iterrows():
                feat_id = map_compound_to_feature(ing_row)
                if feat_id and feat_id not in mapped_ids:
                    mapped_ids.add(feat_id)
                    ing_name = ing_row.get("name", ing_row.get("pref_name", ""))
                    ing_smi = ing_row.get("Smiles", ing_row.get("SMILES", ""))
                    results.append({
                        "compound_id": feat_id,
                        "compound_name": str(ing_name),
                        "ingredient_smiles": str(ing_smi),
                        "herb_name": herb_name_label,
                    })
            print(f"    Mapped to feature compounds (TCMBank+DDID): {len(results)}")

    if len(results) == 0:
        print(f"    (No DrugBank IDs or SMILES from any source matched known feature compounds)")
    return results


# =====================================================================
# 5. Define case studies with comprehensive search terms
# =====================================================================
def define_cases():
    """Return a list of case dicts with comprehensive search terms."""
    cases = []

    # Case 1: St. John's Wort (贯叶连翘) + Warfarin
    cases.append({
        "case_name": "Case 1: St. John's Wort (贯叶连翘) + Warfarin",
        "herb_search": [
            {
                "herb_label": "贯叶连翘 / St. John's Wort",
                "search_terms": [
                    "Hypericum perforatum",
                    "St. John's Wort",
                    "St John's Wort",
                    "StJohnsWort",
                    "St Johns Wort",
                    "贯叶连翘",
                    "Hypericum",
                ],
            }
        ],
        "interacting_drugs": [
            {"drug_name": "Warfarin", "drug_id": "DB00682"},
        ],
        "expected_risk": "High CYP3A4 induction risk, reduced warfarin efficacy",
    })

    # Case 2: Ginkgo biloba (白果叶) + Aspirin
    cases.append({
        "case_name": "Case 2: Ginkgo biloba (白果叶) + Aspirin",
        "herb_search": [
            {
                "herb_label": "白果叶 / Ginkgo biloba",
                "search_terms": [
                    "Ginkgo biloba",
                    "Ginkgo",
                    "Ginkgo Bark",
                    "Ginkgo Leaf",
                    "白果叶",
                    "银杏叶",
                    "Ginkgoaceae",
                ],
            }
        ],
        "interacting_drugs": [
            {"drug_name": "Aspirin", "drug_id": "DB00945"},
        ],
        "expected_risk": "Enhanced anticoagulation, bleeding risk",
    })

    # Case 3: Fuzi Lizhong Wan (附子理中丸) + Digoxin
    cases.append({
        "case_name": "Case 3: Fuzi Lizhong Wan (附子理中丸) + Digoxin",
        "herb_search": [
            {
                "herb_label": "附子 / Aconitum (Fuzi)",
                "search_terms": [
                    "Aconitum",
                    "Aconite",
                    "Aconitum carmichaelii",
                    "Radix Aconiti Lateralis Preparata",
                    "Fuzi",
                    "附子",
                    "Aconiti",
                    "Aconitum kusnezoffii",
                ],
            }
        ],
        "interacting_drugs": [
            {"drug_name": "Digoxin", "drug_id": "DB00390"},
        ],
        "expected_risk": "Aconitine cardiotoxicity, high cardiac risk",
    })

    # Case 4: Liuwei Dihuang Wan (六味地黄丸) + safe drug
    # Six components: Rehmannia, Cornus, Dioscorea, Paeonia, Alisma, Poria
    cases.append({
        "case_name": "Case 4: Liuwei Dihuang Wan (六味地黄丸) + Safe Drug (negative control)",
        "herb_search": [
            {
                "herb_label": "熟地黄 / Rehmannia",
                "search_terms": [
                    "Rehmannia glutinosa",
                    "Rehmanniae",
                    "Rehmannia",
                    "地黄",
                    "熟地黄",
                    "Di Huang",
                ],
            },
            {
                "herb_label": "山茱萸 / Cornus",
                "search_terms": [
                    "Cornus officinalis",
                    "Cornus",
                    "山茱萸",
                    "Shan Zhu Yu",
                ],
            },
            {
                "herb_label": "山药 / Dioscorea",
                "search_terms": [
                    "Dioscorea opposita",
                    "Dioscorea",
                    "山药",
                    "Shan Yao",
                    "Chinese Yam",
                ],
            },
            {
                "herb_label": "牡丹皮 / Paeonia",
                "search_terms": [
                    "Paeonia suffruticosa",
                    "Paeonia",
                    "牡丹皮",
                    "Mu Dan Pi",
                    "Tree Peony",
                ],
            },
            {
                "herb_label": "泽泻 / Alisma",
                "search_terms": [
                    "Alisma orientale",
                    "Alisma",
                    "泽泻",
                    "Ze Xie",
                    "Water Plantain",
                ],
            },
            {
                "herb_label": "茯苓 / Poria",
                "search_terms": [
                    "Poria cocos",
                    "Poria",
                    "茯苓",
                    "Fu Ling",
                    "Wolfiporia",
                ],
            },
        ],
        "interacting_drugs": [
            {"drug_name": "Metformin (safe reference)", "drug_id": "DB00331"},
        ],
        "expected_risk": "Low risk (negative control)",
    })

    return cases


# =====================================================================
# 6. Toxicity prediction for a single compound
# =====================================================================
def predict_toxicity_profile(compound_id):
    """
    For a given compound ID, predict toxicity probabilities across
    all valid endpoints using the best per-endpoint trained model.

    Strategy:
      1. Try full model inference (requires .pkl model files)
      2. Fallback to pre-computed stage1_toxicity_profiles.csv
      3. If neither works, return None

    Returns: dict {endpoint: probability, ...} or None on failure.
    """
    # Check if compound exists in feature space
    row_train = train_feat[train_feat["ID"] == compound_id]
    row_test = test_feat[test_feat["ID"] == compound_id]

    if len(row_train) == 0 and len(row_test) == 0:
        return None

    # Try full model inference
    profile = {}
    all_nan = True
    if len(row_train) > 0:
        row = row_train.iloc[0]
    else:
        row = row_test.iloc[0]
    X = row[FEATURE_COLUMNS].values.astype(np.float32).reshape(1, -1)

    for ep in VALID_ENDPOINTS:
        model = load_model(ep)
        if model is None:
            profile[ep] = np.nan
            continue

        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0, 1]
            elif hasattr(model, "predict"):
                prob = float(model.predict(X)[0])
            else:
                prob = np.nan
            profile[ep] = prob
            if not np.isnan(prob):
                all_nan = False
        except Exception:
            profile[ep] = np.nan

    # If all models unavailable, fall back to pre-computed profiles
    if all_nan:
        tox_df = load_tox_profile_fallback()
        if len(tox_df) > 0 and compound_id in tox_df.index:
            profile = {}
            for ep in VALID_ENDPOINTS:
                # Map column names from stage1_toxicity_profiles format to endpoints
                if ep.startswith("Tox21_"):
                    col = f"tox_{ep}"
                elif ep.startswith("tox_") or ep.startswith("gene_"):
                    col = f"tox_{ep}"
                else:
                    col = ep

                if col in tox_df.columns:
                    val = tox_df.loc[compound_id, col]
                    profile[ep] = float(val) if pd.notna(val) else np.nan
                else:
                    profile[ep] = np.nan

    # Check if we have any non-NaN predictions
    non_nan_count = sum(1 for v in profile.values() if not np.isnan(v))
    if non_nan_count == 0:
        return None

    return profile


# =====================================================================
# 7. Toxicity profile generation for a case
# =====================================================================
def compute_case_profiles(case):
    """
    For a given case, resolve all herb compounds and interacting drugs,
    then compute full toxicity profiles.

    Returns: list of profile dicts, one per compound.
    """
    all_compounds = []

    for herb_spec in case["herb_search"]:
        compounds = resolve_herb_compounds(
            herb_spec["herb_label"], herb_spec["search_terms"]
        )
        all_compounds.extend(compounds)

    # Deduplicate by compound_id
    seen_ids = set()
    unique_compounds = []
    for c in all_compounds:
        if c["compound_id"] not in seen_ids:
            seen_ids.add(c["compound_id"])
            unique_compounds.append(c)

    print(f"\n  Total unique herb compounds to evaluate: {len(unique_compounds)}")

    results = []
    for comp in unique_compounds:
        cid = comp["compound_id"]
        profile = predict_toxicity_profile(cid)

        if profile is None:
            print(f"    Compound {cid} ({comp['compound_name'][:40]}) -> NO PROFILE available")
            continue

        row = {
            "case": case["case_name"],
            "compound_id": cid,
            "compound_name": comp["compound_name"],
            "herb_name": comp["herb_name"],
            "type": "herb_compound",
        }
        row.update(profile)
        results.append(row)
        non_nan = sum(1 for v in profile.values() if not np.isnan(v))
        print(f"    [{cid}] {comp['compound_name'][:50]:50s} -> {non_nan}/{len(profile)} endpoints predicted")

    for drug in case["interacting_drugs"]:
        did = drug["drug_id"]
        profile = predict_toxicity_profile(did)
        if profile is None:
            print(f"    Drug {did} ({drug['drug_name']}) -> NOT FOUND in features")
            continue
        row = {
            "case": case["case_name"],
            "compound_id": did,
            "compound_name": drug["drug_name"],
            "herb_name": drug["drug_name"],
            "type": "interacting_drug",
        }
        row.update(profile)
        results.append(row)
        non_nan = sum(1 for v in profile.values() if not np.isnan(v))
        print(f"    [{did}] {drug['drug_name']:50s} -> {non_nan}/{len(profile)} endpoints predicted")

    return results


# =====================================================================
# 8. Stage 2 DDI Risk Prediction
# =====================================================================
def build_stage2_model():
    """Train Stage 2 LightGBM model for DDI risk prediction."""
    print("\n--- Building Stage 2 DDI model ---")

    # Load DDI data
    ddi_file = PHASE1_DIR / "step4_ddi_integrated" / "ddi_drugbank.csv"
    if not ddi_file.exists():
        print("  WARNING: DDI file not found, skipping Stage 2")
        return None, None, None

    ddi = pd.read_csv(ddi_file)
    df_feat = pd.read_csv(TRAIN_FEATURES, usecols=["ID"])
    all_ids = set(df_feat["ID"])
    all_ids_arr = df_feat["ID"].values

    in_a = ddi["Drug1_ID"].isin(all_ids)
    in_b = ddi["Drug2_ID"].isin(all_ids)
    valid = ddi[in_a & in_b].copy()
    valid["_pair_key"] = valid.apply(
        lambda r: tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])), axis=1)
    valid = valid.drop_duplicates(subset="_pair_key").reset_index(drop=True)

    n_pos = min(len(valid), 50000)
    if len(valid) > 50000:
        valid = valid.sample(n=50000, random_state=42)

    pos_set = set()
    for _, r in valid.iterrows():
        pos_set.add(tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])))

    np.random.seed(42)
    neg_list = []
    while len(neg_list) < n_pos:
        a, b = np.random.randint(0, len(all_ids_arr), 2)
        key = tuple(sorted([all_ids_arr[a], all_ids_arr[b]]))
        if key not in pos_set and a != b:
            neg_list.append(key)

    pos_df = pd.DataFrame({"DrugA_ID": valid["Drug1_ID"].values, "DrugB_ID": valid["Drug2_ID"].values, "label": 1})
    neg_arr = np.array(neg_list)
    neg_df = pd.DataFrame({"DrugA_ID": neg_arr[:, 0], "DrugB_ID": neg_arr[:, 1], "label": 0})
    df_pairs = pd.concat([pos_df, neg_df], ignore_index=True)

    # Build features
    tox_df = load_tox_profile_fallback()
    tox_cols = [c for c in tox_df.columns if c.startswith("tox_")]

    tr_f = pd.read_csv(TRAIN_FEATURES, usecols=["ID"] + FEATURE_COLUMNS)
    te_f = pd.read_csv(TEST_FEATURES, usecols=["ID"] + FEATURE_COLUMNS)
    df_mol = pd.concat([tr_f, te_f], ignore_index=True)

    m_a = df_pairs.merge(df_mol, left_on="DrugA_ID", right_on="ID", how="left")
    m_b = df_pairs.merge(df_mol, left_on="DrugB_ID", right_on="ID", how="left")
    tox_flat = tox_df.reset_index()
    m_a = m_a.merge(tox_flat, on="ID", how="left")
    m_b = m_b.merge(tox_flat, on="ID", how="left")

    mol_a = m_a[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    mol_b = m_b[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    tox_a = m_a[tox_cols].fillna(0).values.astype(np.float32)
    tox_b = m_b[tox_cols].fillna(0).values.astype(np.float32)

    X = np.concatenate([tox_a, tox_b, np.abs(mol_a - mol_b).astype(np.float32), (mol_a * mol_b).astype(np.float32)], axis=1)
    y = df_pairs["label"].values.astype(np.float32)
    fnames = ([f"ToxA_{c}" for c in tox_cols] + [f"ToxB_{c}" for c in tox_cols] +
              [f"Diff_{c}" for c in FEATURE_COLUMNS] + [f"Prod_{c}" for c in FEATURE_COLUMNS])

    n_pos_t, n_neg_t = int(np.sum(y == 1)), int(np.sum(y == 0))
    model = LGBMClassifier(scale_pos_weight=n_neg_t / max(n_pos_t, 1), verbose=-1, n_jobs=-1, random_state=42)
    model.fit(X, y)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"  LightGBM trained: AUC={auc:.4f}")

    # Store for prediction use
    return model, fnames, tox_cols


def predict_ddi_risk(stage2_model, fnames, tox_cols, drug_a_id, drug_b_id):
    """Predict DDI risk for a single drug pair."""
    if stage2_model is None:
        return None

    tox_df = load_tox_profile_fallback()
    df_mol = pd.concat([
        pd.read_csv(TRAIN_FEATURES, usecols=["ID"] + FEATURE_COLUMNS).set_index("ID"),
        pd.read_csv(TEST_FEATURES, usecols=["ID"] + FEATURE_COLUMNS).set_index("ID"),
    ])

    if drug_a_id not in df_mol.index or drug_b_id not in df_mol.index:
        return None

    tox_a = np.nan_to_num(tox_df.loc[drug_a_id, tox_cols].values.astype(np.float32), nan=0.0) if drug_a_id in tox_df.index else np.zeros(len(tox_cols), dtype=np.float32)
    tox_b = np.nan_to_num(tox_df.loc[drug_b_id, tox_cols].values.astype(np.float32), nan=0.0) if drug_b_id in tox_df.index else np.zeros(len(tox_cols), dtype=np.float32)
    mol_a = df_mol.loc[drug_a_id].values.astype(np.float32)
    mol_b = df_mol.loc[drug_b_id].values.astype(np.float32)

    X = np.concatenate([tox_a, tox_b, np.abs(mol_a - mol_b), mol_a * mol_b]).reshape(1, -1)
    prob = float(stage2_model.predict_proba(X)[0, 1])
    return prob


# =====================================================================
# 9. Save results and generate summary
# =====================================================================
def print_summary_table(results_df):
    """Print a formatted summary table grouped by case."""
    print("\n" + "=" * 130)
    print("CASE STUDY RESULTS SUMMARY")
    print("=" * 130)

    for case_name in results_df["case"].unique():
        subset = results_df[results_df["case"] == case_name]
        print(f"\n{'=' * 90}")
        print(f"  {case_name}")
        print(f"{'=' * 90}")

        herb_rows = subset[subset["type"] == "herb_compound"]
        drug_rows = subset[subset["type"] == "interacting_drug"]

        print(f"\n  Herb compounds ({len(herb_rows)}):")
        if len(herb_rows) > 0:
            for cat_label, ep_list in [
                ("Tox21", TOX21_ENDPOINTS),
                ("CTD Toxicity", [e for e in CTD_ENDPOINTS if e.startswith("tox_")]),
                ("CTD Gene", [e for e in CTD_ENDPOINTS if e.startswith("gene_")]),
            ]:
                valid_eps = [e for e in ep_list if e in herb_rows.columns]
                if not valid_eps:
                    continue
                probs = herb_rows[valid_eps].values
                mean_probs = np.nanmean(probs, axis=0)
                top_idx = np.argsort(mean_probs)[-5:][::-1]
                top_strs = [f"{valid_eps[i]}: {mean_probs[i]:.4f}" for i in top_idx]
                print(f"    {cat_label} top means: {', '.join(top_strs)}")

        if len(drug_rows) > 0:
            print(f"\n  Interacting drug(s) ({len(drug_rows)}):")
            for _, drow in drug_rows.iterrows():
                print(f"    [{drow['compound_id']}] {drow['compound_name']}")
                for cat_label, ep_list in [
                    ("Tox21", TOX21_ENDPOINTS),
                    ("CTD Toxicity", [e for e in CTD_ENDPOINTS if e.startswith("tox_")]),
                    ("CTD Gene", [e for e in CTD_ENDPOINTS if e.startswith("gene_")]),
                ]:
                    valid_eps = [e for e in ep_list if e in drow.index]
                    if not valid_eps:
                        continue
                    vals = {e: drow[e] for e in valid_eps if pd.notna(drow[e])}
                    if vals:
                        top_items = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_strs = [f"{e}: {v:.4f}" for e, v in top_items]
                        print(f"      {cat_label} top: {', '.join(top_strs)}")

        expected = subset["expected_risk"].iloc[0] if "expected_risk" in subset.columns else ""
        print(f"\n  Expected risk: {expected}")

    print("\n" + "=" * 130)


def save_results(results_list, output_csv_path, shap_csv_path):
    """Save all results to CSV files."""
    if not results_list:
        print("WARNING: No results to save!")
        placeholder = pd.DataFrame({"message": ["No compounds mapped or predicted"]})
        placeholder.to_csv(output_csv_path, index=False)
        placeholder.to_csv(shap_csv_path, index=False)
        return placeholder

    df = pd.DataFrame(results_list)
    meta_cols = ["case", "compound_id", "compound_name", "herb_name", "type", "expected_risk"]
    endpoint_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + endpoint_cols]

    df.to_csv(output_csv_path, index=False)
    print(f"\n  Results saved to: {output_csv_path}")
    df.to_csv(shap_csv_path, index=False)
    print(f"  Also saved to: {shap_csv_path}")

    print(f"  Total rows: {len(df)}")
    return df


def generate_radar_charts(results_df):
    """Generate radar charts for each case and endpoint category."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from math import pi
    except ImportError:
        print("  matplotlib not available; skipping radar charts")
        return

    tox21_eps = [e for e in TOX21_ENDPOINTS if e in results_df.columns]
    ctd_tox_eps = [e for e in CTD_ENDPOINTS if e.startswith("tox_") and e in results_df.columns]
    ctd_gene_eps = [e for e in CTD_ENDPOINTS if e.startswith("gene_") and e in results_df.columns]
    tox21_names = meta.get("tox21", {}).get("assay_names", {})

    for case_name in results_df["case"].unique():
        subset = results_df[results_df["case"] == case_name]

        for group_name, eps in [
            ("Tox21", tox21_eps),
            ("CTD_Toxicity", ctd_tox_eps),
            ("CTD_Gene", ctd_gene_eps),
        ]:
            if len(eps) < 3:
                continue

            labels = [tox21_names.get(e, e.replace("tox_", "").replace("gene_", "")) for e in eps]

            N = len(labels)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=7)

            colors = plt.cm.tab10(np.linspace(0, 1, len(subset)))
            for i, (_, row) in enumerate(subset.iterrows()):
                values = [row[e] if pd.notna(row[e]) else 0.0 for e in eps]
                values += values[:1]
                label = f"{row['compound_name'][:20]} ({row['type'][:4]})"
                ax.plot(angles, values, "o-", linewidth=1.5, color=colors[i], label=label, alpha=0.8)
                ax.fill(angles, values, alpha=0.05, color=colors[i])

            ax.set_ylim(0, 1)
            ax.set_title(f"{case_name[:60]} - {group_name}", fontsize=10, pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)

            safe_name = case_name.replace(" ", "_").replace("/", "_").replace(":", "")[:40]
            out_path = SHAP_DIR / f"case_study_radar_{safe_name}_{group_name}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Radar chart saved: {out_path}")


# =====================================================================
# 9. Main execution
# =====================================================================
def main():
    print("\n" + "=" * 80)
    print("PHASE 2 - CASE STUDY ANALYSIS")
    print("=" * 80)

    # Build Stage 2 DDI model (once for all cases)
    stage2_model, stage2_fnames, stage2_tox_cols = build_stage2_model()

    cases = define_cases()
    all_results = []
    ddi_results = []  # Stage 2 DDI predictions

    for case in cases:
        print(f"\n{'#' * 70}")
        print(f"# {case['case_name']}")
        print(f"# Expected: {case['expected_risk']}")
        print(f"{'#' * 70}")

        # --- Stage 1: Toxicity profiles ---
        profiles = compute_case_profiles(case)
        for p in profiles:
            p["expected_risk"] = case["expected_risk"]
        all_results.extend(profiles)

        # --- Stage 2: DDI risk predictions ---
        if stage2_model is not None:
            print(f"\n  --- Stage 2 DDI Predictions ---")
            herb_compounds = [p for p in profiles if p["type"] == "herb_compound"]
            drug_compounds = [p for p in profiles if p["type"] == "interacting_drug"]

            for hc in herb_compounds:
                for dc in drug_compounds:
                    ddi_prob = predict_ddi_risk(stage2_model, stage2_fnames, stage2_tox_cols,
                                                hc["compound_id"], dc["compound_id"])
                    if ddi_prob is not None:
                        ddi_results.append({
                            "case": case["case_name"],
                            "herb_compound_id": hc["compound_id"],
                            "herb_compound_name": hc["compound_name"],
                            "herb_name": hc["herb_name"],
                            "drug_id": dc["compound_id"],
                            "drug_name": dc["compound_name"],
                            "ddi_risk_probability": round(ddi_prob, 4),
                            "risk_level": "HIGH" if ddi_prob > 0.7 else ("MODERATE" if ddi_prob > 0.4 else "LOW"),
                            "expected_risk": case["expected_risk"],
                        })

            # Case-level DDI summary
            ddi_probs = [r["ddi_risk_probability"] for r in ddi_results if r["case"] == case["case_name"]]
            if ddi_probs:
                print(f"    Mean DDI risk: {np.mean(ddi_probs):.4f}")
                print(f"    Max DDI risk:  {np.max(ddi_probs):.4f}")
                print(f"    High risk pairs: {sum(1 for p in ddi_probs if p > 0.7)}/{len(ddi_probs)}")
                print(f"    Moderate risk:   {sum(1 for p in ddi_probs if 0.4 < p <= 0.7)}/{len(ddi_probs)}")

                # Top-5 most risky herb compounds
                top5 = sorted(ddi_results, key=lambda r: r["ddi_risk_probability"], reverse=True)[:5]
                print(f"    Top-5 risky herb compounds:")
                for t in top5:
                    print(f"      {t['herb_compound_id']:20s} {t['herb_compound_name'][:30]:30s} + {t['drug_name']:15s} -> DDI risk={t['ddi_risk_probability']:.4f}")

    # Save results
    output_csv = CASE_STUDY_DIR / "case_study_results.csv"
    shap_csv = SHAP_DIR / "case_study_results.csv"
    df = save_results(all_results, output_csv, shap_csv)

    # Save DDI results separately
    if ddi_results:
        ddi_df = pd.DataFrame(ddi_results)
        ddi_csv = CASE_STUDY_DIR / "case_study_ddi_predictions.csv"
        ddi_df.to_csv(ddi_csv, index=False)
        print(f"  DDI predictions saved to: {ddi_csv}")
        print(f"  Total DDI pairs predicted: {len(ddi_df)}")

        # Print DDI summary table
        print("\n" + "=" * 100)
        print("STAGE 2 DDI RISK SUMMARY")
        print("=" * 100)
        print(f"{'Case':<40s} {'Mean DDI Risk':<15s} {'Max Risk':<12s} {'High%':<10s} {'Expected':<20s}")
        print("-" * 100)
        for case_name in ddi_df["case"].unique():
            sub = ddi_df[ddi_df["case"] == case_name]
            mean_r = sub["ddi_risk_probability"].mean()
            max_r = sub["ddi_risk_probability"].max()
            high_pct = (sub["ddi_risk_probability"] > 0.7).mean() * 100
            expected = sub["expected_risk"].iloc[0]
            print(f"{case_name[:38]:40s} {mean_r:.4f}{'':11s} {max_r:.4f}{'':8s} {high_pct:.0f}%{'':7s} {expected[:18]:20s}")
        print("=" * 100)

    # Print summary
    if df is not None and len(df) > 0:
        print_summary_table(df)
        print("\n--- Generating radar charts ---")
        generate_radar_charts(df)

    # Matching statistics
    print("\n--- Matching Statistics ---")
    herb_count = sum(1 for r in all_results if r["type"] == "herb_compound")
    drug_count = sum(1 for r in all_results if r["type"] == "interacting_drug")
    print(f"  Herb compounds mapped & predicted: {herb_count}")
    print(f"  Interacting drugs mapped & predicted: {drug_count}")
    print(f"  Total Stage 1 predictions: {len(all_results)}")
    if ddi_results:
        print(f"  Total Stage 2 DDI predictions: {len(ddi_results)}")

    print("\nDONE. Case study analysis complete.")


if __name__ == "__main__":
    main()