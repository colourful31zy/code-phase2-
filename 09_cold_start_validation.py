"""
09_cold_start_validation.py — Unseen-drug (cold-start) generalization test for Stage 2

Concept
-------
审稿人最常问 DDI 预测论文的一个问题：模型在"训练时完全没见过的化合物"
组成的药物对上表现如何？这就是 cold-start / unseen-drug 泛化。

设计
----
1. 从 DrugBank DDI 正例 + 随机负例构建药物对（与 05_stage2_ddi.py 相同）。
2. 收集全部涉及化合物 ID，按 ID 随机抽取 20% 作为"未见药物"集合 C_unseen。
3. 训练集 = 不含任何 C_unseen 化合物的药物对（其余 80% 化合物）。
4. 测试集 = 两侧都来自 C_unseen 的全新药物对（正负平衡，各取等量）。
   注意：测试集中没有化合物在训练集里出现过 → 真正的 cold-start。
5. 训练 LightGBM（与 05_stage2_ddi.py 相同的 2462 维特征），在测试集上评估
   AUC / AUPRC。
6. 对比三个数字：
   - 普通划分（GroupKFold）训练 AUC：约 0.985（任务偏易）
   - cold-start 测试 AUC：全新化合物的真实泛化
   - DDID 外部验证 AUC：0.745（真实草药交互）
   三个数字一起报 → 论文里直接回应"未见药物泛化"质疑。

输出
----
data/02_processed/phase2/results/cold_start_validation/
  - cold_start_results.csv   (逐对预测)
  - cold_start_summary.txt   (三类对比数字)
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SEED = 42
UNSEEN_FRAC = 0.20          # 20% 化合物作为 unseen
N_POS_PAIRS = 50000         # 正例对上限（同 05）
NEG_POS_RATIO = 1.0
MAX_NEG_ATTEMPTS = 2000000

# Paths
PHASE1 = config.PHASE1_DIR
SPLIT_DIR = config.SPLIT_DIR
RESULTS_DIR = config.STAGE2_RESULTS
OUT_DIR = os.path.join(config.PROJECT_ROOT, "data", "02_processed", "phase2",
                       "results", "cold_start_validation")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(PHASE1, "step3_features", "all_features.csv")
DRUGBANK_DDI_FILE = os.path.join(PHASE1, "step4_ddi_integrated", "ddi_drugbank.csv")
TOX_PROFILE_FILE = os.path.join(RESULTS_DIR, "stage1_toxicity_profiles.csv")


def log(msg):
    print(f"  {msg}")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════
# 1. Build drug pairs (same as 05_stage2_ddi.py)
# ══════════════════════════════════════════
def build_pairs():
    print_header("Step 1: Build drug pairs")
    df_feat = pd.read_csv(FEATURES_FILE, usecols=["ID"])
    all_ids = df_feat["ID"].values
    all_ids_set = set(all_ids)
    n_compounds = len(all_ids)
    log(f"Feature space: {n_compounds} compounds")

    ddi = pd.read_csv(DRUGBANK_DDI_FILE)
    log(f"Raw DrugBank DDI: {len(ddi)}")

    in_a = ddi["Drug1_ID"].isin(all_ids_set)
    in_b = ddi["Drug2_ID"].isin(all_ids_set)
    valid = ddi[in_a & in_b].copy()
    log(f"Both in feature space: {len(valid)}")

    valid["_pair_key"] = valid.apply(
        lambda r: tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])), axis=1)
    valid = valid.drop_duplicates(subset="_pair_key").reset_index(drop=True)
    log(f"After dedup: {len(valid)}")

    n_pos = min(len(valid), N_POS_PAIRS)
    if len(valid) > N_POS_PAIRS:
        valid = valid.sample(n=N_POS_PAIRS, random_state=SEED)

    pos_set = set()
    for _, r in valid.iterrows():
        pos_set.add(tuple(sorted([r["Drug1_ID"], r["Drug2_ID"]])))

    pos_df = pd.DataFrame({
        "DrugA_ID": valid["Drug1_ID"].values,
        "DrugB_ID": valid["Drug2_ID"].values,
        "label": np.ones(n_pos, dtype=np.int32),
    })
    a, b = pos_df["DrugA_ID"].values, pos_df["DrugB_ID"].values
    sw = a > b
    if sw.any():
        pos_df["DrugA_ID"] = np.where(sw, b, a)
        pos_df["DrugB_ID"] = np.where(sw, a, b)

    n_neg = int(n_pos * NEG_POS_RATIO)
    np.random.seed(SEED)
    neg_list = []
    attempts = 0
    while len(neg_list) < n_neg and attempts < MAX_NEG_ATTEMPTS:
        batch = min(MAX_NEG_ATTEMPTS, (n_neg - len(neg_list)) * 10)
        ia = np.random.randint(0, n_compounds, batch)
        ib = np.random.randint(0, n_compounds, batch)
        for x, y in zip(ia, ib):
            if x == y:
                continue
            key = tuple(sorted([all_ids[x], all_ids[y]]))
            if key not in pos_set:
                neg_list.append(key)
                attempts += 1
                if len(neg_list) >= n_neg:
                    break
    neg_arr = np.array(neg_list[:n_neg])
    neg_df = pd.DataFrame({
        "DrugA_ID": neg_arr[:, 0],
        "DrugB_ID": neg_arr[:, 1],
        "label": np.zeros(n_neg, dtype=np.int32),
    })
    a, b = neg_df["DrugA_ID"].values, neg_df["DrugB_ID"].values
    sw = a > b
    if sw.any():
        neg_df["DrugA_ID"] = np.where(sw, b, a)
        neg_df["DrugB_ID"] = np.where(sw, a, b)

    pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    pairs = pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
    log(f"Total pairs: {len(pairs)} ({n_pos} pos, {n_neg} neg)")
    return pairs, all_ids


# ══════════════════════════════════════════
# 2. Build features (same as 05_stage2_ddi.py)
# ══════════════════════════════════════════
def build_features(pairs):
    print_header("Step 2: Build 2462-dim features")
    with open(os.path.join(SPLIT_DIR, "split_meta.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train_features.csv"),
                     usecols=["ID"] + feature_cols)
    te = pd.read_csv(os.path.join(SPLIT_DIR, "test_features.csv"),
                     usecols=["ID"] + feature_cols)
    df_mol = pd.concat([tr, te], ignore_index=True)

    tox_df = pd.read_csv(TOX_PROFILE_FILE)
    tox_cols = [c for c in tox_df.columns if c.startswith("tox_")]
    tox_flat = tox_df.set_index("ID")

    m_a = pairs.merge(df_mol, left_on="DrugA_ID", right_on="ID", how="left")
    m_b = pairs.merge(df_mol, left_on="DrugB_ID", right_on="ID", how="left")

    mol_a = m_a[feature_cols].fillna(0).values.astype(np.float32)
    mol_b = m_b[feature_cols].fillna(0).values.astype(np.float32)

    tox_a = m_a["DrugA_ID"].map(lambda x:
        tox_flat.loc[x, tox_cols].values.astype(np.float32)
        if x in tox_flat.index else np.zeros(len(tox_cols), dtype=np.float32))
    tox_b = m_b["DrugB_ID"].map(lambda x:
        tox_flat.loc[x, tox_cols].values.astype(np.float32)
        if x in tox_flat.index else np.zeros(len(tox_cols), dtype=np.float32))
    tox_a = np.stack(tox_a.values).astype(np.float32)
    tox_b = np.stack(tox_b.values).astype(np.float32)

    X = np.concatenate([
        tox_a, tox_b,
        np.abs(mol_a - mol_b).astype(np.float32),
        (mol_a * mol_b).astype(np.float32),
    ], axis=1)
    y = pairs["label"].values.astype(np.float32)

    fnames = ([f"ToxA_{c}" for c in tox_cols] + [f"ToxB_{c}" for c in tox_cols] +
              [f"Diff_{c}" for c in feature_cols] + [f"Prod_{c}" for c in feature_cols])
    log(f"X: {X.shape}, y pos rate: {y.mean():.4f}")
    return X, y, fnames


# ══════════════════════════════════════════
# 3. Unseen-drug split
# ══════════════════════════════════════════
def cold_start_split(pairs, all_ids, frac=UNSEEN_FRAC):
    print_header(f"Step 3: Unseen-drug split (frac={frac})")
    # All compounds involved in pairs
    involved = set(pairs["DrugA_ID"]) | set(pairs["DrugB_ID"])
    involved_arr = np.array(sorted(involved))
    np.random.seed(SEED)
    n_unseen = int(len(involved_arr) * frac)
    unseen_idx = np.random.choice(len(involved_arr), n_unseen, replace=False)
    unseen = set(involved_arr[unseen_idx])
    log(f"Compounds involved: {len(involved)}, unseen: {n_unseen} ({len(unseen)/len(involved)*100:.1f}%)")

    # Train: pairs with NO unseen compound
    train_mask = ~(pairs["DrugA_ID"].isin(unseen) | pairs["DrugB_ID"].isin(unseen))
    # Test: pairs where BOTH sides are unseen
    test_mask = pairs["DrugA_ID"].isin(unseen) & pairs["DrugB_ID"].isin(unseen)
    log(f"Train pairs: {train_mask.sum()}, test (both unseen) pairs: {test_mask.sum()}")

    train_pairs = pairs[train_mask].reset_index(drop=True)
    test_pairs = pairs[test_mask].reset_index(drop=True)
    return train_pairs, test_pairs, unseen


# ══════════════════════════════════════════
# 4. Train + evaluate
# ══════════════════════════════════════════
def run():
    t0 = time.time()
    pairs, all_ids = build_pairs()
    train_pairs, test_pairs, unseen = cold_start_split(pairs, all_ids)

    Xtr, ytr, fnames = build_features(train_pairs)
    Xte, yte, _ = build_features(test_pairs)

    n_pos_t, n_neg_t = int(np.sum(ytr == 1)), int(np.sum(ytr == 0))
    model = LGBMClassifier(scale_pos_weight=n_neg_t / max(n_pos_t, 1),
                           verbose=-1, n_jobs=-1, random_state=SEED)
    model.fit(Xtr, ytr)
    train_auc = roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1])
    log(f"Train AUC (in-sample): {train_auc:.4f}")

    y_prob_te = model.predict_proba(Xte)[:, 1]
    te_auc = roc_auc_score(yte, y_prob_te)
    te_auprc = average_precision_score(yte, y_prob_te)
    log(f"Cold-start test AUC: {te_auc:.4f}")
    log(f"Cold-start test AUPRC: {te_auprc:.4f}")

    # Save predictions
    pred_df = test_pairs.copy()
    pred_df["ddi_risk_probability"] = y_prob_te
    pred_df.to_csv(os.path.join(OUT_DIR, "cold_start_predictions.csv"), index=False)
    log(f"Saved predictions: cold_start_predictions.csv ({len(pred_df)} rows)")

    # Summary
    summary = f"""Cold-start (unseen-drug) validation summary
==========================================
Split: {UNSEEN_FRAC*100:.0f}% compounds held out as unseen ({len(unseen)} of {len(set(pairs['DrugA_ID'])|set(pairs['DrugB_ID']))} involved)

In-sample train AUC (LightGBM, same 2462-dim features): {train_auc:.4f}
COLD-START test AUC (both molecules unseen):            {te_auc:.4f}
COLD-START test AUPRC:                                  {te_auprc:.4f}
Test set: {len(test_pairs)} pairs ({int(np.sum(yte==1))} pos, {int(np.sum(yte==0))} neg)

Reference numbers from the paper:
- GroupKFold in-sample AUC: 0.9853 ± 0.0014
- DDID external validation AUC: 0.7451
"""
    with open(os.path.join(OUT_DIR, "cold_start_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print_header("Cold-start summary")
    print(summary)
    log(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
