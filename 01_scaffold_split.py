"""
Phase 2.1 — 数据集划分 (Scaffold Split 80/20)
==============================================
功能:
  1. 读取去重化合物 + SMILES + 特征 + 标签
  2. 基于 Murcko scaffold 进行 80/20 划分 (保持化学多样性)
  3. 合并 MACCS 指纹 (all_features 目前缺 MACCS)
  4. 按终点统计正负样本比例
  5. 输出: 训练集/测试集特征 + 标签 + 划分元数据

输入:
  - data/02_processed/phase1/step2_deduplicated/compounds_deduplicated.csv
  - data/02_processed/phase1/step3_features/all_features.csv
  - data/02_processed/phase1/step3_features/maccs_fingerprints.csv
  - data/02_processed/phase1/step5_labels/tox21_labels.csv
  - data/02_processed/phase1/step5_labels/ctd_labels.csv

输出:
  - data/02_processed/phase2/split/
    ├── train_features.csv      训练集特征 (含 ID, SMILES)
    ├── test_features.csv       测试集特征
    ├── tox21_train_labels.csv  Tox21 训练标签
    ├── tox21_test_labels.csv   Tox21 测试标签
    ├── ctd_train_labels.csv    CTD 训练标签
    ├── ctd_test_labels.csv     CTD 测试标签
    └── split_meta.json         划分元数据 & 统计
"""

import os, json, sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── 路径 ──────────────────────────────────────────────
# __file__ = code/phase2/01_scaffold_split.py → 往上两层到项目根目录
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "02_processed", "phase1"))
OUT  = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "02_processed", "phase2", "split"))

COMPOUNDS_FILE = os.path.join(ROOT, "step2_deduplicated", "compounds_deduplicated.csv")
FEATURES_FILE  = os.path.join(ROOT, "step3_features", "all_features.csv")
MACCS_FILE     = os.path.join(ROOT, "step3_features", "maccs_fingerprints.csv")
TOX21_FILE     = os.path.join(ROOT, "step5_labels", "tox21_labels.csv")
# 使用有 SMILES 的 CTD 文件 (1919 条), 通过 SMILES 匹配而非 chemical_id
CTD_FILE       = os.path.join(ROOT, "ctd_processed", "ctd_stage1_with_smiles.csv")

SEED   = 42
RATIO  = 0.8   # 训练集比例


def get_scaffold(smiles):
    """从 SMILES 提取 Murcko scaffold; 失败返回空串."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=scaf)
    except Exception:
        return ""


def scaffold_split(ids, smiles_list, ratio=0.8, seed=42):
    """
    基于 scaffold 的划分:
      - 同一 scaffold 的化合物只出现在 train 或 test 之一
      - 按 scaffold 组大小降序, 贪心分配到较小的集合
    返回: (train_indices, test_indices)
    """
    scaffold_map = {}
    no_scaffold = []
    for i, smi in enumerate(smiles_list):
        scaf = get_scaffold(smi)
        if scaf:
            scaffold_map.setdefault(scaf, []).append(i)
        else:
            no_scaffold.append(i)

    scaffold_sets = sorted(scaffold_map.values(), key=len, reverse=True)

    rng = np.random.RandomState(seed)
    n_total = len(ids)
    n_train_target = int(n_total * ratio)

    train_idx, test_idx = [], []

    for sset in scaffold_sets:
        if len(train_idx) < n_train_target:
            train_idx.extend(sset)
        else:
            test_idx.extend(sset)

    rng.shuffle(no_scaffold)
    split_point = max(0, n_train_target - len(train_idx))
    train_idx.extend(no_scaffold[:split_point])
    test_idx.extend(no_scaffold[split_point:])

    return sorted(train_idx), sorted(test_idx)


def label_stats(labels_df, endpoints):
    """统计每个终点的正/负/缺失样本数."""
    stats = {}
    for ep in endpoints:
        col = labels_df[ep]
        pos = int((col == 1).sum())
        neg = int((col == 0).sum())
        na  = int(col.isna().sum())
        total = pos + neg
        stats[ep] = {
            "positive": pos, "negative": neg, "missing": na,
            "total": total,
            "pos_ratio": round(pos / total, 4) if total > 0 else 0
        }
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ── 1. 读取数据 ──────────────────────────────────
    print("[1/7] 读取化合物信息 ...")
    compounds = pd.read_csv(COMPOUNDS_FILE)
    smiles_col = "SMILES_standardized" if "SMILES_standardized" in compounds.columns else "SMILES"
    comp_smiles = compounds[["ID", smiles_col]].copy()
    comp_smiles.columns = ["ID", "SMILES"]
    # 保留原始 SMILES 用于 CTD 匹配 (CTD 的 SMILES 未经标准化)
    if "SMILES" in compounds.columns and smiles_col != "SMILES":
        comp_smiles["SMILES_raw"] = compounds["SMILES"].values
    else:
        comp_smiles["SMILES_raw"] = comp_smiles["SMILES"].values
    print(f"  化合物数: {len(comp_smiles)}")

    print("[2/7] 读取分子特征 (all_features + MACCS) ...")
    features = pd.read_csv(FEATURES_FILE)
    maccs    = pd.read_csv(MACCS_FILE)

    id_col_name = "ID" if "ID" in features.columns else features.columns[19]
    print(f"  all_features: {features.shape}, ID列='{id_col_name}'")
    print(f"  MACCS: {maccs.shape}")

    # 合并: 将 ID 列移到最前, 再拼接 MACCS (去掉 MACCS 的 ID 列避免重复)
    feat_cols = [c for c in features.columns if c != id_col_name]
    features = features[[id_col_name] + feat_cols]
    maccs_cols = [c for c in maccs.columns if c != id_col_name]
    maccs = maccs[maccs_cols]
    full_features = pd.concat([features.reset_index(drop=True), maccs.reset_index(drop=True)], axis=1)
    print(f"  合并后: {full_features.shape}")

    print("[3/7] 读取标签 ...")
    tox21_labels = pd.read_csv(TOX21_FILE)
    ctd_labels   = pd.read_csv(CTD_FILE)
    tox21_endpoints = [c for c in tox21_labels.columns if c != "SMILES"]
    ctd_endpoints   = [c for c in ctd_labels.columns if c not in ("chemical_id", "chemical_name", "cas_rn", "smiles")]
    print(f"  Tox21: {len(tox21_labels)} 行, {len(tox21_endpoints)} 终点")
    print(f"  CTD:   {len(ctd_labels)} 行, {len(ctd_endpoints)} 终点")

    # ── 2. Scaffold Split ──────────────────────────────
    print("[4/7] 执行 scaffold split (这可能需要几分钟) ...")
    all_ids    = comp_smiles["ID"].values
    all_smiles = comp_smiles["SMILES"].values

    train_idx, test_idx = scaffold_split(all_ids, all_smiles, ratio=RATIO, seed=SEED)
    print(f"  训练集: {len(train_idx)} ({len(train_idx)/len(all_ids)*100:.1f}%)")
    print(f"  测试集: {len(test_idx)} ({len(test_idx)/len(all_ids)*100:.1f}%)")

    # 验证 scaffold 无泄露
    train_scaffolds = set()
    test_scaffolds  = set()
    for i in train_idx:
        s = get_scaffold(all_smiles[i])
        if s: train_scaffolds.add(s)
    for i in test_idx:
        s = get_scaffold(all_smiles[i])
        if s: test_scaffolds.add(s)
    overlap = train_scaffolds & test_scaffolds
    print(f"  训练集 scaffolds: {len(train_scaffolds)}")
    print(f"  测试集 scaffolds: {len(test_scaffolds)}")
    print(f"  scaffold 重叠: {len(overlap)}")

    # ── 3. 构建训练/测试特征 ──────────────────────────
    print("[5/7] 构建训练/测试特征集 ...")
    train_ids_set = set(all_ids[train_idx])
    test_ids_set  = set(all_ids[test_idx])

    train_feat = full_features[full_features[id_col_name].isin(train_ids_set)].copy()
    test_feat  = full_features[full_features[id_col_name].isin(test_ids_set)].copy()

    smiles_map = dict(zip(comp_smiles["ID"], comp_smiles["SMILES"]))
    smiles_raw_map = dict(zip(comp_smiles["ID"], comp_smiles["SMILES_raw"]))
    train_feat.insert(1, "SMILES", train_feat[id_col_name].map(smiles_map))
    test_feat.insert(1, "SMILES", test_feat[id_col_name].map(smiles_map))
    # SMILES_raw 仅用于 CTD 匹配，不写入输出文件
    train_smiles_raw_map = dict(zip(train_feat[id_col_name], train_feat[id_col_name].map(smiles_raw_map)))
    test_smiles_raw_map  = dict(zip(test_feat[id_col_name], test_feat[id_col_name].map(smiles_raw_map)))
    print(f"  训练特征: {train_feat.shape}")
    print(f"  测试特征: {test_feat.shape}")

    # ── 4. 构建训练/测试标签 ──────────────────────────
    print("[6/7] 构建训练/测试标签 ...")

    # Tox21 标签用 SMILES 标识
    train_smiles_set = set(train_feat["SMILES"])
    test_smiles_set  = set(test_feat["SMILES"])
    tox21_train = tox21_labels[tox21_labels["SMILES"].isin(train_smiles_set)].copy()
    tox21_test  = tox21_labels[tox21_labels["SMILES"].isin(test_smiles_set)].copy()
    print(f"  Tox21 训练标签: {tox21_train.shape}")
    print(f"  Tox21 测试标签: {tox21_test.shape}")

    # CTD 标签用 SMILES 标识 (ctd_stage1_with_smiles.csv)
    # 优先用原始 SMILES 直接匹配; 未匹配的用 RDKit canonical 兜底
    train_smiles_raw_set = set(train_smiles_raw_map.values())
    test_smiles_raw_set  = set(test_smiles_raw_map.values())
    ctd_matched_train = ctd_labels[ctd_labels["smiles"].isin(train_smiles_raw_set)].copy()
    ctd_matched_test  = ctd_labels[ctd_labels["smiles"].isin(test_smiles_raw_set)].copy()

    # RDKit canonical 兜底: 对未匹配的 CTD 条目做 canonical 化再匹配
    ctd_unmatched = ctd_labels[~ctd_labels.index.isin(ctd_matched_train.index) & ~ctd_labels.index.isin(ctd_matched_test.index)]
    if len(ctd_unmatched) > 0:
        # 构建 canonical → raw SMILES 的映射
        train_canonical_map = {}
        test_canonical_map = {}
        for raw_smi in train_smiles_raw_set:
            mol = Chem.MolFromSmiles(raw_smi)
            if mol:
                train_canonical_map.setdefault(Chem.MolToSmiles(mol), set()).add(raw_smi)
        for raw_smi in test_smiles_raw_set:
            mol = Chem.MolFromSmiles(raw_smi)
            if mol:
                test_canonical_map.setdefault(Chem.MolToSmiles(mol), set()).add(raw_smi)

        extra_train, extra_test = [], []
        for _, row in ctd_unmatched.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                canon = Chem.MolToSmiles(mol)
                if canon in train_canonical_map:
                    extra_train.append(row)
                elif canon in test_canonical_map:
                    extra_test.append(row)

        if extra_train:
            ctd_matched_train = pd.concat([ctd_matched_train, pd.DataFrame(extra_train)], ignore_index=True)
        if extra_test:
            ctd_matched_test = pd.concat([ctd_matched_test, pd.DataFrame(extra_test)], ignore_index=True)

    ctd_train = ctd_matched_train.drop_duplicates(subset="smiles").copy()
    ctd_test  = ctd_matched_test.drop_duplicates(subset="smiles").copy()
    print(f"  CTD 训练标签: {ctd_train.shape}")
    print(f"  CTD 测试标签: {ctd_test.shape}")

    # ── 5. 统计 ──────────────────────────────────────
    print("[7/7] 统计标签分布 ...")
    tox21_stats_train = label_stats(tox21_train, tox21_endpoints)
    tox21_stats_test  = label_stats(tox21_test, tox21_endpoints)
    ctd_stats_train   = label_stats(ctd_train, ctd_endpoints)
    ctd_stats_test    = label_stats(ctd_test, ctd_endpoints)

    print("\n  Tox21 训练集:")
    for ep, s in tox21_stats_train.items():
        print(f"    {ep}: pos={s['positive']}, neg={s['negative']}, ratio={s['pos_ratio']:.3f}")

    print("\n  CTD 训练集:")
    for ep, s in ctd_stats_train.items():
        print(f"    {ep}: pos={s['positive']}, neg={s['negative']}, ratio={s['pos_ratio']:.3f}")

    # ── 6. 保存 ──────────────────────────────────────
    print("\n保存文件 ...")
    # 保存特征时去掉 SMILES_raw 列 (仅用于 CTD 匹配)
    save_cols = [c for c in train_feat.columns if c != "SMILES_raw"]
    train_feat[save_cols].to_csv(os.path.join(OUT, "train_features.csv"), index=False)
    test_feat[save_cols].to_csv(os.path.join(OUT, "test_features.csv"), index=False)
    tox21_train.to_csv(os.path.join(OUT, "tox21_train_labels.csv"), index=False)
    tox21_test.to_csv(os.path.join(OUT, "tox21_test_labels.csv"), index=False)
    ctd_train.to_csv(os.path.join(OUT, "ctd_train_labels.csv"), index=False)
    ctd_test.to_csv(os.path.join(OUT, "ctd_test_labels.csv"), index=False)

    # Tox21 assay 语义名称 (方便论文中引用)
    tox21_assay_names = {
        "Tox21_0": "NR.AhR",
        "Tox21_1": "NR.AR",
        "Tox21_2": "NR.AR.LBD",
        "Tox21_3": "NR.Aromatase",
        "Tox21_4": "NR.ER",
        "Tox21_5": "NR.ER.LBD",
        "Tox21_6": "NR.PPAR.gamma",
        "Tox21_7": "SR.ARE",
        "Tox21_8": "SR.ATAD5",
        "Tox21_9": "SR.HSE",
        "Tox21_10": "SR.MMP",
        "Tox21_11": "SR.p53"
    }

    # 标记无正样本的终点
    no_positive_endpoints = []
    for ep, s in tox21_stats_train.items():
        if s["positive"] == 0:
            no_positive_endpoints.append(ep)

    meta = {
        "seed": SEED,
        "ratio": RATIO,
        "total_compounds": int(len(all_ids)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_pct": round(len(train_idx) / len(all_ids) * 100, 1),
        "test_pct": round(len(test_idx) / len(all_ids) * 100, 1),
        "scaffold_overlap_count": int(len(overlap)),
        "train_scaffold_count": int(len(train_scaffolds)),
        "test_scaffold_count": int(len(test_scaffolds)),
        "feature_dim": 1210,
        "feature_columns": [c for c in full_features.columns if c not in (id_col_name, "SMILES", "SMILES_raw")],
        "feature_composition": {
            "molecular_descriptors": 19,
            "morgan_fingerprints": 1024,
            "maccs_fingerprints": 167,
            "total": 1210
        },
        "tox21": {
            "endpoints": tox21_endpoints,
            "assay_names": tox21_assay_names,
            "train_size": int(len(tox21_train)),
            "test_size": int(len(tox21_test)),
            "train_stats": tox21_stats_train,
            "test_stats": tox21_stats_test
        },
        "ctd": {
            "endpoints": ctd_endpoints,
            "train_size": int(len(ctd_train)),
            "test_size": int(len(ctd_test)),
            "train_stats": ctd_stats_train,
            "test_stats": ctd_stats_test
        },
        "warnings": {
            "no_positive_train_endpoints": no_positive_endpoints,
            "low_positive_endpoints": [
                ep for ep, s in {**tox21_stats_train, **ctd_stats_train}.items()
                if 0 < s.get("positive", 0) / max(s.get("total", 1), 1) < 0.01
            ]
        },
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    with open(os.path.join(OUT, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n完成! 耗时 {time.time()-t0:.1f}s")
    print(f"输出目录: {OUT}")
    for fname in sorted(os.listdir(OUT)):
        fpath = os.path.join(OUT, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size/1024:.0f} KB")


if __name__ == "__main__":
    main()
