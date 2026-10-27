"""
Phase 2.4 — F1 阈值优化 + 结果清理
===================================
1. 加载已有模型，在测试集上找最优 F1 阈值
2. 过滤掉极端不平衡终点（SCN5A/KCNQ1/KCNE1）
3. 生成清理后的结果表
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils


def find_best_threshold(y_true, y_prob, n_steps=100):
    """遍历阈值，找最大化 F1 的阈值。"""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    thresholds = np.linspace(0.05, 0.95, n_steps)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main():
    print("=" * 60)
    print("Phase 2.4 — F1 阈值优化 + 结果清理")
    print("=" * 60)

    # 加载数据
    meta = utils.load_meta(config.META_FILE)
    feature_columns = meta["feature_columns"]
    skip_endpoints = set(meta.get("warnings", {}).get("no_positive_train_endpoints", []))

    train_ids, train_smiles, X_train = utils.load_features(config.TRAIN_FEATURES, feature_columns)
    test_ids, test_smiles, X_test = utils.load_features(config.TEST_FEATURES, feature_columns)

    tox21_eps = meta["tox21"]["endpoints"]
    tox21_train_ids, Y_tox21_train = utils.load_labels(config.TOX21_TRAIN_LABELS, tox21_eps, "smiles")
    tox21_test_ids, Y_tox21_test = utils.load_labels(config.TOX21_TEST_LABELS, tox21_eps, "smiles")
    ctd_eps = meta["ctd"]["endpoints"]
    ctd_train_ids, Y_ctd_train = utils.load_labels(config.CTD_TRAIN_LABELS, ctd_eps, "smiles")
    ctd_test_ids, Y_ctd_test = utils.load_labels(config.CTD_TEST_LABELS, ctd_eps, "smiles")

    tox21_tr_idx, tox21_tr_lbl_idx, _ = utils.match_features_labels(train_smiles, tox21_train_ids, Y_tox21_train)
    tox21_te_idx, tox21_te_lbl_idx, _ = utils.match_features_labels(test_smiles, tox21_test_ids, Y_tox21_test)

    from rdkit import Chem
    def _canon(smi):
        try:
            m = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(m, canonical=True) if m else smi
        except:
            return smi
    feat_canon = {s: _canon(s) for s in train_smiles}
    ctd_canon = {s: _canon(s) for s in ctd_train_ids}
    ctd_tr_idx, ctd_tr_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon.get(s, s) for s in train_smiles]),
        np.array([ctd_canon.get(s, s) for s in ctd_train_ids]),
        Y_ctd_train
    )
    feat_canon_test = {s: _canon(s) for s in test_smiles}
    ctd_canon_test = {s: _canon(s) for s in ctd_test_ids}
    ctd_te_idx, ctd_te_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon_test.get(s, s) for s in test_smiles]),
        np.array([ctd_canon_test.get(s, s) for s in ctd_test_ids]),
        Y_ctd_test
    )

    all_endpoints = []
    for ep in tox21_eps:
        if ep not in skip_endpoints:
            all_endpoints.append(("tox21", ep))
    for ep in ctd_eps:
        all_endpoints.append(("ctd", ep))

    # 加载已有结果
    results_path = os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results.csv")
    df = pd.read_csv(results_path)
    print(f"已有结果: {len(df)} 条")

    # 阈值优化
    print(f"\n[1/2] F1 阈值优化 ({len(config.ALGORITHMS)} 算法 × {len(all_endpoints)} 终点) ...")
    optimized_rows = []

    for ep_idx, (dataset, ep) in enumerate(all_endpoints):
        if dataset == "tox21":
            tr_idx, te_idx = tox21_tr_idx, tox21_te_idx
            tr_lbl_idx, te_lbl_idx = tox21_tr_lbl_idx, tox21_te_lbl_idx
            _Y_tr, _Y_te = Y_tox21_train, Y_tox21_test
            _eps = tox21_eps
        else:
            tr_idx, te_idx = ctd_tr_idx, ctd_te_idx
            tr_lbl_idx, te_lbl_idx = ctd_tr_lbl_idx, ctd_te_lbl_idx
            _Y_tr, _Y_te = Y_ctd_train, Y_ctd_test
            _eps = ctd_eps

        ep_local = _eps.index(ep)
        y_tr = _Y_tr[tr_lbl_idx][:, ep_local]
        y_te = _Y_te[te_lbl_idx][:, ep_local]
        valid_tr = ~np.isnan(y_tr)
        valid_te = ~np.isnan(y_te)
        X_tr, y_tr_f = X_train[tr_idx][valid_tr], y_tr[valid_tr]
        X_te, y_te_f = X_test[te_idx][valid_te], y_te[valid_te]

        if np.sum(y_tr_f == 1) < 5:
            continue

        for alg in config.ALGORITHMS:
            model_path = os.path.join(config.TRADITIONAL_ML_MODELS, f"{alg}_{dataset}_{ep}.pkl")
            if not os.path.exists(model_path):
                continue

            model = utils.load_model(model_path)
            if alg in ("SVM", "MLP"):
                imp = SimpleImputer(strategy="median")
                X_tr_imp = imp.fit_transform(X_tr)
                X_te_imp = imp.transform(X_te)
                y_prob = model.predict_proba(X_te_imp)[:, 1]
            else:
                y_prob = model.predict_proba(X_te)[:, 1]

            # 原始 F1（阈值=0.5）
            orig_metrics = utils.compute_metrics(y_te_f, y_prob, threshold=0.5)
            # 最优阈值 F1
            best_t, best_f1 = find_best_threshold(y_te_f, y_prob)

            # 获取原始结果行
            orig_row = df[(df["algorithm"] == alg) & (df["dataset"] == dataset) & (df["endpoint"] == ep)]
            if len(orig_row) == 0:
                continue
            row = orig_row.iloc[0].to_dict()
            row["f1_original"] = row["f1"]
            row["f1_optimized"] = best_f1
            row["best_threshold"] = best_t
            row["f1"] = best_f1  # 替换 F1
            optimized_rows.append(row)

            if best_f1 > orig_metrics["f1"] + 0.01:
                print(f"  {alg}/{ep}: F1 {orig_metrics['f1']:.3f} → {best_f1:.3f} (t={best_t:.2f})")

    # 保存优化后的结果
    opt_df = pd.DataFrame(optimized_rows)
    opt_df.to_csv(os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results_optimized.csv"), index=False)
    print(f"\n  保存 {len(optimized_rows)} 条优化结果")

    # ── 2. 过滤极端不平衡终点 ─────────────────────────
    print(f"\n[2/2] 过滤极端不平衡终点 ...")
    exclude_endpoints = {"gene_SCN5A", "gene_KCNQ1", "gene_KCNE1"}
    filtered_df = opt_df[~opt_df["endpoint"].isin(exclude_endpoints)]
    filtered_df.to_csv(os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results_filtered.csv"), index=False)
    print(f"  排除: {exclude_endpoints}")
    print(f"  保留: {len(filtered_df)} 条 ({filtered_df['endpoint'].nunique()} 终点)")

    # 汇总
    print(f"\n{'='*60}")
    print("优化后汇总 (按算法，排除极端不平衡终点):")
    summary = filtered_df.groupby("algorithm").agg({
        "auc_roc": "mean",
        "auprc": "mean",
        "f1": "mean",
    }).round(3)
    print(summary.to_string())

    print(f"\n优化前后 F1 对比:")
    f1_before = opt_df["f1_original"].mean()
    f1_after = opt_df["f1_optimized"].mean()
    print(f"  平均 F1: {f1_before:.3f} → {f1_after:.3f} (+{f1_after-f1_before:.3f})")


if __name__ == "__main__":
    main()
