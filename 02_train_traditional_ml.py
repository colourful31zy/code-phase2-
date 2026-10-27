"""
Phase 2.2 — 传统 ML 训练 (5 算法 × 26 终点)
==============================================
功能:
  1. 加载 scaffold split 后的特征和标签
  2. 对每个 (算法, 终点) 组合做 Optuna 调参 (50 trials, 5-fold CV)
  3. 用最优参数训练最终模型，在测试集上评估
  4. NB 集成 (5 算法预测 → NB 元分类器)
  5. Y-randomization (100 次)

输出:
  - models/traditional_ml/{algorithm}_{endpoint}.pkl
  - results/evaluation/traditional_ml_results.csv
  - results/evaluation/nb_ensemble_results.csv
  - results/evaluation/y_randomization_results.csv
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 导入配置和工具 ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils


def train_single(algorithm, endpoint, X_train, y_train, X_test, y_test):
    """
    训练单个 (算法, 终点)：
      1. Optuna 调参 (5-fold CV，同时收集 OOF 预测)
      2. 用最优参数训练最终模型
      3. 在测试集上评估
    返回: (metrics_dict, best_params, model, y_prob_test, oof_train)
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    # Optuna 调参 —— 同时收集 OOF 预测（用于 NB 集成）
    oof_train = np.full(len(y_train), np.nan)
    skf_oof = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)
    fold_indices = list(skf_oof.split(X_train, y_train))

    # SVM 子采样：训练集 >3000 时采样加速调参
    SVM_SUBSAMPLE = 3000
    if algorithm == "SVM" and len(X_train) > SVM_SUBSAMPLE:
        rng = np.random.RandomState(config.SEED)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos_sample = min(len(pos_idx), SVM_SUBSAMPLE // 3)
        n_neg_sample = SVM_SUBSAMPLE - n_pos_sample
        sampled_idx = np.concatenate([
            rng.choice(pos_idx, n_pos_sample, replace=False),
            rng.choice(neg_idx, min(n_neg_sample, len(neg_idx)), replace=False),
        ])
        X_tune, y_tune = X_train[sampled_idx], y_train[sampled_idx]
    else:
        X_tune, y_tune = X_train, y_train

    # SVM/MLP 不支持 NaN，需要填充
    NEED_IMPUTE = algorithm in ("SVM", "MLP")
    if NEED_IMPUTE:
        from sklearn.impute import SimpleImputer
        _imp = SimpleImputer(strategy="median")
        X_tune_imp = _imp.fit_transform(X_tune)
    else:
        X_tune_imp = X_tune

    # SVM 子采样时 OOF 无法映射回全量训练集，跳过
    COLLECT_OOF = not (algorithm == "SVM" and len(X_train) > SVM_SUBSAMPLE)

    def _objective(trial):
        params = {}
        for name, sampler in config.OPTUNA_SPACES[algorithm].items():
            if name == "dropout":
                continue
            params[name] = sampler(trial)
        model = utils.build_model(algorithm, params, n_pos=n_pos, n_neg=n_neg)
        auprc_scores = []
        skf_tune = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)
        for fi, (tr_idx, va_idx) in enumerate(skf_tune.split(X_tune_imp, y_tune)):
            X_tr, X_va = X_tune_imp[tr_idx], X_tune_imp[va_idx]
            y_tr, y_va = y_tune[tr_idx], y_tune[va_idx]
            sw = utils.get_sample_weight(y_tr, algorithm)
            if sw is not None:
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_va)[:, 1]
            auprc_scores.append(average_precision_score(y_va, y_prob))
            if trial.number == 0 and COLLECT_OOF:
                oof_train[va_idx] = y_prob
        return np.mean(auprc_scores)

    study = optuna.create_study(direction="maximize", study_name=f"{algorithm}_{endpoint}")
    study.optimize(_objective, n_trials=config.N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_cv_auprc = study.best_value

    # 用最优参数训练最终模型（全量训练集）
    model = utils.build_model(algorithm, best_params, n_pos=n_pos, n_neg=n_neg)
    sw = utils.get_sample_weight(y_train, algorithm)
    if NEED_IMPUTE:
        from sklearn.impute import SimpleImputer
        imp_final = SimpleImputer(strategy="median")
        X_train_fit = imp_final.fit_transform(X_train)
        X_test_eval = imp_final.transform(X_test)
    else:
        X_train_fit, X_test_eval = X_train, X_test
    if sw is not None:
        model.fit(X_train_fit, y_train, sample_weight=sw)
    else:
        model.fit(X_train_fit, y_train)

    # 测试集评估
    y_prob_test = model.predict_proba(X_test_eval)[:, 1]
    metrics = utils.compute_metrics(y_test, y_prob_test)
    metrics["cv_auprc"] = best_cv_auprc

    return metrics, best_params, model, y_prob_test, oof_train


def run_nb_ensemble(all_algo_probs_train, y_train, all_algo_probs_test, y_test, endpoints):
    """NB 集成：对每个终点，用 5 算法的预测概率训练 NB。"""
    print("\n[NB Ensemble] 训练 NB 集成分类器 ...")
    results = []

    for i, ep in enumerate(endpoints):
        # 收集该终点在各算法上的测试集预测概率
        probs_train = np.column_stack([all_algo_probs_train[alg][:, i] for alg in config.ALGORITHMS])
        probs_test = np.column_stack([all_algo_probs_test[alg][:, i] for alg in config.ALGORITHMS])

        # 过滤 NaN (某些算法可能对该终点失败)
        valid_mask = ~np.any(np.isnan(probs_train), axis=1) & ~np.isnan(y_train)
        if np.sum(valid_mask) < 10:
            continue

        y_prob_nb, _ = utils.train_nb_ensemble(
            probs_train[valid_mask], y_train[valid_mask], probs_test
        )
        metrics = utils.compute_metrics(y_test, y_prob_nb)
        metrics["endpoint"] = ep
        results.append(metrics)

    return results


def run_y_randomization(algorithm, best_params, X_train, y_train, X_test, y_test, endpoint):
    """Y-randomization：打乱训练标签 100 次，重新训练。"""
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    real_metrics = utils.compute_metrics(y_test, model_predict(algorithm, best_params, X_train, y_train, X_test, n_pos, n_neg))

    rand_aucs = []
    for rep in range(config.Y_RAND_REPEATS):
        y_train_perm = np.random.permutation(y_train)
        y_prob = model_predict(algorithm, best_params, X_train, y_train_perm, X_test, n_pos, n_neg)
        m = utils.compute_metrics(y_test, y_prob)
        rand_aucs.append(m["auc_roc"])

    return {
        "endpoint": endpoint,
        "algorithm": algorithm,
        "real_auc_roc": real_metrics["auc_roc"],
        "rand_mean_auc_roc": np.nanmean(rand_aucs),
        "rand_std_auc_roc": np.nanstd(rand_aucs),
        "p_value": np.mean([r >= real_metrics["auc_roc"] for r in rand_aucs]),
    }


def model_predict(algorithm, params, X_train, y_train, X_test, n_pos, n_neg):
    """快速训练+预测（用于 Y-randomization）。"""
    model = utils.build_model(algorithm, params, n_pos=n_pos, n_neg=n_neg)
    sw = utils.get_sample_weight(y_train, algorithm)
    if sw is not None:
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 2.2 — 传统 ML 训练 (5 算法 × 26 终点)")
    print("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────
    print("\n[1/6] 加载数据 ...")
    meta = utils.load_meta(config.META_FILE)
    feature_columns = meta["feature_columns"]
    warnings_info = meta.get("warnings", {})

    # 跳过的终点
    skip_endpoints = set(warnings_info.get("no_positive_train_endpoints", []))
    low_pos_endpoints = set(warnings_info.get("low_positive_endpoints", []))

    print(f"  特征维度: {len(feature_columns)}")
    print(f"  跳过终点 (无正样本): {skip_endpoints}")
    print(f"  低正样本终点: {low_pos_endpoints}")

    # 加载特征
    train_ids, train_smiles, X_train = utils.load_features(config.TRAIN_FEATURES, feature_columns)
    test_ids, test_smiles, X_test = utils.load_features(config.TEST_FEATURES, feature_columns)
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    # 加载 Tox21 标签
    tox21_eps = meta["tox21"]["endpoints"]
    tox21_train_ids, Y_tox21_train = utils.load_labels(config.TOX21_TRAIN_LABELS, tox21_eps, "smiles")
    tox21_test_ids, Y_tox21_test = utils.load_labels(config.TOX21_TEST_LABELS, tox21_eps, "smiles")

    # 加载 CTD 标签
    ctd_eps = meta["ctd"]["endpoints"]
    ctd_train_ids, Y_ctd_train = utils.load_labels(config.CTD_TRAIN_LABELS, ctd_eps, "smiles")
    ctd_test_ids, Y_ctd_test = utils.load_labels(config.CTD_TEST_LABELS, ctd_eps, "smiles")

    # 匹配特征和标签
    # Tox21: 用 SMILES 匹配（特征的 SMILES 列 ↔ 标签的 SMILES 列）
    tox21_tr_idx, tox21_tr_lbl_idx, _ = utils.match_features_labels(train_smiles, tox21_train_ids, Y_tox21_train)
    tox21_te_idx, tox21_te_lbl_idx, _ = utils.match_features_labels(test_smiles, tox21_test_ids, Y_tox21_test)
    # CTD: SMILES 格式可能不一致，用 RDKit canonical 化后匹配
    from rdkit import Chem
    feat_canon = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in set(train_smiles) | set(test_smiles) if Chem.MolFromSmiles(s)}
    ctd_canon  = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in set(ctd_train_ids) | set(ctd_test_ids) if Chem.MolFromSmiles(s)}
    feat_canon_to_idx_train = {v: k for k, v in feat_canon.items()}
    feat_canon_to_idx_test  = {v: k for k, v in feat_canon.items()}
    ctd_tr_idx, ctd_tr_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon.get(s, s) for s in train_smiles]),
        np.array([ctd_canon.get(s, s) for s in ctd_train_ids]),
        Y_ctd_train
    )
    ctd_te_idx, ctd_te_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon.get(s, s) for s in test_smiles]),
        np.array([ctd_canon.get(s, s) for s in ctd_test_ids]),
        Y_ctd_test
    )

    print(f"  Tox21 匹配: train {len(tox21_tr_idx)}, test {len(tox21_te_idx)}")
    print(f"  CTD 匹配: train {len(ctd_tr_idx)}, test {len(ctd_te_idx)}")

    # 合并所有终点
    all_endpoints = []
    for ep in tox21_eps:
        if ep not in skip_endpoints:
            all_endpoints.append(("tox21", ep))
    for ep in ctd_eps:
        all_endpoints.append(("ctd", ep))

    print(f"\n  总终点数: {len(all_endpoints)} (跳过 {len(skip_endpoints)} 个)")

    # ── 2. 训练 ─────────────────────────────────────
    print(f"\n[2/6] 训练 ({len(config.ALGORITHMS)} 算法 × {len(all_endpoints)} 终点) ...")
    all_results = []
    best_params_map = {}  # (alg, ep) -> best_params（用于 Y-randomization）
    all_algo_probs_test = {alg: np.full((len(test_ids), len(all_endpoints)), np.nan) for alg in config.ALGORITHMS}
    all_algo_probs_train = {alg: np.full((len(train_ids), len(all_endpoints)), np.nan) for alg in config.ALGORITHMS}

    for ep_idx, (dataset, ep) in enumerate(all_endpoints):
        print(f"\n  === [{ep_idx+1}/{len(all_endpoints)}] {dataset}/{ep} ===")

        # 获取对应的特征和标签
        if dataset == "tox21":
            X_tr = X_train[tox21_tr_idx]
            y_tr_all = Y_tox21_train[tox21_tr_lbl_idx]
            X_te = X_test[tox21_te_idx]
            y_te_all = Y_tox21_test[tox21_te_lbl_idx]
            ep_local_idx = tox21_eps.index(ep)
            y_tr = y_tr_all[:, ep_local_idx]
            y_te = y_te_all[:, ep_local_idx]
            tr_idx_map = tox21_tr_idx
            te_idx_map = tox21_te_idx
        else:
            X_tr = X_train[ctd_tr_idx]
            y_tr_all = Y_ctd_train[ctd_tr_lbl_idx]
            X_te = X_test[ctd_te_idx]
            y_te_all = Y_ctd_test[ctd_te_lbl_idx]
            ep_local_idx = ctd_eps.index(ep)
            y_tr = y_tr_all[:, ep_local_idx]
            y_te = y_te_all[:, ep_local_idx]
            tr_idx_map = ctd_tr_idx
            te_idx_map = ctd_te_idx

        # 检查正样本数
        n_pos_train = int(np.nansum(y_tr == 1))
        n_pos_test = int(np.nansum(y_te == 1))
        if n_pos_train < 5:
            print(f"    跳过: 训练集正样本 {n_pos_train} < 5")
            continue

        # 去除 NaN
        valid_tr = ~np.isnan(y_tr)
        valid_te = ~np.isnan(y_te)
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
        X_te, y_te = X_te[valid_te], y_te[valid_te]

        for alg in config.ALGORITHMS:
            print(f"    {alg} ...", end="", flush=True)
            try:
                metrics, best_params, model, y_prob_test, oof_train = train_single(
                    alg, ep, X_tr, y_tr, X_te, y_te
                )
                # 保存模型
                model_path = os.path.join(config.TRADITIONAL_ML_MODELS, f"{alg}_{dataset}_{ep}.pkl")
                utils.save_model(model, model_path)

                # 记录结果
                best_params_map[(alg, ep)] = best_params
                metrics["algorithm"] = alg
                metrics["dataset"] = dataset
                metrics["endpoint"] = ep
                metrics["n_train"] = len(y_tr)
                metrics["n_test"] = len(y_te)
                metrics["n_pos_train"] = int(np.sum(y_tr == 1))
                metrics["n_pos_test"] = int(np.sum(y_te == 1))
                all_results.append(metrics)

                # 保存预测概率（用于 NB 集成）
                all_algo_probs_test[alg][te_idx_map, ep_idx] = y_prob_test
                all_algo_probs_train[alg][tr_idx_map[valid_tr], ep_idx] = oof_train

                print(f" AUC={metrics['auc_roc']:.3f} AUPRC={metrics['auprc']:.3f} F1={metrics['f1']:.3f}")
            except Exception as e:
                print(f" ERROR: {e}")
                all_results.append({
                    "algorithm": alg, "dataset": dataset, "endpoint": ep,
                    "auc_roc": np.nan, "auprc": np.nan, "f1": np.nan,
                    "balanced_accuracy": np.nan, "cv_auprc": np.nan,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "n_pos_train": n_pos_train, "n_pos_test": n_pos_test,
                    "error": str(e),
                })
        # 中间保存：每完成一个 endpoint 就保存一次（防止中断丢失）
        utils.save_results(all_results, os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results.csv"))

    # ── 3. 保存传统 ML 结果 ─────────────────────────
    print(f"\n[3/6] 保存传统 ML 结果 ...")
    results_df = utils.save_results(all_results, os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results.csv"))
    print(f"  保存 {len(all_results)} 条记录")

    # ── 4. NB 集成 ──────────────────────────────────
    print(f"\n[4/6] NB 集成 ...")
    nb_results = []
    # 预存各终点的训练标签（用于 NB 训练）
    ep_train_labels = {}
    for dataset_i, ep_i in all_endpoints:
        if dataset_i == "tox21":
            _lbl = Y_tox21_train[tox21_tr_lbl_idx][:, tox21_eps.index(ep_i)]
            _idx = tox21_tr_idx
        else:
            _lbl = Y_ctd_train[ctd_tr_lbl_idx][:, ctd_eps.index(ep_i)]
            _idx = ctd_tr_idx
        full_lbl = np.full(len(train_ids), np.nan)
        full_lbl[_idx] = _lbl
        ep_train_labels[ep_i] = full_lbl

    for i, (dataset, ep) in enumerate(all_endpoints):
        probs_train_list = []
        probs_test_list = []
        for alg in config.ALGORITHMS:
            probs_train_list.append(all_algo_probs_train[alg][:, i])
            probs_test_list.append(all_algo_probs_test[alg][:, i])
        probs_train = np.column_stack(probs_train_list)
        probs_test = np.column_stack(probs_test_list)
        # SVM 的 OOF 预测可能为 NaN，用中位数填充
        from sklearn.impute import SimpleImputer
        _imp_nb = SimpleImputer(strategy="median")
        probs_train = _imp_nb.fit_transform(probs_train)
        probs_test = _imp_nb.transform(probs_test)
        y_lbl = ep_train_labels[ep]
        valid_mask = ~np.any(np.isnan(probs_train), axis=1) & ~np.isnan(y_lbl)
        if np.sum(valid_mask) < 20:
            continue
        y_prob_nb, _ = utils.train_nb_ensemble(probs_train[valid_mask], y_lbl[valid_mask], probs_test)
        # 测试集标签（split 后无 NaN）
        if dataset == "tox21":
            y_test_ep = Y_tox21_test[tox21_te_lbl_idx][:, tox21_eps.index(ep)]
        else:
            y_test_ep = Y_ctd_test[ctd_te_lbl_idx][:, ctd_eps.index(ep)]
        metrics = utils.compute_metrics(y_test_ep, y_prob_nb)
        metrics["endpoint"] = ep
        metrics["dataset"] = dataset
        nb_results.append(metrics)
        print(f"  {ep}: AUC={metrics['auc_roc']:.3f} AUPRC={metrics['auprc']:.3f}")
    utils.save_results(nb_results, os.path.join(config.NB_ENSEMBLE_RESULTS, "nb_ensemble_results.csv"))
    print(f"  保存 {len(nb_results)} 条记录")

    # ── 5. Y-randomization ──────────────────────────
    print(f"\n[5/6] Y-randomization (选取最优算法对前 3 个终点) ...")
    y_rand_results = []
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        top_eps = results_df.dropna(subset=["auprc"]).groupby("endpoint")["auprc"].max().nlargest(3)
        for ep_name, best_auprc in top_eps.items():
            ep_row = results_df[results_df["endpoint"] == ep_name].dropna(subset=["auprc"]).iloc[0]
            best_alg = ep_row["algorithm"]
            best_params = best_params_map.get((best_alg, ep_name), {})
            dataset = ep_row["dataset"]
            print(f"  {ep_name} (best={best_alg}, AUPRC={best_auprc:.3f}) ...", end="", flush=True)
            # 加载数据
            if dataset == "tox21":
                _tr_idx = tox21_tr_idx
                _te_idx = tox21_te_idx
                _tr_lbl_idx = tox21_tr_lbl_idx
                _te_lbl_idx = tox21_te_lbl_idx
                _Y_tr = Y_tox21_train
                _Y_te = Y_tox21_test
                _eps = tox21_eps
            else:
                _tr_idx = ctd_tr_idx
                _te_idx = ctd_te_idx
                _tr_lbl_idx = ctd_tr_lbl_idx
                _te_lbl_idx = ctd_te_lbl_idx
                _Y_tr = Y_ctd_train
                _Y_te = Y_ctd_test
                _eps = ctd_eps
            _ep_local = _eps.index(ep_name)
            _X_tr = X_train[_tr_idx]
            _y_tr = _Y_tr[_tr_lbl_idx][:, _ep_local]
            _X_te = X_test[_te_idx]
            _y_te = _Y_te[_te_lbl_idx][:, _ep_local]
            _valid_tr = ~np.isnan(_y_tr)
            _valid_te = ~np.isnan(_y_te)
            _X_tr, _y_tr = _X_tr[_valid_tr], _y_tr[_valid_tr]
            _X_te, _y_te = _X_te[_valid_te], _y_te[_valid_te]
            if np.sum(_y_tr == 1) < 5:
                print(" 跳过(正样本不足)")
                continue
            # 重新训练最优模型获取真实 AUPRC
            real_prob = model_predict(best_alg, {}, _X_tr, _y_tr, _X_te, int(np.sum(_y_tr == 1)), int(np.sum(_y_tr == 0)))
            real_auprc = average_precision_score(_y_te, real_prob)
            rand_auprcs = []
            for _ in range(config.Y_RAND_REPEATS):
                _y_tr_perm = np.random.permutation(_y_tr)
                _prob = model_predict(best_alg, {}, _X_tr, _y_tr_perm, _X_te, int(np.sum(_y_tr == 1)), int(np.sum(_y_tr == 0)))
                rand_auprcs.append(average_precision_score(_y_te, _prob))
            p_value = np.mean([r >= real_auprc for r in rand_auprcs])
            y_rand_results.append({
                "endpoint": ep_name,
                "algorithm": best_alg,
                "real_auprc": real_auprc,
                "rand_mean_auprc": np.mean(rand_auprcs),
                "rand_std_auprc": np.std(rand_auprcs),
                "p_value": p_value,
            })
            print(f" real_AUPRC={real_auprc:.3f} rand_mean={np.mean(rand_auprcs):.3f}±{np.std(rand_auprcs):.3f} p={p_value:.3f}")

    utils.save_results(y_rand_results, os.path.join(config.Y_RAND_RESULTS, "y_randomization_results.csv"))

    # ── 6. 汇总 ─────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n[6/6] 完成!")
    print(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  模型目录: {config.TRADITIONAL_ML_MODELS}")
    print(f"  结果目录: {config.TRADITIONAL_ML_RESULTS}")

    # 打印汇总表
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*60}")
        print("汇总 (按算法):")
        summary = df.groupby("algorithm").agg({
            "auc_roc": "mean",
            "auprc": "mean",
            "f1": "mean",
        }).round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
