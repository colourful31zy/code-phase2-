"""
Phase 2.2b — NB 集成 + Y-randomization（从已有模型运行）
=====================================================
使用 traditional_ml_results.csv 中已保存的模型，不重跑 Optuna。
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils


def model_predict(algorithm, params, X_train, y_train, X_test, n_pos, n_neg):
    model = utils.build_model(algorithm, params, n_pos=n_pos, n_neg=n_neg)
    sw = utils.get_sample_weight(y_train, algorithm)
    if sw is not None:
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 2.2b — NB 集成 + Y-randomization")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据 ...")
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

    # CTD: 用 canonical SMILES 匹配
    from rdkit import Chem
    def _canon(smi):
        try:
            m = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(m, canonical=True) if m else smi
        except:
            return smi
    feat_canon = {s: _canon(s) for s in train_smiles}
    feat_canon_test = {s: _canon(s) for s in test_smiles}
    ctd_canon = {s: _canon(s) for s in ctd_train_ids}
    ctd_canon_test = {s: _canon(s) for s in ctd_test_ids}
    ctd_tr_idx, ctd_tr_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon.get(s, s) for s in train_smiles]),
        np.array([ctd_canon.get(s, s) for s in ctd_train_ids]),
        Y_ctd_train
    )
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

    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  终点数: {len(all_endpoints)}")

    # 加载已有结果
    results_path = os.path.join(config.TRADITIONAL_ML_RESULTS, "traditional_ml_results.csv")
    if not os.path.exists(results_path):
        print("ERROR: 未找到 traditional_ml_results.csv，请先运行主训练脚本")
        return
    all_results_df = pd.read_csv(results_path)
    print(f"  已有结果: {len(all_results_df)} 条")

    # ── 2. 用已有模型生成预测 ──────────────────────────
    print(f"\n[2/4] 用已有模型生成预测 ...")
    all_algo_probs_test = {alg: np.full((len(test_ids), len(all_endpoints)), np.nan) for alg in config.ALGORITHMS}
    all_algo_probs_train = {alg: np.full((len(train_ids), len(all_endpoints)), np.nan) for alg in config.ALGORITHMS}

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
                print(f"    {alg}/{ep}: 模型文件不存在，跳过")
                continue
            print(f"    {alg}/{ep} ...", end="", flush=True)
            model = utils.load_model(model_path)
            # 需要对 SVM/MLP 做同样的 imputation
            if alg in ("SVM", "MLP"):
                imp = SimpleImputer(strategy="median")
                X_tr_imp = imp.fit_transform(X_tr)
                X_te_imp = imp.transform(X_te)
                y_prob_test = model.predict_proba(X_te_imp)[:, 1]
                y_prob_train = model.predict_proba(X_tr_imp)[:, 1]
            else:
                y_prob_test = model.predict_proba(X_te)[:, 1]
                y_prob_train = model.predict_proba(X_tr)[:, 1]
            all_algo_probs_test[alg][te_idx[valid_te], ep_idx] = y_prob_test
            all_algo_probs_train[alg][tr_idx[valid_tr], ep_idx] = y_prob_train
            print(" done")

    # ── 3. NB 集成 ──────────────────────────────────
    print(f"\n[3/4] NB 集成 ...")
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

    nb_results = []
    for i, (dataset, ep) in enumerate(all_endpoints):
        # 确定测试集索引
        if dataset == "tox21":
            te_idx = tox21_te_idx
            y_test_ep = Y_tox21_test[tox21_te_lbl_idx][:, tox21_eps.index(ep)]
        else:
            te_idx = ctd_te_idx
            y_test_ep = Y_ctd_test[ctd_te_lbl_idx][:, ctd_eps.index(ep)]

        probs_train = np.column_stack([all_algo_probs_train[alg][:, i] for alg in config.ALGORITHMS])
        probs_test_full = np.column_stack([all_algo_probs_test[alg][:, i] for alg in config.ALGORITHMS])
        # 只取该终点对应的测试集行
        probs_test = probs_test_full[te_idx]

        y_lbl = ep_train_labels[ep]
        valid_mask = ~np.any(np.isnan(probs_train), axis=1) & ~np.isnan(y_lbl)
        if np.sum(valid_mask) < 20:
            continue
        imp_nb = SimpleImputer(strategy="median")
        probs_train_imp = imp_nb.fit_transform(probs_train[valid_mask])
        probs_test_imp = imp_nb.transform(probs_test)
        y_prob_nb, _ = utils.train_nb_ensemble(probs_train_imp, y_lbl[valid_mask], probs_test_imp)
        metrics = utils.compute_metrics(y_test_ep, y_prob_nb)
        metrics["endpoint"] = ep
        metrics["dataset"] = dataset
        nb_results.append(metrics)
        print(f"  {ep}: AUC={metrics['auc_roc']:.3f} AUPRC={metrics['auprc']:.3f}")
    utils.save_results(nb_results, os.path.join(config.NB_ENSEMBLE_RESULTS, "nb_ensemble_results.csv"))
    print(f"  保存 {len(nb_results)} 条")

    # ── 4. Y-randomization ──────────────────────────
    print(f"\n[4/4] Y-randomization (前 3 个终点) ...")
    y_rand_results = []
    top_eps = all_results_df.dropna(subset=["auprc"]).groupby("endpoint")["auprc"].max().nlargest(3)
    for ep_name, best_auprc in top_eps.items():
        ep_row = all_results_df[all_results_df["endpoint"] == ep_name].dropna(subset=["auprc"]).iloc[0]
        best_alg = ep_row["algorithm"]
        dataset = ep_row["dataset"]
        print(f"  {ep_name} (best={best_alg}, AUPRC={best_auprc:.3f}) ...", end="", flush=True)
        if dataset == "tox21":
            _tr_idx, _te_idx = tox21_tr_idx, tox21_te_idx
            _tr_lbl, _te_lbl = tox21_tr_lbl_idx, tox21_te_lbl_idx
            _Y_tr, _Y_te = Y_tox21_train, Y_tox21_test
            _eps = tox21_eps
        else:
            _tr_idx, _te_idx = ctd_tr_idx, ctd_te_idx
            _tr_lbl, _te_lbl = ctd_tr_lbl_idx, ctd_te_lbl_idx
            _Y_tr, _Y_te = Y_ctd_train, Y_ctd_test
            _eps = ctd_eps
        _ep_local = _eps.index(ep_name)
        _X_tr = X_train[_tr_idx]
        _y_tr = _Y_tr[_tr_lbl][:, _ep_local]
        _X_te = X_test[_te_idx]
        _y_te = _Y_te[_te_lbl][:, _ep_local]
        _vt = ~np.isnan(_y_tr); _vte = ~np.isnan(_y_te)
        _X_tr, _y_tr = _X_tr[_vt], _y_tr[_vt]
        _X_te, _y_te = _X_te[_vte], _y_te[_vte]
        if np.sum(_y_tr == 1) < 5:
            print(" 跳过"); continue
        if best_alg in ("SVM", "MLP"):
            imp_yr = SimpleImputer(strategy="median")
            _X_tr = imp_yr.fit_transform(_X_tr)
            _X_te = imp_yr.transform(_X_te)
        n_pos = int(np.sum(_y_tr == 1)); n_neg = int(np.sum(_y_tr == 0))
        real_prob = model_predict(best_alg, {}, _X_tr, _y_tr, _X_te, n_pos, n_neg)
        real_auprc = average_precision_score(_y_te, real_prob)
        rand_auprcs = []
        for _ in range(config.Y_RAND_REPEATS):
            _yp = np.random.permutation(_y_tr)
            _p = model_predict(best_alg, {}, _X_tr, _yp, _X_te, n_pos, n_neg)
            rand_auprcs.append(average_precision_score(_y_te, _p))
        p_value = np.mean([r >= real_auprc for r in rand_auprcs])
        y_rand_results.append({
            "endpoint": ep_name, "algorithm": best_alg,
            "real_auprc": real_auprc, "rand_mean_auprc": np.mean(rand_auprcs),
            "rand_std_auprc": np.std(rand_auprcs), "p_value": p_value,
        })
        print(f" real={real_auprc:.3f} rand={np.mean(rand_auprcs):.3f}±{np.std(rand_auprcs):.3f} p={p_value:.3f}")

    utils.save_results(y_rand_results, os.path.join(config.Y_RAND_RESULTS, "y_randomization_results.csv"))

    elapsed = time.time() - t0
    print(f"\n完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
