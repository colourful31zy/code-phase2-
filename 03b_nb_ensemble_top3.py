"""
03b_nb_ensemble_top3.py — NB Ensemble with Top 3 algorithms only (RF/XGBoost/LightGBM)
Drop SVM and MLP which degraded the original ensemble.

Usage:
    python code/phase2/03b_nb_ensemble_top3.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils

TOP3_ALGOS = ["RF", "XGBoost", "LightGBM"]

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    valid = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true, y_prob = y_true[valid], y_prob[valid]
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": np.nan, "auprc": np.nan, "f1": np.nan, "balanced_accuracy": np.nan}
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

def main():
    t0 = time.time()
    print("=" * 60)
    print("NB Ensemble v2 — Top 3 algorithms only (RF/XGBoost/LightGBM)")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    meta = utils.load_meta(config.META_FILE)
    feature_columns = meta["feature_columns"]
    skip_eps = set(meta.get("warnings", {}).get("no_positive_train_endpoints", []))

    train_ids, train_smiles, X_train = utils.load_features(config.TRAIN_FEATURES, feature_columns)
    test_ids, test_smiles, X_test = utils.load_features(config.TEST_FEATURES, feature_columns)

    tox21_eps = [e for e in meta["tox21"]["endpoints"] if e not in skip_eps]
    ctd_eps = meta["ctd"]["endpoints"]
    all_eps = list(tox21_eps) + list(ctd_eps)

    # Tox21 labels
    t21_tr_ids, Y_t21_tr = utils.load_labels(config.TOX21_TRAIN_LABELS, tox21_eps, "smiles")
    t21_te_ids, Y_t21_te = utils.load_labels(config.TOX21_TEST_LABELS, tox21_eps, "smiles")
    tr_i_tox, tr_l_tox, _ = utils.match_features_labels(train_smiles, t21_tr_ids, Y_t21_tr)
    te_i_tox, te_l_tox, _ = utils.match_features_labels(test_smiles, t21_te_ids, Y_t21_te)

    # CTD labels (canonical SMILES)
    def _canon(smi):
        from rdkit import Chem
        try:
            m = Chem.MolFromSmiles(str(smi))
            return Chem.MolToSmiles(m, canonical=True) if m else smi
        except:
            return smi
    ctd_tr_ids, Y_ctd_tr = utils.load_labels(config.CTD_TRAIN_LABELS, ctd_eps, "smiles")
    ctd_te_ids, Y_ctd_te = utils.load_labels(config.CTD_TEST_LABELS, ctd_eps, "smiles")
    fc = {s: _canon(s) for s in train_smiles}
    lc = {s: _canon(s) for s in ctd_tr_ids}
    tr_i_ctd, tr_l_ctd, _ = utils.match_features_labels(
        np.array([fc.get(s, s) for s in train_smiles]),
        np.array([lc.get(s, s) for s in ctd_tr_ids]), Y_ctd_tr)
    fc_te = {s: _canon(s) for s in test_smiles}
    lc_te = {s: _canon(s) for s in ctd_te_ids}
    te_i_ctd, te_l_ctd, _ = utils.match_features_labels(
        np.array([fc_te.get(s, s) for s in test_smiles]),
        np.array([lc_te.get(s, s) for s in ctd_te_ids]), Y_ctd_te)

    # ── Prepare per-endpoint data ────────────────────────────────
    # Build endpoint -> (X_tr_ep, y_tr_ep, X_te_ep, y_te_ep) mapping
    endpoint_data = {}
    for ep_i, ep in enumerate(tox21_eps):
        y_tr = Y_t21_tr[tr_l_tox][:, ep_i]
        y_te = Y_t21_te[te_l_tox][:, ep_i]
        vt = ~np.isnan(y_tr); vte = ~np.isnan(y_te)
        if np.sum(y_tr[vt] == 1) < 5:
            continue
        ep_obj = type('obj', (object,), {
            'name': ep, 'dataset': 'tox21',
            'X_tr': X_train[tr_i_tox][vt], 'y_tr': y_tr[vt],
            'X_te': X_test[te_i_tox][vte], 'y_te': y_te[vte],
        })
        endpoint_data[ep] = ep_obj

    for ep_i, ep in enumerate(ctd_eps):
        y_tr = Y_ctd_tr[tr_l_ctd][:, ep_i]
        y_te = Y_ctd_te[te_l_ctd][:, ep_i]
        vt = ~np.isnan(y_tr); vte = ~np.isnan(y_te)
        if np.sum(y_tr[vt] == 1) < 5:
            continue
        ep_obj = type('obj', (object,), {
            'name': ep, 'dataset': 'ctd',
            'X_tr': X_train[tr_i_ctd][vt], 'y_tr': y_tr[vt],
            'X_te': X_test[te_i_ctd][vte], 'y_te': y_te[vte],
        })
        endpoint_data[ep] = ep_obj

    print(f"  Endpoints: {len(endpoint_data)} (with >=5 pos samples)")

    # ── Train & OOF per endpoint ─────────────────────────────────
    print("\n[2/4] Training 3 algorithms + OOF predictions...")
    imp = SimpleImputer(strategy="median")
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    oof_dict = {}     # ep -> {alg: oof_preds}
    test_dict = {}    # ep -> {alg: test_prob}
    best_single = {}  # ep -> {'algo': , 'auc': , 'auprc':}

    for ep_name, ep_data in sorted(endpoint_data.items()):
        X_tr = imp.fit_transform(ep_data.X_tr)
        X_te = imp.transform(ep_data.X_te)
        y_tr, y_te = ep_data.y_tr, ep_data.y_te
        n_pos = int(np.sum(y_tr == 1))
        n_neg = len(y_tr) - n_pos

        print(f"\n  [{ep_name}] train={len(y_tr)} ({n_pos} pos, {n_pos/len(y_tr)*100:.1f}%)")

        oof_dict[ep_name] = {}
        test_dict[ep_name] = {}
        best_single[ep_name] = {'algo': '', 'auc': 0, 'auprc': 0}

        for alg in TOP3_ALGOS:
            # Quick Optuna: only 10 trials (fast, ~same as 20 for NB ensemble purposes)
            def _objective(trial):
                p = {}
                for name, fn in config.OPTUNA_SPACES[alg].items():
                    p[name] = fn(trial)
                cv_auprcs = []
                for tr_i, va_i in skf.split(X_tr, y_tr):
                    m = utils.build_model(alg, p, n_pos=n_pos, n_neg=n_neg)
                    m.fit(X_tr[tr_i], y_tr[tr_i])
                    yp = m.predict_proba(X_tr[va_i])[:, 1]
                    cv_auprcs.append(average_precision_score(y_tr[va_i], yp))
                return np.mean(cv_auprcs)

            import optuna
            from optuna.samplers import TPESampler
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.SEED))
            study.optimize(_objective, n_trials=10, show_progress_bar=False)
            best_p = study.best_params

            # OOF predictions
            oof = np.full(len(y_tr), np.nan)
            for tr_i, va_i in skf.split(X_tr, y_tr):
                m = utils.build_model(alg, best_p, n_pos=n_pos, n_neg=n_neg)
                n_p = int(np.sum(y_tr[tr_i] == 1))
                n_n = len(y_tr[tr_i]) - n_p
                m = utils.build_model(alg, best_p, n_pos=n_p, n_neg=n_n)
                m.fit(X_tr[tr_i], y_tr[tr_i])
                oof[va_i] = m.predict_proba(X_tr[va_i])[:, 1]
            oof_dict[ep_name][alg] = oof

            # Test predictions
            m = utils.build_model(alg, best_p, n_pos=n_pos, n_neg=n_neg)
            m.fit(X_tr, y_tr)
            tp = m.predict_proba(X_te)[:, 1]
            test_dict[ep_name][alg] = tp

            # Single model metrics
            mets = compute_metrics(y_te, tp)
            auc = mets.get('auc_roc', 0) or 0
            auprc = mets.get('auprc', 0) or 0
            print(f"    {alg:<10} AUC={auc:.4f} AUPRC={auprc:.4f}")
            if auc > best_single[ep_name]['auc']:
                best_single[ep_name] = {'algo': alg, 'auc': auc, 'auprc': auprc}

    # ── NB Ensemble ──────────────────────────────────────────────
    print("\n[3/4] NB Ensemble (GaussianNB on Top 3 OOF predictions)...")
    nb_results = []
    best_results = []

    for ep_name in sorted(endpoint_data.keys()):
        y_te = endpoint_data[ep_name].y_te

        # Stack OOF predictions: n_samples × 3
        oof_stack = np.column_stack([oof_dict[ep_name][a] for a in TOP3_ALGOS])
        test_stack = np.column_stack([test_dict[ep_name][a] for a in TOP3_ALGOS])

        # Handle NaN in OOF
        if np.any(np.isnan(oof_stack)):
            oof_stack = np.nan_to_num(oof_stack, nan=0.5)
        if np.any(np.isnan(test_stack)):
            test_stack = np.nan_to_num(test_stack, nan=0.5)

        # Fit GaussianNB
        nb = GaussianNB()
        nb.fit(oof_stack, endpoint_data[ep_name].y_tr)
        nb_prob = nb.predict_proba(test_stack)[:, 1]

        mets = compute_metrics(y_te, nb_prob)
        nb_results.append({
            'endpoint': ep_name,
            'dataset': endpoint_data[ep_name].dataset,
            'auc_roc': mets['auc_roc'], 'auprc': mets['auprc'],
            'f1': mets['f1'], 'balanced_accuracy': mets['balanced_accuracy'],
        })
        print(f"  {ep_name:<20} NB_Ensemble AUC={mets['auc_roc']:.4f} AUPRC={mets['auprc']:.4f}")

        # Per-endpoint best single model
        bs = best_single[ep_name]
        best_results.append({
            'endpoint': ep_name,
            'dataset': endpoint_data[ep_name].dataset,
            'algorithm': bs['algo'],
            'auc_roc': bs['auc'],
            'auprc': bs['auprc'],
        })

    # ── Save & Report ────────────────────────────────────────────
    print("\n[4/4] Saving results...")

    # NB Ensemble results
    nb_df = pd.DataFrame(nb_results)
    nb_path = os.path.join(config.TRADITIONAL_ML_RESULTS, "nb_ensemble_top3_results.csv")
    utils.save_results(nb_results, nb_path)
    print(f"  NB Ensemble saved: {nb_path}")

    # Best single model per endpoint
    best_df = pd.DataFrame(best_results)
    best_path = os.path.join(config.TRADITIONAL_ML_RESULTS, "best_per_endpoint.csv")
    best_df.to_csv(best_path, index=False)
    print(f"  Best per-endpoint saved: {best_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)
    nb_auc = nb_df['auc_roc'].dropna().mean()
    nb_auprc = nb_df['auprc'].dropna().mean()
    bs_auc = best_df['auc_roc'].dropna().mean()
    bs_auprc = best_df['auprc'].dropna().mean()
    print(f"  NB Ensemble (Top3):   AUC={nb_auc:.4f} AUPRC={nb_auprc:.4f}")
    print(f"  Best single per-ep:   AUC={bs_auc:.4f} AUPRC={bs_auprc:.4f}")
    print(f"\n  Time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()