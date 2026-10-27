"""
02c_train_improved.py — Grouped training with Feature Selection + class_weight
Plan D v2: Mutual Information feature selection, no SMOTE, 50 Optuna trials.
Also: per-endpoint best-algorithm-only mode.

Usage:
    python code/phase2/02c_train_improved.py --group tox21
    python code/phase2/02c_train_improved.py --group ctd_tox
    python code/phase2/02c_train_improved.py --group ctd_cyp
"""
import os, sys, time, argparse, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import optuna
from optuna.samplers import TPESampler

OUTPUT_TAG = "improved_v3"

# ── Group definitions ─────────────────────────────────────────────────
GROUPS = {
    "tox21": {
        "endpoints": None,
        "dataset": "tox21",
        "algorithms": ["XGBoost", "LightGBM", "RF"],
        "optuna_trials": 50,
        "desc": "Tox21 HTS (10 eps, 2-14% pos)",
    },
    "ctd_tox": {
        "endpoints": ["tox_hepatotoxicity", "tox_cardiotoxicity",
                       "tox_nephrotoxicity", "tox_neurotoxicity"],
        "dataset": "ctd",
        "algorithms": ["LightGBM", "RF", "XGBoost"],
        "optuna_trials": 50,
        "desc": "CTD organ toxicity (4 eps, 9-61% pos)",
    },
    "ctd_cyp": {
        "endpoints": ["gene_CYP1A2", "gene_CYP2C9", "gene_CYP2C19",
                       "gene_CYP2D6", "gene_CYP3A4", "gene_CYP3A5",
                       "gene_KCNH2"],
        "dataset": "ctd",
        "algorithms": ["RF", "XGBoost", "LightGBM"],
        "optuna_trials": 50,
        "desc": "CTD CYP450 (7 eps, 4-17% pos)",
    },
}


def _canon(smi):
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except:
        return smi


def mutual_info_select(X_tr, y_tr, X_te, k=500):
    """Select top K features by mutual information. Returns X_tr_sel, X_te_sel."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
    selector.fit(X_tr, y_tr)
    return selector.transform(X_tr), selector.transform(X_te)


def train_endpoint(alg, ep_name, X_tr, y_tr, X_te, y_te, optuna_trials=50,
                   n_feat=500):
    """Train one endpoint: feature selection + class_weight + Optuna."""
    valid_tr = ~np.isnan(y_tr)
    valid_te = ~np.isnan(y_te)
    X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
    X_te, y_te = X_te[valid_te], y_te[valid_te]

    n_pos = int(np.sum(y_tr == 1))
    if n_pos < 5:
        return None, None, None, None, None

    # IMPUTE (handle NaN from RDKit)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    # ── Feature Selection ─────────────────────────────────
    X_tr_fs, X_te_fs = mutual_info_select(X_tr, y_tr, X_te, k=n_feat)
    n_neg = len(y_tr) - n_pos

    # ── Optuna ─────────────────────────────────────────────
    def _objective(trial):
        p = {}
        for name, fn in config.OPTUNA_SPACES[alg].items():
            p[name] = fn(trial)
        m = utils.build_model(alg, p, n_pos=n_pos, n_neg=n_neg)
        skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)
        auprcs = []
        for tr_i_cv, va_i in skf.split(X_tr_fs, y_tr):
            cv_m = utils.build_model(alg, p, n_pos=n_pos, n_neg=n_neg)
            cv_m.fit(X_tr_fs[tr_i_cv], y_tr[tr_i_cv])
            yp = cv_m.predict_proba(X_tr_fs[va_i])[:, 1]
            auprcs.append(average_precision_score(y_tr[va_i], yp))
        return np.mean(auprcs)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.SEED))
    study.optimize(_objective, n_trials=optuna_trials, show_progress_bar=False)
    best_params = study.best_params
    cv_auprc = study.best_value

    # ── Final model (on FS data) ──────────────────────────
    model = utils.build_model(alg, best_params, n_pos=n_pos, n_neg=n_neg)
    model.fit(X_tr_fs, y_tr)

    # ── OOF on FS data (for NB ensemble) ──────────────────
    oof = np.full(len(y_tr), np.nan)
    skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)
    for tr_i_cv, va_i in skf.split(X_tr_fs, y_tr):
        cv_m = utils.build_model(alg, best_params, n_pos=n_pos, n_neg=n_neg)
        cv_m.fit(X_tr_fs[tr_i_cv], y_tr[tr_i_cv])
        oof[va_i] = cv_m.predict_proba(X_tr_fs[va_i])[:, 1]

    # ── Test predict ─────────────────────────────────────
    y_prob_test = model.predict_proba(X_te_fs)[:, 1]

    metrics = utils.compute_metrics(y_te, y_prob_test)
    metrics["cv_auprc"] = cv_auprc
    return metrics, best_params, model, y_prob_test, oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True,
                        choices=["tox21", "ctd_tox", "ctd_cyp"])
    parser.add_argument("--n-feat", type=int, default=500,
                        help="Number of features to select (0 = no selection)")
    args = parser.parse_args()
    gc = GROUPS[args.group]

    t0 = time.time()
    print("=" * 70)
    print(f"Phase 2C Improved Training v3 [Group: {args.group}]")
    print(gc["desc"])
    print(f"Algorithms: {gc['algorithms']}")
    print(f"SMOTE: OFF (using class_weight only)")
    print(f"Feature Selection: MI Top-{args.n_feat}" if args.n_feat > 0 else "Feature Selection: OFF")
    print(f"Optuna trials: {gc['optuna_trials']}")
    print("=" * 70)

    # ── 1. Load data ───────────────────────────────────────
    print("\n[1/3] Loading data...")
    meta = utils.load_meta(config.META_FILE)
    feature_columns = meta["feature_columns"]
    n_feat_actual = min(args.n_feat, len(feature_columns)) if args.n_feat > 0 else len(feature_columns)

    tr_ids, tr_smi, X_tr = utils.load_features(config.TRAIN_FEATURES, feature_columns)
    te_ids, te_smi, X_te = utils.load_features(config.TEST_FEATURES, feature_columns)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    # ── 2. Load labels ────────────────────────────────────
    if args.group == "tox21":
        tox21_eps = meta["tox21"]["endpoints"]
        skip = set(meta.get("warnings", {}).get("no_positive_train_endpoints", []))
        gc["endpoints"] = [e for e in tox21_eps if e not in skip]
        lbl_tr_ids, Y_tr = utils.load_labels(config.TOX21_TRAIN_LABELS, gc["endpoints"], "smiles")
        lbl_te_ids, Y_te = utils.load_labels(config.TOX21_TEST_LABELS, gc["endpoints"], "smiles")
        tr_i, tr_l, _ = utils.match_features_labels(tr_smi, lbl_tr_ids, Y_tr)
        te_i, te_l, _ = utils.match_features_labels(te_smi, lbl_te_ids, Y_te)
        print(f"  Endpoints: {gc['endpoints']} (n={len(gc['endpoints'])})")
    else:
        valid_eps = [e for e in gc["endpoints"] if e in meta["ctd"]["endpoints"]]
        gc["endpoints"] = valid_eps
        lbl_tr_ids, Y_tr = utils.load_labels(config.CTD_TRAIN_LABELS, valid_eps, "smiles")
        lbl_te_ids, Y_te = utils.load_labels(config.CTD_TEST_LABELS, valid_eps, "smiles")
        fc = {s: _canon(s) for s in tr_smi}
        lc = {s: _canon(s) for s in lbl_tr_ids}
        tr_i, tr_l, _ = utils.match_features_labels(
            np.array([fc.get(s, s) for s in tr_smi]),
            np.array([lc.get(s, s) for s in lbl_tr_ids]), Y_tr)
        fc_te = {s: _canon(s) for s in te_smi}
        lc_te = {s: _canon(s) for s in lbl_te_ids}
        te_i, te_l, _ = utils.match_features_labels(
            np.array([fc_te.get(s, s) for s in te_smi]),
            np.array([lc_te.get(s, s) for s in lbl_te_ids]), Y_te)
        print(f"  Endpoints: {valid_eps} (n={len(valid_eps)})")

    print(f"  Matched: train={len(tr_i)}, test={len(te_i)}")

    # ── 3. Train ───────────────────────────────────────────
    print(f"\n[2/3] Training ({len(gc['algorithms'])} alg x {len(gc['endpoints'])} ep)...")

    all_results = []
    best_params_map = {}
    n_ep = len(gc["endpoints"])
    all_ptest = {a: np.full((len(te_ids), n_ep), np.nan) for a in gc["algorithms"]}
    all_ptrain = {a: np.full((len(tr_ids), n_ep), np.nan) for a in gc["algorithms"]}

    for ep_i, ep in enumerate(gc["endpoints"]):
        print(f"\n  === [{ep_i+1}/{n_ep}] {ep} ===")
        y_tr = Y_tr[tr_l][:, ep_i]
        y_te = Y_te[te_l][:, ep_i]
        vt = ~np.isnan(y_tr)
        vte = ~np.isnan(y_te)
        X_tr_e = X_tr[tr_i][vt]
        y_tr_e = y_tr[vt]
        X_te_e = X_te[te_i][vte]
        y_te_e = y_te[vte]
        np_ = int(np.sum(y_tr_e == 1))
        print(f"    train={len(X_tr_e)} ({np_} pos, {np_/len(X_tr_e)*100:.1f}%), test={len(X_te_e)}")
        if np_ < 5:
            print("    SKIP")
            continue

        for alg in gc["algorithms"]:
            print(f"    {alg} ...", end=" ", flush=True)
            try:
                mets, bp, model, ypt, oof = train_endpoint(
                    alg, ep, X_tr_e, y_tr_e, X_te_e, y_te_e,
                    optuna_trials=gc["optuna_trials"],
                    n_feat=n_feat_actual)
                if mets is None:
                    print("SKIP")
                    continue
                mp = os.path.join(config.TRADITIONAL_ML_MODELS,
                                  f"{alg}_{args.group}_{ep}_v3.pkl")
                utils.save_model(model, mp)
                best_params_map[(alg, ep)] = bp
                mets["algorithm"] = alg
                mets["dataset"] = args.group
                mets["endpoint"] = ep
                mets["n_train"] = len(y_tr_e)
                mets["n_test"] = len(y_te_e)
                mets["n_pos_train"] = np_
                mets["n_pos_test"] = int(np.sum(y_te_e == 1))
                all_results.append(mets)
                all_ptest[alg][te_i[vte], ep_i] = ypt
                all_ptrain[alg][tr_i[vt], ep_i] = oof
                print(f"AUC={mets['auc_roc']:.3f} AUPRC={mets['auprc']:.3f} F1={mets['f1']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        utils.save_results(all_results,
            os.path.join(config.TRADITIONAL_ML_RESULTS,
                         f"{OUTPUT_TAG}_{args.group}_interim.csv"))

    # ── Save ────────────────────────────────────────────────
    print(f"\n[3/3] Saving results...")
    rp = os.path.join(config.TRADITIONAL_ML_RESULTS, f"{OUTPUT_TAG}_{args.group}.csv")
    utils.save_results(all_results, rp)
    print(f"  Saved {len(all_results)} rows")

    df = pd.DataFrame(all_results)
    if len(df) > 0:
        print("\nSummary:")
        for a in gc["algorithms"]:
            s = df[df["algorithm"] == a]
            if len(s) == 0:
                continue
            print(f"  {a}: AUC={s['auc_roc'].mean():.4f} AUPRC={s['auprc'].mean():.4f} F1={s['f1'].mean():.4f}")

    print(f"\nTime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()