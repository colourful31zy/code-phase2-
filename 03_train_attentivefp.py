"""
03_train_attentivefp.py — AttentiveFP multi-task GNN training
Phase 2: Train AttentiveFP on 21 filtered endpoints with Optuna tuning.
Uses PyTorch Geometric (PyG) AttentiveFP implementation.

Usage:
    python code/phase2/03_train_attentivefp.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, utils

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP as AttentiveFPModel
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")

# ── Filtered endpoints (same as traditional ML optimization) ─────────
FILTERED_ENDPOINTS = [
    "Tox21_0", "Tox21_1", "Tox21_2", "Tox21_3", "Tox21_4", "Tox21_5",
    "Tox21_6", "Tox21_7", "Tox21_8", "Tox21_9",
    "gene_CYP1A2", "gene_CYP2C19", "gene_CYP2C9", "gene_CYP2D6",
    "gene_CYP3A4", "gene_CYP3A5", "gene_KCNH2",
    "tox_cardiotoxicity", "tox_hepatotoxicity", "tox_nephrotoxicity", "tox_neurotoxicity",
]


# ── Featurization ────────────────────────────────────────────────────
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def _one_hot(val, choices, include_unknown=True):
    n = len(choices) + (1 if include_unknown else 0)
    vec = [0] * n
    if val in choices:
        vec[choices.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def featurize_atom(atom):
    """39-dim atom features (AttentiveFP standard)."""
    features = []
    features += _one_hot(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35, 53])  # 10
    features += _one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])  # 7
    features += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])  # 6
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)  # 6
    features.append(int(atom.GetIsAromatic()))  # 1
    features += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 6
    features.append(int(atom.IsInRing()))  # 1
    while len(features) < 39:
        features.append(0)
    return features[:39]


def featurize_bond(bond):
    """10-dim bond features."""
    features = []
    features += _one_hot(bond.GetBondType(),
                         [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])  # 5
    features.append(int(bond.GetIsConjugated()))  # 1
    features.append(int(bond.IsInRing()))  # 1
    features += _one_hot(bond.GetStereo(),
                         [Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOE,
                          Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREONONE])  # 4
    while len(features) < 10:
        features.append(0)
    return features[:10]


def smiles_to_pyg(smi):
    """Convert SMILES to PyG Data object. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        atom_features = [featurize_atom(a) for a in mol.GetAtoms()]
        if len(atom_features) == 0:
            return None
        src, dst, edge_features = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = featurize_bond(bond)
            src += [i, j]
            dst += [j, i]
            edge_features += [bf, bf]
        if len(src) == 0:
            src, dst = [0], [0]
            edge_features = [[0] * 10]
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except:
        return None


def compute_pos_weight(Y):
    """pos_weight = neg/pos per endpoint."""
    pos = np.nansum(Y, axis=0)
    neg = np.sum(~np.isnan(Y), axis=0) - pos
    pos = np.maximum(pos, 1)
    return torch.FloatTensor(neg / pos)


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


def prepare_data():
    """Load data, match labels, convert SMILES to PyG graphs."""
    print("=" * 60)
    print("Loading data...")

    # Load meta for endpoint names and feature columns
    meta = utils.load_meta(config.META_FILE)
    feature_columns = meta["feature_columns"]
    tox21_eps = meta["tox21"]["endpoints"]
    ctd_eps = meta["ctd"]["endpoints"]

    # Load features (we only need SMILES for GNN, not fingerprints)
    tr_ids, tr_smiles, _ = utils.load_features(config.TRAIN_FEATURES, feature_columns)
    te_ids, te_smiles, _ = utils.load_features(config.TEST_FEATURES, feature_columns)

    # Load labels
    tr_tox_ids, Y_tr_tox = utils.load_labels(config.TOX21_TRAIN_LABELS, tox21_eps, "smiles")
    te_tox_ids, Y_te_tox = utils.load_labels(config.TOX21_TEST_LABELS, tox21_eps, "smiles")
    tr_ctd_ids, Y_tr_ctd = utils.load_labels(config.CTD_TRAIN_LABELS, ctd_eps, "smiles")
    te_ctd_ids, Y_te_ctd = utils.load_labels(config.CTD_TEST_LABELS, ctd_eps, "smiles")

    # Match Tox21
    tr_tox_idx, tr_tox_lbl_idx, _ = utils.match_features_labels(tr_smiles, tr_tox_ids, Y_tr_tox)
    te_tox_idx, te_tox_lbl_idx, _ = utils.match_features_labels(te_smiles, te_tox_ids, Y_te_tox)

    # Match CTD (canonical SMILES)
    def _canon(smi):
        try:
            m = Chem.MolFromSmiles(str(smi))
            return Chem.MolToSmiles(m, canonical=True) if m else smi
        except:
            return smi

    feat_canon = {s: _canon(s) for s in tr_smiles}
    ctd_canon = {s: _canon(s) for s in tr_ctd_ids}
    tr_ctd_idx, tr_ctd_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon.get(s, s) for s in tr_smiles]),
        np.array([ctd_canon.get(s, s) for s in tr_ctd_ids]), Y_tr_ctd)
    feat_canon_te = {s: _canon(s) for s in te_smiles}
    ctd_canon_te = {s: _canon(s) for s in te_ctd_ids}
    te_ctd_idx, te_ctd_lbl_idx, _ = utils.match_features_labels(
        np.array([feat_canon_te.get(s, s) for s in te_smiles]),
        np.array([ctd_canon_te.get(s, s) for s in te_ctd_ids]), Y_te_ctd)

    # Build unified label matrix
    all_eps = list(tox21_eps) + list(ctd_eps)
    n_tr, n_te = len(tr_smiles), len(te_smiles)
    Y_tr = np.full((n_tr, len(all_eps)), np.nan)
    Y_te = np.full((n_te, len(all_eps)), np.nan)
    n_tox = len(tox21_eps)
    for i, li in enumerate(tr_tox_lbl_idx):
        Y_tr[tr_tox_idx[i], :n_tox] = Y_tr_tox[li]
    for i, li in enumerate(te_tox_lbl_idx):
        Y_te[te_tox_idx[i], :n_tox] = Y_te_tox[li]
    for i, li in enumerate(tr_ctd_lbl_idx):
        Y_tr[tr_ctd_idx[i], n_tox:] = Y_tr_ctd[li]
    for i, li in enumerate(te_ctd_lbl_idx):
        Y_te[te_ctd_idx[i], n_tox:] = Y_te_ctd[li]

    # Filter endpoints
    valid_idx, valid_eps, skipped = utils.get_valid_endpoints(Y_tr, Y_te, all_eps)
    ep_to_idx = {ep: i for i, ep in enumerate(valid_eps)}
    filt_idx = [ep_to_idx[ep] for ep in FILTERED_ENDPOINTS if ep in ep_to_idx]
    Y_tr = Y_tr[:, [valid_idx[i] for i in filt_idx]]
    Y_te = Y_te[:, [valid_idx[i] for i in filt_idx]]
    endpoints = [valid_eps[i] for i in filt_idx]
    print(f"Endpoints: {len(endpoints)}, Skipped: {[s[0] for s in skipped]}")

    # Convert SMILES to PyG graphs (store labels as Data.y)
    print("Converting SMILES to PyG graphs...")
    tr_graphs, tr_kept_Y = [], []
    for i in range(n_tr):
        g = smiles_to_pyg(tr_smiles[i])
        if g is not None:
            g.y = torch.FloatTensor(np.nan_to_num(Y_tr[i], nan=0.0))
            tr_graphs.append(g)
            tr_kept_Y.append(np.nan_to_num(Y_tr[i], nan=0.0))

    te_graphs, te_kept_Y = [], []
    for i in range(n_te):
        g = smiles_to_pyg(te_smiles[i])
        if g is not None:
            g.y = torch.FloatTensor(np.nan_to_num(Y_te[i], nan=0.0))
            te_graphs.append(g)
            te_kept_Y.append(np.nan_to_num(Y_te[i], nan=0.0))

    Y_tr_kept = np.array(tr_kept_Y)
    Y_te_kept = np.array(te_kept_Y)

    print(f"Train: {len(tr_graphs)}, Test: {len(te_graphs)}")
    node_feat = tr_graphs[0].x.shape[1]
    edge_feat = tr_graphs[0].edge_attr.shape[1]
    print(f"Node features: {node_feat}, Edge features: {edge_feat}")

    return {
        "tr_graphs": tr_graphs, "Y_tr": Y_tr_kept,
        "te_graphs": te_graphs, "Y_te": Y_te_kept,
        "endpoints": endpoints, "node_feat": node_feat, "edge_feat": edge_feat,
    }


def train_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels = batch.y.view(logits.shape)
        valid = ~torch.isnan(labels)
        labels_c = torch.where(valid, labels, torch.zeros_like(labels))
        loss = F.binary_cross_entropy_with_logits(
            logits, labels_c, weight=valid.float(), pos_weight=pos_weight.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, graphs, batch_size, device):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def run_trial(data, trial, device, n_epochs=200, patience=20, bs=128):
    """One Optuna trial with 5-fold CV."""
    gnn_out = trial.suggest_categorical("gnn_out_feats", [64, 128, 256])
    n_layers = trial.suggest_int("num_layers", 2, 4)
    n_ts = trial.suggest_int("num_timesteps", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    print(f"\n  Trial {trial.number}: gnn_out={gnn_out}, layers={n_layers}, ts={n_ts}, "
          f"dropout={dropout:.3f}, lr={lr:.5f}, wd={wd:.6f}", flush=True)

    n_tasks = len(data["endpoints"])
    pos_weight = compute_pos_weight(data["Y_tr"])
    n = len(data["tr_graphs"])
    strat = data["Y_tr"][:, 0]
    skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)

    fold_auprcs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n), strat)):
        tr_ds = [data["tr_graphs"][i] for i in tr_idx]
        va_ds = [data["tr_graphs"][i] for i in va_idx]
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False)

        model = AttentiveFPModel(
            in_channels=data["node_feat"], hidden_channels=gnn_out,
            out_channels=n_tasks, edge_dim=data["edge_feat"],
            num_layers=n_layers, num_timesteps=n_ts, dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=10)
        best_auprc, wait = 0, 0
        print(f"    Fold {fold+1}/5: train={len(tr_ds)}, val={len(va_ds)}", flush=True)

        for epoch in range(n_epochs):
            train_epoch(model, tr_loader, optimizer, pos_weight, device)
            # Evaluate on validation
            va_probs = predict(model, va_ds, bs, device)
            va_labels = np.array([g.y.numpy() for g in va_ds])

            auprcs = []
            for j in range(n_tasks):
                y_t = va_labels[:, j]
                valid = ~np.isnan(y_t)
                if valid.sum() > 0 and len(np.unique(y_t[valid])) >= 2:
                    auprcs.append(average_precision_score(y_t[valid], va_probs[valid, j]))
            mean_auprc = np.mean(auprcs) if auprcs else 0
            scheduler.step(mean_auprc)

            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        fold_auprcs.append(best_auprc)
        print(f"    Fold {fold+1} done: best_AUPRC={best_auprc:.4f} (epochs={epoch+1})", flush=True)

    mean_auprc = np.mean(fold_auprcs)
    print(f"  Trial {trial.number} done: mean_AUPRC={mean_auprc:.4f}", flush=True)
    return mean_auprc


def train_final(data, params, device, n_epochs=200, patience=20, bs=128):
    """Train final model on full training set."""
    n_tasks = len(data["endpoints"])
    pos_weight = compute_pos_weight(data["Y_tr"])

    model = AttentiveFPModel(
        in_channels=data["node_feat"], hidden_channels=params["gnn_out_feats"],
        out_channels=n_tasks, edge_dim=data["edge_feat"],
        num_layers=params["num_layers"], num_timesteps=params["num_timesteps"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=10)

    tr_loader = DataLoader(data["tr_graphs"], batch_size=bs, shuffle=True)

    best_auprc, best_state, wait = 0, None, 0
    for epoch in range(n_epochs):
        loss = train_epoch(model, tr_loader, optimizer, pos_weight, device)
        # Quick train AUPRC
        tr_probs = predict(model, data["tr_graphs"], bs*2, device)
        tr_labels = data["Y_tr"]
        auprcs = []
        for j in range(n_tasks):
            y_t = tr_labels[:, j]
            valid = ~np.isnan(y_t)
            if valid.sum() > 0 and len(np.unique(y_t[valid])) >= 2:
                auprcs.append(average_precision_score(y_t[valid], tr_probs[valid, j]))
        mean_auprc = np.mean(auprcs) if auprcs else 0
        scheduler.step(mean_auprc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, AUPRC={mean_auprc:.4f}")

        if mean_auprc > best_auprc:
            best_auprc = mean_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = prepare_data()

    # ── Optuna tuning ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Optuna tuning ({config.N_OPTUNA_TRIALS} trials)")
    print("=" * 60)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.SEED))
    study.optimize(lambda trial: run_trial(data, trial, device), n_trials=config.N_OPTUNA_TRIALS)
    print(f"\nBest CV AUPRC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # ── Final model ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training final model")
    print("=" * 60)

    model = train_final(data, study.best_params, device)

    # Save
    model_path = os.path.join(config.MODELS_DIR, "attentivefp_multitask.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "best_params": study.best_params,
        "endpoints": data["endpoints"],
        "node_feat": data["node_feat"],
        "edge_feat": data["edge_feat"],
    }, model_path)
    print(f"Model saved: {model_path}")

    # ── Test evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test evaluation")
    print("=" * 60)

    test_probs = predict(model, data["te_graphs"], 256, device)
    test_labels = data["Y_te"]

    results = []
    for i, ep in enumerate(data["endpoints"]):
        y_t, y_p = test_labels[:, i], test_probs[:, i]
        valid = ~np.isnan(y_t)
        y_t, y_p = y_t[valid], y_p[valid]
        if len(np.unique(y_t)) < 2:
            continue
        m = compute_metrics(y_t, y_p)
        results.append({
            "algorithm": "AttentiveFP", "dataset": "tox21" if ep.startswith("Tox21") else "ctd",
            "endpoint": ep, **m,
            "n_train": len(data["tr_graphs"]), "n_test": len(data["te_graphs"]),
            "n_pos_train": int(np.nansum(data["Y_tr"][:, i])),
            "n_pos_test": int(np.nansum(data["Y_te"][:, i])),
        })
        print(f"  {ep}: AUC-ROC={m['auc_roc']:.4f}, AUPRC={m['auprc']:.4f}, F1={m['f1']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(config.TRADITIONAL_ML_RESULTS, "attentivefp_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    if len(results_df) > 0:
        print(f"\n{'='*60}")
        print("AttentiveFP Summary")
        print("=" * 60)
        for metric in ["auc_roc", "auprc", "f1"]:
            vals = results_df[metric].dropna()
            print(f"  {metric}: {vals.mean():.4f} +/- {vals.std():.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
