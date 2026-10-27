"""
Phase 2 公共工具函数
====================
数据加载、评估指标、模型构建、可视化。
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════


def load_meta(meta_file):
    """加载 split_meta.json，返回 dict。"""
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_features(features_file, feature_columns):
    """加载特征文件，只取 feature_columns 列。返回 (ids, smiles, X)。"""
    df = pd.read_csv(features_file)
    ids = df["ID"].values
    smiles = df["SMILES"].values
    X = df[feature_columns].values.astype(np.float32)
    return ids, smiles, X


def load_labels(labels_file, endpoints, id_type="smiles"):
    """
    加载标签文件，返回 (ids, Y)。
    自动检测 SMILES/smiles 列名。
    """
    df = pd.read_csv(labels_file)
    # 自动检测 SMILES 列名
    if "SMILES" in df.columns:
        ids = df["SMILES"].values
    elif "smiles" in df.columns:
        ids = df["smiles"].values
    elif "chemical_id" in df.columns:
        ids = df["chemical_id"].values
    else:
        raise ValueError(f"No ID column found in {labels_file}. Columns: {list(df.columns)}")
    Y = df[endpoints].values.astype(np.float32)
    return ids, Y


def get_valid_endpoints(Y_train, Y_test, endpoints, min_pos=5):
    """
    过滤掉训练集中正样本数 < min_pos 的终点。
    返回: (valid_indices, valid_endpoints, skipped_endpoints)
    """
    valid_idx = []
    valid_eps = []
    skipped = []
    for i, ep in enumerate(endpoints):
        n_pos = np.nansum(Y_train[:, i] == 1)
        if n_pos >= min_pos:
            valid_idx.append(i)
            valid_eps.append(ep)
        else:
            skipped.append((ep, int(n_pos)))
    return valid_idx, valid_eps, skipped


def match_features_labels(feat_ids, label_ids, Y):
    """
    按 ID 匹配特征和标签。
    返回: (matched_feat_indices, matched_label_indices, matched_Y)
    """
    feat_id_set = set(feat_ids)
    matched_feat_idx = []
    matched_label_idx = []
    for i, lid in enumerate(label_ids):
        if lid in feat_id_set:
            # 找到特征文件中对应的行
            idx = np.where(feat_ids == lid)[0]
            if len(idx) > 0:
                matched_feat_idx.append(idx[0])
                matched_label_idx.append(i)
    return np.array(matched_feat_idx), np.array(matched_label_idx), Y[matched_label_idx]


# ═══════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    计算二分类指标。
    返回 dict: auc_roc, auprc, f1, balanced_accuracy
    """
    # 处理全 0 或全 1 的情况
    if len(np.unique(y_true)) < 2:
        return {
            "auc_roc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "balanced_accuracy": np.nan,
        }

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def compute_metrics_cv(y_true_list, y_prob_list):
    """
    计算交叉验证的平均指标。
    返回 dict: mean_auc, mean_auprc, mean_f1, mean_ba
    """
    all_metrics = [compute_metrics(yt, yp) for yt, yp in zip(y_true_list, y_prob_list)]
    return {
        "mean_auc_roc": np.nanmean([m["auc_roc"] for m in all_metrics]),
        "mean_auprc": np.nanmean([m["auprc"] for m in all_metrics]),
        "mean_f1": np.nanmean([m["f1"] for m in all_metrics]),
        "mean_ba": np.nanmean([m["balanced_accuracy"] for m in all_metrics]),
    }


# ═══════════════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════════════


def build_model(algorithm, params, n_pos=None, n_neg=None):
    """
    根据算法名和参数构建 sklearn 兼容的模型。
    处理类别不平衡：
      - RF, SVM: class_weight='balanced'
      - XGBoost, LightGBM: scale_pos_weight
      - MLP: sample_weight (在 fit 时传入)
    """
    if algorithm == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif algorithm == "XGBoost":
        from xgboost import XGBClassifier
        spw = n_neg / max(n_pos, 1) if n_pos and n_neg else 1.0
        return XGBClassifier(
            **params,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    elif algorithm == "LightGBM":
        from lightgbm import LGBMClassifier
        spw = n_neg / max(n_pos, 1) if n_pos and n_neg else 1.0
        return LGBMClassifier(
            **params,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    elif algorithm == "SVM":
        from sklearn.svm import SVC
        return SVC(
            **params,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    elif algorithm == "MLP":
        from sklearn.neural_network import MLPClassifier
        hidden = params.get("hidden_layer_sizes", (128,))
        lr = params.get("learning_rate_init", 0.001)
        alpha = params.get("alpha", 0.0001)
        dropout = params.get("dropout", 0.2)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            learning_rate_init=lr,
            alpha=alpha,
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_sample_weight(y, algorithm):
    """
    为 MLP 生成 sample_weight（其他算法通过 class_weight/scale_pos_weight 处理）。
    """
    if algorithm != "MLP":
        return None
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        return np.ones(len(y))
    w_pos = n_neg / max(n_pos, 1)
    weights = np.where(y == 1, w_pos, 1.0)
    return weights


# ═══════════════════════════════════════════════════════
# NB 集成
# ═══════════════════════════════════════════════════════


def train_nb_ensemble(algorithm_probs_train, y_train, algorithm_probs_test):
    """
    NB 集成：将 5 种算法的预测概率作为特征，训练 GaussianNB。
    algorithm_probs_train: shape (n_samples, 5)
    algorithm_probs_test:  shape (n_samples, 5)
    返回: (y_prob_test, nb_model)
    """
    nb = GaussianNB()
    nb.fit(algorithm_probs_train, y_train)
    y_prob = nb.predict_proba(algorithm_probs_test)[:, 1]
    return y_prob, nb


# ═══════════════════════════════════════════════════════
# 文件 I/O
# ═══════════════════════════════════════════════════════


def save_model(model, path):
    """保存模型到 pickle 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """从 pickle 文件加载模型。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_results(results_list, path):
    """保存结果列表到 CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(path, index=False)
    return df


def print_progress(current, total, desc=""):
    """打印进度。"""
    pct = current / total * 100
    print(f"  [{current}/{total}] {pct:.0f}% {desc}")
