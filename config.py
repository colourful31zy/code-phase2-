"""
Phase 2 全局配置
================
所有脚本共享的路径、超参数、随机种子。
"""

import os
import numpy as np

# ── 路径 ──────────────────────────────────────────────
# 项目根目录 = code/phase2/ 往上两层
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SPLIT_DIR   = os.path.join(PROJECT_ROOT, "data", "02_processed", "phase2", "split")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "data", "02_processed", "phase2", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "02_processed", "phase2", "results")

TRAIN_FEATURES = os.path.join(SPLIT_DIR, "train_features.csv")
TEST_FEATURES  = os.path.join(SPLIT_DIR, "test_features.csv")

TOX21_TRAIN_LABELS = os.path.join(SPLIT_DIR, "tox21_train_labels.csv")
TOX21_TEST_LABELS  = os.path.join(SPLIT_DIR, "tox21_test_labels.csv")
CTD_TRAIN_LABELS   = os.path.join(SPLIT_DIR, "ctd_train_labels.csv")
CTD_TEST_LABELS    = os.path.join(SPLIT_DIR, "ctd_test_labels.csv")
META_FILE          = os.path.join(SPLIT_DIR, "split_meta.json")

# ── 随机种子 ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 训练配置 ──────────────────────────────────────────
N_OPTUNA_TRIALS = 20          # Optuna 调参试验次数 (OpenBayes 可改回 50)
N_CV_FOLDS = 5                # 交叉验证折数
OPTUNA_METRIC = "auprc"       # Optuna 优化指标
Y_RAND_REPEATS = 100          # Y-randomization 重复次数

# ── 算法列表 ──────────────────────────────────────────
ALGORITHMS = ["RF", "XGBoost", "LightGBM", "SVM", "MLP"]

# ── 算法超参数搜索空间 (Optuna) ──────────────────────
OPTUNA_SPACES = {
    "RF": {
        "n_estimators":      lambda t: t.suggest_int("n_estimators", 100, 1000),
        "max_depth":         lambda t: t.suggest_int("max_depth", 3, 30),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  lambda t: t.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "XGBoost": {
        "learning_rate":     lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         lambda t: t.suggest_int("max_depth", 3, 12),
        "subsample":         lambda t: t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight":  lambda t: t.suggest_int("min_child_weight", 1, 10),
        "gamma":             lambda t: t.suggest_float("gamma", 0, 5.0),
    },
    "LightGBM": {
        "learning_rate":     lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        lambda t: t.suggest_int("num_leaves", 20, 300),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 100),
        "subsample":         lambda t: t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    },
    "SVM": {
        "C":                 lambda t: t.suggest_float("C", 0.01, 100.0, log=True),
        "gamma":             lambda t: t.suggest_float("gamma", 1e-5, 1.0, log=True),
    },
    "MLP": {
        "hidden_layer_sizes": lambda t: t.suggest_categorical("hidden_layer_sizes", [
            (128,), (256,), (128, 64), (256, 128), (256, 128, 64)
        ]),
        "learning_rate_init": lambda t: t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "alpha":              lambda t: t.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "dropout":            lambda t: t.suggest_float("dropout", 0.1, 0.5),
    },
}

# ── 输出子目录 ────────────────────────────────────────
TRADITIONAL_ML_MODELS  = os.path.join(MODELS_DIR, "traditional_ml")
TRADITIONAL_ML_RESULTS = os.path.join(RESULTS_DIR, "evaluation")
NB_ENSEMBLE_RESULTS    = os.path.join(RESULTS_DIR, "evaluation")
Y_RAND_RESULTS         = os.path.join(RESULTS_DIR, "evaluation")

# ── Phase 1 数据路径 ─────────────────────────────────
PHASE1_DIR = os.path.join(PROJECT_ROOT, "data", "02_processed", "phase1")

# ── Stage 2 ───────────────────────────────────────────
STAGE2_RESULTS = os.path.join(RESULTS_DIR, "evaluation")
STAGE2_MODELS  = os.path.join(MODELS_DIR, "stage2")

# ── 特征列名 (从 split_meta.json 加载) ───────────────
FEATURE_COLUMNS = None  # 由脚本动态从 META_FILE 加载
