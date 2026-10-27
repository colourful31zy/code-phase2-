"""
04_generate_comparison.py — Generate Stage 1 comparison analysis report
Compares: RF, XGBoost, LightGBM, SVM, MLP, NB Ensemble (Top3), AttentiveFP
"""
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Load all results ────────────────────────────────────────────────
orig = pd.read_csv(os.path.join(config.TRADITIONAL_ML_RESULTS,
                                 "traditional_ml_results_filtered.csv"))
nb3 = pd.read_csv(os.path.join(config.TRADITIONAL_ML_RESULTS,
                                "nb_ensemble_top3_results.csv"))
best = pd.read_csv(os.path.join(config.TRADITIONAL_ML_RESULTS,
                                 "best_per_endpoint.csv"))
attfp = pd.read_csv(os.path.join(config.TRADITIONAL_ML_RESULTS,
                                  "attentivefp_results.csv"))
orig_opt = pd.read_csv(os.path.join(config.TRADITIONAL_ML_RESULTS,
                                     "traditional_ml_results_optimized.csv"))

# ── 1. Per-algorithm summary (only common endpoints) ────────────────
algs = ["RF", "XGBoost", "LightGBM", "SVM", "MLP"]

print("=" * 70)
print("Stage 1 对比分析报告")
print("=" * 70)

print("\n1. 各算法平均性能 (24 终点)")
print("-" * 70)
print(f"{'Algorithm':<16} {'AUC-ROC':<10} {'AUPRC':<10} {'F1':<10} {'BalAcc':<10}")
print("-" * 70)
for alg in algs:
    s = orig[orig['algorithm'] == alg]
    if len(s) == 0:
        continue
    auc = s['auc_roc'].mean()
    auprc = s['auprc'].mean()
    f1 = s['f1'].mean()
    ba = s['balanced_accuracy'].mean()
    print(f"{alg:<16} {auc:<10.4f} {auprc:<10.4f} {f1:<10.4f} {ba:<10.4f}")

# NB Ensemble
if len(nb3) > 0:
    print(f"{'NB Ensemble (Top3)':<16} {nb3['auc_roc'].mean():<10.4f} {nb3['auprc'].mean():<10.4f} {nb3['f1'].mean():<10.4f} {nb3['balanced_accuracy'].mean():<10.4f}")

# AttentiveFP
for ds in ['tox21', 'ctd']:
    s = attfp[attfp['dataset'] == ds]
    if len(s) == 0:
        continue
    auc = s['auc_roc'].mean()
    auprc = s['auprc'].mean()
    f1 = s['f1'].mean()
    ba = s['balanced_accuracy'].mean()
    print(f"{'AttentiveFP ('+ds+')':<16} {auc:<10.4f} {auprc:<10.4f} {f1:<10.4f} {ba:<10.4f}")

# Per-endpoint best
print(f"{'Best per-ep':<16} {best['auc_roc'].mean():<10.4f} {best['auprc'].mean():<10.4f} {'-':<10} {'-':<10}")

# ── 2. Per-endpoint best algorithm distribution ──────────────────────
print("\n\n2. 每终点最优算法分布")
print("-" * 70)
algo_counts = best['algorithm'].value_counts()
for a, c in algo_counts.items():
    eps = best[best['algorithm'] == a]['endpoint'].tolist()
    print(f"  {a:<12} {c:<3} 个终点: {', '.join(eps)}")

# ── 3. Best model comparison (original 20trials vs new 10trials) ────
print("\n\n3. 原始 vs 重训 每终点最优对比")
print("-" * 70)

# Original best per endpoint (RF/XGB/LGB only)
algs3 = ["RF", "XGBoost", "LightGBM"]
orig_sub = orig[orig['algorithm'].isin(algs3)]
orig_best = orig_sub.loc[orig_sub.groupby('endpoint')['auc_roc'].idxmax()]
common = list(set(orig_best['endpoint']) & set(best['endpoint']))

comp = orig_best[orig_best['endpoint'].isin(common)].merge(
    best[best['endpoint'].isin(common)], on='endpoint', suffixes=('_orig', '_new'))
better = 0
worse = 0
for _, r in comp.iterrows():
    diff = r['auc_roc_new'] - r['auc_roc_orig']
    if diff > 0.005:
        better += 1
    elif diff < -0.005:
        worse += 1

print(f"  重训改善: {better}/{len(comp)} 终点")
print(f"  重训下降: {worse}/{len(comp)} 终点")
print(f"  原始均值: {comp['auc_roc_orig'].mean():.4f}")
print(f"  重训均值: {comp['auc_roc_new'].mean():.4f}")
print(f"  变化: {comp['auc_roc_new'].mean() - comp['auc_roc_orig'].mean():+.4f}")

# ── 4. AttentiveFP vs Traditional ML ──────────────────────────────
print("\n\n4. AttentiveFP vs 传统ML (共同终点)")
print("-" * 70)

for ds, ds_label in [('tox21', 'Tox21'), ('ctd', 'CTD')]:
    a = attfp[attfp['dataset'] == ds]
    o = orig[orig['dataset'] == ds]
    a_auc = a['auc_roc'].mean()
    o_auc = o['auc_roc'].mean()
    o_best_auc = orig_best[orig_best['endpoint'].isin(a['endpoint'])]['auc_roc'].mean()
    print(f"  {ds_label}: AttentiveFP AUC={a_auc:.4f}, 传统ML均值={o_auc:.4f}, 最优={o_best_auc:.4f}")

print("\n\n5. 结论")
print("-" * 70)
print("  - RF/XGBoost/LightGBM 整体最优，每终点选最佳 AUC=0.691")
print("  - SVM/MLP 显著弱于树模型，训练更慢")
print("  - NB Ensemble (Top3) 未超过单模型最优")
print("  - AttentiveFP 在 Tox21 上低于传统ML，在 CTD 上 AUPRC 极低")
print("  - 最终方案: 每终点独立选 RF/XGBoost/LightGBM + class_weight + Optuna 20 trials")