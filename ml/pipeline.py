"""
ml/pipeline.py — Comparative ML Pipeline for Scholarship Recommendation

Compares Logistic Regression, Random Forest, and XGBoost across all 32 scholarship
targets using 5-fold stratified cross-validation, optional PCA, threshold optimization,
and rigorous train/test separation.

Usage:
    python ml/pipeline.py
"""
import os
import sys

# Ensure project root is on sys.path for both script and module usage
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

from ml.features import (
    RAW_FEATURES, TARGET_COLUMNS, ENGINEERED_FEATURES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    engineer_features
)

warnings.filterwarnings('ignore')

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def fit_preprocessor(X_train):
    """Fit scaler and encoder on training data. Returns dict (cross-version safe)."""
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERICAL_FEATURES])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    return {
        'scaler': scaler,
        'encoder': encoder,
        'numerical_features': list(NUMERICAL_FEATURES),
        'categorical_features': list(CATEGORICAL_FEATURES),
    }


def transform(preprocessor, X):
    """Transform features using fitted preprocessor dict."""
    num_data = preprocessor['scaler'].transform(X[preprocessor['numerical_features']])
    cat_data = preprocessor['encoder'].transform(X[preprocessor['categorical_features']])
    return np.hstack([num_data, cat_data])


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def get_model_configs():
    """Returns list of model configurations to evaluate."""
    configs = []

    # Logistic Regression (8 configs)
    for C in [0.01, 0.1, 1, 10]:
        for cw in [None, 'balanced']:
            configs.append({
                'name': 'LogisticRegression',
                'params': {'C': C, 'class_weight': cw, 'max_iter': 1000,
                           'random_state': 42, 'solver': 'lbfgs'},
            })

    # Random Forest (8 configs)
    for n_est in [100, 200]:
        for depth in [8, None]:
            for cw in [None, 'balanced']:
                configs.append({
                    'name': 'RandomForest',
                    'params': {'n_estimators': n_est, 'max_depth': depth,
                               'min_samples_leaf': 2, 'class_weight': cw,
                               'random_state': 42, 'n_jobs': -1},
                })

    # XGBoost (8 configs)
    for n_est in [100, 200]:
        for depth in [3, 5]:
            for lr in [0.05, 0.1]:
                configs.append({
                    'name': 'XGBoost',
                    'params': {'n_estimators': n_est, 'max_depth': depth,
                               'learning_rate': lr, 'random_state': 42,
                               'n_jobs': -1, 'eval_metric': 'logloss'},
                })

    return configs


def create_model(config):
    """Instantiate a fresh model from config dict."""
    name = config['name']
    params = config['params']
    if name == 'LogisticRegression':
        return LogisticRegression(**params)
    elif name == 'RandomForest':
        return RandomForestClassifier(**params)
    elif name == 'XGBoost':
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


# ============================================================
# CROSS-VALIDATION FOR ONE CONFIG
# ============================================================

def evaluate_config(X_features, y, config, use_pca, skf):
    """
    Run 5-fold stratified CV for one model config on one target.
    Preprocessor is fitted inside each fold (no data leakage).
    Returns dict with mean CV metrics, best threshold, and OOF F1.
    """
    n = len(y)
    oof_probs = np.full(n, np.nan)
    fold_metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}

    for train_idx, val_idx in skf.split(X_features, y):
        X_tr = X_features.iloc[train_idx]
        X_val = X_features.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Fit preprocessor on fold train only
        prep = fit_preprocessor(X_tr)
        X_tr_proc = transform(prep, X_tr)
        X_val_proc = transform(prep, X_val)

        # Optional PCA fitted on fold train only
        if use_pca:
            pca = PCA(n_components=0.95, random_state=42)
            X_tr_proc = pca.fit_transform(X_tr_proc)
            X_val_proc = pca.transform(X_val_proc)

        # Train model
        model = create_model(config)
        model.fit(X_tr_proc, y_tr)

        # Predict probabilities
        if hasattr(model, 'predict_proba') and len(np.unique(y_tr)) > 1:
            probs = model.predict_proba(X_val_proc)[:, 1]
        else:
            probs = model.predict(X_val_proc).astype(float)

        oof_probs[val_idx] = probs

        # Fold metrics at default threshold 0.5
        preds = (probs >= 0.5).astype(int)
        fold_metrics['accuracy'].append(accuracy_score(y_val, preds))
        fold_metrics['precision'].append(precision_score(y_val, preds, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_val, preds, zero_division=0))
        fold_metrics['f1'].append(f1_score(y_val, preds, zero_division=0))
        try:
            auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        except Exception:
            auc = 0.5
        fold_metrics['roc_auc'].append(auc)

    # Threshold optimization on OOF predictions
    valid = ~np.isnan(oof_probs)
    best_threshold, best_f1 = 0.5, 0.0
    for t in THRESHOLDS:
        f1_t = f1_score(y[valid], (oof_probs[valid] >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    return {
        'cv_accuracy_mean': np.mean(fold_metrics['accuracy']),
        'cv_accuracy_std': np.std(fold_metrics['accuracy']),
        'cv_precision_mean': np.mean(fold_metrics['precision']),
        'cv_recall_mean': np.mean(fold_metrics['recall']),
        'cv_f1_mean': np.mean(fold_metrics['f1']),
        'cv_f1_std': np.std(fold_metrics['f1']),
        'cv_roc_auc_mean': np.mean(fold_metrics['roc_auc']),
        'cv_roc_auc_std': np.std(fold_metrics['roc_auc']),
        'best_threshold': best_threshold,
        'threshold_f1': best_f1,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, config_name, feature_names):
    """Extract top-10 feature importances from fitted model."""
    if config_name in ('RandomForest', 'XGBoost') and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif config_name == 'LogisticRegression' and hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return []

    if len(importances) != len(feature_names):
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs[:10]


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Main pipeline: load data, compare models, select best, save artifact."""
    print("=" * 80)
    print("SCHOLARSHIP ML COMPARATIVE PIPELINE")
    print("=" * 80)

    # --- Load data ---
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'scholarship_dataset.csv')
    raw_df = pd.read_csv(csv_path)
    print(f"\nLoaded dataset: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    # Verify columns
    for col in RAW_FEATURES + TARGET_COLUMNS:
        assert col in raw_df.columns, f"Missing column: {col}"

    # Engineer features dynamically from raw features only
    df = engineer_features(raw_df[RAW_FEATURES].copy())
    X_all = df[ALL_FEATURES]

    print(f"Features after engineering: {len(ALL_FEATURES)} "
          f"({len(CATEGORICAL_FEATURES)} cat + {len(NUMERICAL_FEATURES)} num)")
    print(f"Targets: {len(TARGET_COLUMNS)} scholarships")

    # --- Class distribution ---
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION (full dataset)")
    print(f"{'='*80}")
    print(f"{'Scholarship':<35} {'Pos':>6} {'Neg':>6} {'Pos%':>7}")
    print("-" * 56)
    for target in TARGET_COLUMNS:
        pos = int(raw_df[target].sum())
        neg = len(raw_df) - pos
        print(f"{target:<35} {pos:>6} {neg:>6} {100*pos/len(raw_df):>6.1f}%")

    # --- 80/20 train/test split ---
    X_train_feat, X_test_feat, idx_train, idx_test = train_test_split(
        X_all, raw_df.index, test_size=0.20, random_state=42
    )
    y_train_all = {t: raw_df.loc[idx_train, t].values for t in TARGET_COLUMNS}
    y_test_all = {t: raw_df.loc[idx_test, t].values for t in TARGET_COLUMNS}
    print(f"\nTrain: {len(X_train_feat)}, Test: {len(X_test_feat)} (held out)")

    # --- Model configs ---
    model_configs = get_model_configs()
    pca_options = [False, True]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    total_cfgs = len(model_configs) * len(pca_options)
    print(f"Configs per target: {total_cfgs} ({len(model_configs)} models x {len(pca_options)} PCA)")
    print(f"Total experiments: {total_cfgs * len(TARGET_COLUMNS)}\n")

    # --- Cross-validation ---
    all_results = []
    best_per_target = {}
    start = time.time()

    for t_idx, target in enumerate(TARGET_COLUMNS, 1):
        y_train = y_train_all[target]
        pos_train = int(y_train.sum())
        pos_pct = 100 * pos_train / len(y_train)

        print(f"[{t_idx:02d}/32] {target}  (pos={pos_train}/{len(y_train)}, {pos_pct:.0f}%)")

        if pos_train < 5 or (len(y_train) - pos_train) < 5:
            print(f"  SKIP — extreme class imbalance")
            continue

        target_results = []
        for config in model_configs:
            for use_pca in pca_options:
                pca_str = "PCA95" if use_pca else "NoPCA"
                try:
                    result = evaluate_config(X_train_feat, y_train, config, use_pca, skf)
                    row = {
                        'scholarship': target,
                        'model': config['name'],
                        'pca': pca_str,
                        'params': str({k: v for k, v in config['params'].items()
                                       if k not in ('random_state', 'n_jobs', 'solver',
                                                     'max_iter', 'eval_metric')}),
                        **result,
                    }
                    target_results.append((row, config, use_pca))
                    all_results.append(row)
                except Exception as e:
                    print(f"  ERROR {config['name']}|{pca_str}: {e}")

        if target_results:
            best_row, best_cfg, best_pca = max(target_results, key=lambda x: x[0]['cv_f1_mean'])
            best_per_target[target] = (best_row, best_cfg, best_pca)
            print(f"  BEST: {best_row['model']}|{best_row['pca']}  "
                  f"CV-F1={best_row['cv_f1_mean']:.4f}  thr={best_row['best_threshold']}")

    elapsed = time.time() - start
    print(f"\nCV completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Final test evaluation ---
    print(f"\n{'='*80}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*80}")

    # Fit final preprocessor on full training set
    final_prep = fit_preprocessor(X_train_feat)
    X_train_proc = transform(final_prep, X_train_feat)
    X_test_proc = transform(final_prep, X_test_feat)

    # Feature names after preprocessing
    num_names = list(NUMERICAL_FEATURES)
    cat_names = list(final_prep['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))
    all_proc_names = num_names + list(cat_names)

    final_models = {}
    fi_rows = []

    print(f"\n{'Target':<35} {'Model':<18} {'PCA':<6} {'TestF1':>7} {'TestAUC':>8} {'TestAcc':>8}")
    print("-" * 85)

    for target in TARGET_COLUMNS:
        if target not in best_per_target:
            continue

        best_row, best_cfg, use_pca = best_per_target[target]
        y_train = y_train_all[target]
        y_test = y_test_all[target]
        threshold = best_row['best_threshold']

        # Apply PCA if selected
        X_tr = X_train_proc.copy()
        X_te = X_test_proc.copy()
        pca_obj = None
        if use_pca:
            pca_obj = PCA(n_components=0.95, random_state=42)
            X_tr = pca_obj.fit_transform(X_tr)
            X_te = pca_obj.transform(X_te)

        # Train final model on full training set
        model = create_model(best_cfg)
        model.fit(X_tr, y_train)

        # Evaluate on test set
        if hasattr(model, 'predict_proba') and len(np.unique(y_train)) > 1:
            test_probs = model.predict_proba(X_te)[:, 1]
        else:
            test_probs = model.predict(X_te).astype(float)

        test_preds = (test_probs >= threshold).astype(int)
        t_acc = accuracy_score(y_test, test_preds)
        t_prec = precision_score(y_test, test_preds, zero_division=0)
        t_rec = recall_score(y_test, test_preds, zero_division=0)
        t_f1 = f1_score(y_test, test_preds, zero_division=0)
        try:
            t_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
        except Exception:
            t_auc = 0.5

        pca_label = "PCA95" if use_pca else "NoPCA"
        print(f"{target:<35} {best_row['model']:<18} {pca_label:<6} {t_f1:>7.4f} {t_auc:>8.4f} {t_acc:>8.4f}")

        final_models[target] = {
            'model': model,
            'model_type': best_row['model'],
            'pca': pca_obj,
            'threshold': threshold,
            'cv_metrics': {
                'cv_f1_mean': best_row['cv_f1_mean'],
                'cv_f1_std': best_row['cv_f1_std'],
                'cv_accuracy_mean': best_row['cv_accuracy_mean'],
                'cv_roc_auc_mean': best_row['cv_roc_auc_mean'],
            },
            'test_metrics': {
                'accuracy': t_acc, 'precision': t_prec,
                'recall': t_rec, 'f1': t_f1, 'roc_auc': t_auc,
            },
        }

        # Mark selected row in all_results
        for r in all_results:
            if (r['scholarship'] == target and r['model'] == best_row['model']
                    and r['pca'] == best_row['pca'] and r['params'] == best_row['params']):
                r['test_f1'] = t_f1
                r['test_roc_auc'] = t_auc
                r['test_accuracy'] = t_acc
                r['selected'] = True
                break

        # Feature importance
        feat_names = all_proc_names if not use_pca else [f'PC{i+1}' for i in range(X_tr.shape[1])]
        for fname, imp in get_feature_importance(model, best_row['model'], feat_names):
            fi_rows.append({'scholarship': target, 'model_type': best_row['model'],
                            'feature': fname, 'importance': round(imp, 6)})

    # --- Save comparison CSV ---
    comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_comparison.csv')
    pd.DataFrame(all_results).to_csv(comp_path, index=False)
    print(f"\nSaved: {comp_path} ({len(all_results)} rows)")

    # --- Save feature importance CSV ---
    if fi_rows:
        fi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_importance.csv')
        pd.DataFrame(fi_rows).to_csv(fi_path, index=False)
        print(f"Saved: {fi_path}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for mn in ['LogisticRegression', 'RandomForest', 'XGBoost']:
        for ps in ['NoPCA', 'PCA95']:
            subset = [r for r in all_results if r['model'] == mn and r['pca'] == ps]
            if subset:
                print(f"  {mn:<22} {ps:<6}  AvgCvF1={np.mean([r['cv_f1_mean'] for r in subset]):.4f}  "
                      f"AvgCvAUC={np.mean([r['cv_roc_auc_mean'] for r in subset]):.4f}")

    # Best model selection frequency
    print("\nBest model selection:")
    for name, count in Counter(m['model_type'] for m in final_models.values()).most_common():
        print(f"  {name}: {count}/32 scholarships")
    pca_counts = Counter('PCA' if m['pca'] is not None else 'NoPCA' for m in final_models.values())
    for name, count in pca_counts.most_common():
        print(f"  {name}: {count}/32 scholarships")

    # Test performance
    if final_models:
        tf1 = np.mean([m['test_metrics']['f1'] for m in final_models.values()])
        tauc = np.mean([m['test_metrics']['roc_auc'] for m in final_models.values()])
        tacc = np.mean([m['test_metrics']['accuracy'] for m in final_models.values()])
        print(f"\nFinal test performance (avg across {len(final_models)} targets):")
        print(f"  F1:  {tf1:.4f}")
        print(f"  AUC: {tauc:.4f}")
        print(f"  Acc: {tacc:.4f}")

    # --- Save artifact ---
    artifact = {
        'preprocessor': final_prep,
        'models': final_models,
        'target_columns': TARGET_COLUMNS,
        'feature_columns': ALL_FEATURES,
        'raw_features': RAW_FEATURES,
        'engineered_features': ENGINEERED_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
    }

    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scholarship_system.pkl')
    joblib.dump(artifact, pkl_path, compress=3)
    mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"\nSaved artifact: {pkl_path} ({mb:.2f} MB)")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    run_pipeline()
