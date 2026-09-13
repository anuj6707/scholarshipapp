"""
ml/pipeline.py — Train scholarship recommendation models.

Compares Logistic Regression, Random Forest, and XGBoost for each of 32
scholarship targets. Picks the best model per target using 5-fold
stratified cross-validation (F1 score). Saves a single artifact.

Usage:  python ml/pipeline.py
"""
import os, sys, time, warnings, joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

# Allow running as script: python ml/pipeline.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.features import (
    RAW_FEATURES, TARGET_COLUMNS, ENGINEERED_FEATURES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ALL_FEATURES,
    engineer_features,
)

warnings.filterwarnings('ignore')

# ─── Model Configs ───────────────────────────────────────────────────────────
# Each config: (label, model_constructor). Keep it simple.
MODELS = [
    ("LR_C0.1_bal",   lambda: LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)),
    ("LR_C1_bal",     lambda: LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=42)),
    ("LR_C10",        lambda: LogisticRegression(C=10, max_iter=1000, random_state=42)),
    ("RF_100_d8_bal",  lambda: RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("RF_200_bal",     lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("RF_100",         lambda: RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("XGB_100_d3",     lambda: XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')),
    ("XGB_100_d5",     lambda: XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')),
    ("XGB_200_d5",     lambda: XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')),
]

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ─── Preprocessing ───────────────────────────────────────────────────────────

def fit_preprocessor(X):
    """Fit scaler + encoder on training data. Returns a dict (safe to serialize)."""
    scaler = StandardScaler().fit(X[NUMERICAL_FEATURES])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X[CATEGORICAL_FEATURES])
    return {
        'scaler': scaler, 'encoder': encoder,
        'numerical_features': list(NUMERICAL_FEATURES),
        'categorical_features': list(CATEGORICAL_FEATURES),
    }

def transform(prep, X):
    """Apply fitted preprocessor."""
    return np.hstack([
        prep['scaler'].transform(X[prep['numerical_features']]),
        prep['encoder'].transform(X[prep['categorical_features']]),
    ])


# ─── Cross-Validation ────────────────────────────────────────────────────────

def cv_evaluate(X_feat, y, model_fn, skf):
    """
    Run 5-fold stratified CV. Returns best threshold and mean metrics.
    Preprocessor is fitted inside each fold (no leakage).
    """
    oof_probs = np.full(len(y), np.nan)
    f1s, aucs = [], []

    for tr_idx, val_idx in skf.split(X_feat, y):
        prep = fit_preprocessor(X_feat.iloc[tr_idx])
        X_tr = transform(prep, X_feat.iloc[tr_idx])
        X_val = transform(prep, X_feat.iloc[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') and len(np.unique(y_tr)) > 1 else np.zeros(len(y_val))
        oof_probs[val_idx] = probs

        preds = (probs >= 0.5).astype(int)
        f1s.append(f1_score(y_val, preds, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5)
        except: aucs.append(0.5)

    # Find best threshold on OOF predictions
    valid = ~np.isnan(oof_probs)
    best_thr, best_f1 = 0.5, 0
    for t in THRESHOLDS:
        f = f1_score(y[valid], (oof_probs[valid] >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t

    return {'f1': np.mean(f1s), 'auc': np.mean(aucs), 'threshold': best_thr, 'thr_f1': best_f1}


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("SCHOLARSHIP ML PIPELINE")
    print("=" * 70)

    # Load data
    csv_path = os.path.join(_ROOT, 'data', 'scholarship_dataset.csv')
    raw = pd.read_csv(csv_path)
    df = engineer_features(raw[RAW_FEATURES].copy())
    X_all = df[ALL_FEATURES]
    print(f"\nDataset: {len(raw)} rows, {len(ALL_FEATURES)} features, {len(TARGET_COLUMNS)} targets")

    # Class distribution
    print(f"\n{'Target':<35} {'Pos':>5} {'Neg':>5} {'%':>6}")
    print("-" * 55)
    for t in TARGET_COLUMNS:
        p = int(raw[t].sum())
        print(f"{t:<35} {p:>5} {len(raw)-p:>5} {100*p/len(raw):>5.1f}%")

    # 80/20 split
    X_train, X_test, idx_tr, idx_te = train_test_split(X_all, raw.index, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results, final_models = [], {}

    # Final preprocessor on full training set (for artifact)
    final_prep = fit_preprocessor(X_train)
    X_train_proc = transform(final_prep, X_train)
    X_test_proc = transform(final_prep, X_test)
    feat_names = list(NUMERICAL_FEATURES) + list(final_prep['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))

    start = time.time()

    for t_idx, target in enumerate(TARGET_COLUMNS, 1):
        y_tr = raw.loc[idx_tr, target].values
        y_te = raw.loc[idx_te, target].values
        pos = int(y_tr.sum())

        if pos < 5 or (len(y_tr) - pos) < 5:
            print(f"[{t_idx:02d}] {target}: SKIP (too few samples)")
            continue

        # Try all models, pick best by CV F1
        best, best_label, best_fn = None, None, None
        for label, model_fn in MODELS:
            try:
                result = cv_evaluate(X_train, y_tr, model_fn, skf)
                all_results.append({'target': target, 'model': label, **result})
                if best is None or result['thr_f1'] > best['thr_f1']:
                    best, best_label, best_fn = result, label, model_fn
            except Exception as e:
                pass

        if best is None:
            print(f"[{t_idx:02d}] {target}: FAILED")
            continue

        # Retrain best on full training set
        model = best_fn()
        model.fit(X_train_proc, y_tr)

        # Test evaluation
        probs = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') and len(np.unique(y_tr)) > 1 else np.zeros(len(y_te))
        preds = (probs >= best['threshold']).astype(int)
        t_f1 = f1_score(y_te, preds, zero_division=0)
        try: t_auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
        except: t_auc = 0.5

        model_type = best_label.split('_')[0]  # LR, RF, or XGB
        print(f"[{t_idx:02d}] {target:<35} -> {best_label:<16} CV-F1={best['thr_f1']:.4f}  Test-F1={t_f1:.4f}  thr={best['threshold']}")

        # Feature importance
        fi = {}
        if hasattr(model, 'feature_importances_'):
            pairs = sorted(zip(feat_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
            fi = {name: round(float(imp), 6) for name, imp in pairs[:10]}
        elif hasattr(model, 'coef_'):
            pairs = sorted(zip(feat_names, np.abs(model.coef_[0])), key=lambda x: x[1], reverse=True)
            fi = {name: round(float(imp), 6) for name, imp in pairs[:10]}

        final_models[target] = {
            'model': model, 'model_type': model_type, 'pca': None,
            'threshold': best['threshold'],
            'cv_metrics': {'f1': best['f1'], 'auc': best['auc']},
            'test_metrics': {'f1': t_f1, 'auc': t_auc, 'accuracy': accuracy_score(y_te, preds)},
            'feature_importance': fi,
        }

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Summary
    counts = Counter(m['model_type'] for m in final_models.values())
    avg_f1 = np.mean([m['test_metrics']['f1'] for m in final_models.values()])
    avg_auc = np.mean([m['test_metrics']['auc'] for m in final_models.values()])
    print(f"\nModel selection: {dict(counts.most_common())}")
    print(f"Avg test F1: {avg_f1:.4f}, Avg test AUC: {avg_auc:.4f}")

    # Save comparison CSV
    pd.DataFrame(all_results).to_csv(os.path.join(os.path.dirname(__file__), 'model_comparison.csv'), index=False)

    # Save feature importance CSV
    fi_rows = []
    for target, info in final_models.items():
        for feat, imp in info.get('feature_importance', {}).items():
            fi_rows.append({'target': target, 'feature': feat, 'importance': imp})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(os.path.join(os.path.dirname(__file__), 'feature_importance.csv'), index=False)

    # Save artifact
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
    pkl_path = os.path.join(os.path.dirname(__file__), 'scholarship_system.pkl')
    joblib.dump(artifact, pkl_path, compress=3)
    mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"\nSaved: {pkl_path} ({mb:.2f} MB)")
    print("=" * 70)


if __name__ == '__main__':
    run()
