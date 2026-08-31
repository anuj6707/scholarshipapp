import os
import joblib
import pandas as pd
import numpy as np

def inspect_dataset(csv_path="data/synthetic_students.csv"):
    print("=" * 60)
    print(f"INSPECTING DATASET: {csv_path}")
    print("=" * 60)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    target_cols = [c for c in df.columns if c.startswith('target_')]
    feature_cols = [c for c in df.columns if not c.startswith('target_') and c != 'student_id']
    
    print(f"\nFeature Columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target Columns ({len(target_cols)}): {target_cols}")
    
    print("\nColumns, Data Types, and Missing Values:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        print(f"  - {col}: {df[col].dtype} (missing: {missing})")
    
    print("\nCategorical Features Unique Values:")
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            print(f"  - {col}: {df[col].unique().tolist()}")
            
    print("\nNumerical Features Range:")
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
            print(f"  - {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
            
    print("\nTarget Column Positive Counts:")
    for col in target_cols:
        pos = (df[col] == 1).sum()
        print(f"  - {col}: {pos}/{len(df)} ({pos/len(df)*100:.1f}%)")

def inspect_model_obj(pkl_path):
    print("\n" + "=" * 60)
    print(f"INSPECTING MODEL: {pkl_path}")
    print("=" * 60)
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return
    try:
        model = joblib.load(pkl_path)
        print(f"Loaded with joblib successfully.")
    except Exception as e:
        print(f"Joblib error: {e}")
        import pickle
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
            
    print(f"Model Class: {type(model)}")
    
    # Check if pipeline
    if hasattr(model, 'steps'):
        print("\nPipeline Steps:")
        for name, step in model.steps:
            print(f"  Step '{name}': {type(step)}")
            if hasattr(step, 'transformers_'):
                print(f"    Transformers in '{name}':")
                for t in step.transformers_:
                    print(f"      - {t[0]}: {type(t[1])}, columns: {t[2]}")
                    if hasattr(t[1], 'categories_'):
                        print(f"        categories: {t[1].categories_}")
                        
    # Check feature names
    if hasattr(model, 'feature_names_in_'):
        print(f"\nfeature_names_in_ ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
    elif hasattr(model, 'n_features_in_'):
        print(f"\nn_features_in_: {model.n_features_in_}")
        
    # Check MultiOutput / Estimators
    if hasattr(model, 'estimators_'):
        print(f"\nMultiOutput estimators count: {len(model.estimators_)}")
        print(f"Estimator 0 class: {type(model.estimators_[0])}")
        if hasattr(model.estimators_[0], 'classes_'):
            print(f"Estimator 0 classes: {model.estimators_[0].classes_}")
            
    # Check predict and predict_proba
    print(f"\nHas 'predict': {hasattr(model, 'predict')}")
    print(f"Has 'predict_proba': {hasattr(model, 'predict_proba')}")
    
    # Test sample prediction with first row of synthetic_students.csv
    try:
        df = pd.read_csv("data/synthetic_students.csv")
        target_cols = [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if not c.startswith('target_')]
        
        # Test passing DataFrame with feature columns
        sample_df = df[feature_cols].iloc[0:2]
        print(f"\nTesting predict with sample DataFrame (with columns {feature_cols}):")
        if 'student_id' in sample_df.columns and hasattr(model, 'feature_names_in_') and 'student_id' not in model.feature_names_in_:
            sample_df = sample_df.drop(columns=['student_id'])
            
        preds = model.predict(sample_df)
        print(f"predict shape: {preds.shape}, values:\n{preds}")
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(sample_df)
            print(f"predict_proba type: {type(probas)}")
            if isinstance(probas, list):
                print(f"predict_proba list len: {len(probas)}")
                print(f"proba for output 0 shape: {probas[0].shape}, values:\n{probas[0]}")
            else:
                print(f"predict_proba shape: {probas.shape}, values:\n{probas}")
    except Exception as e:
        print(f"Test prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_dataset("data/synthetic_students.csv")
    inspect_model_obj("ml/model.pkl")
    inspect_model_obj(r"C:\Users\anujd\Downloads\model.pkl")
    inspect_model_obj(r"C:\Users\anujd\Downloads\xgboost.pkl")
