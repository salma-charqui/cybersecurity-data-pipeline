"""
Machine Learning training pipeline for CIC-IDS2017 intrusion detection.

Trains binary classifiers (Random Forest + XGBoost) on the Gold layer
to detect attacks vs benign network traffic.

Handles class imbalance with:
- class_weight='balanced' for Random Forest
- scale_pos_weight for XGBoost

Usage on Databricks:
    from src.ml.train import train_models
    results = train_models(spark)
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# === Logging ===
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1 - Data preparation
# =============================================================================

def prepare_training_data(
    spark: SparkSession,
    sample_size: int = 200000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load Gold table, sample, prepare features and target for ML training.
    
    Args:
        spark: SparkSession
        sample_size: target number of rows to sample (default 200K)
        random_state: random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info("Loading Gold table from Delta...")
    df_gold = spark.read.table("workspace.cybersecurity.gold")
    
    total_rows = df_gold.count()
    logger.info(f"  Total rows in Gold: {total_rows}")
    
    # Stratified sampling to preserve class distribution
    # We use sampleBy with the same fraction for each class
    sample_fraction = min(1.0, sample_size / total_rows)
    logger.info(f"  Sampling fraction: {sample_fraction:.4f} (target: {sample_size} rows)")
    
    df_sample = (
        df_gold
        .sampleBy("is_attack", fractions={0: sample_fraction, 1: sample_fraction}, seed=random_state)
        .toPandas()  # Collect to driver for sklearn
    )
    logger.info(f"  Sampled rows: {len(df_sample)}")
    
    # === Drop columns that would cause target leakage ===
    # These columns directly or indirectly reveal the label
    leakage_cols = [
        'label',           # multi-class label (target leakage!)
        'attack_category', # derived from label (target leakage!)
        'day_of_week',     # Monday = 0% attacks (would cheat in production)
        '_source_file',    # metadata (cheat: file name reveals the day)
        '_ingestion_ts',   # metadata (timestamp reveals the day)
        'timestamp',       # if exists, reveals when the flow happened
    ]
    available_leakage = [c for c in leakage_cols if c in df_sample.columns]
    logger.info(f"  Dropping leakage columns: {available_leakage}")
    
    # === One-hot encode protocol_name (categorical) ===
    if 'protocol_name' in df_sample.columns:
        protocol_dummies = pd.get_dummies(df_sample['protocol_name'], prefix='protocol')
        df_sample = pd.concat([df_sample.drop(columns=['protocol_name']), protocol_dummies], axis=1)
        logger.info(f"  One-hot encoded protocol_name -> {len(protocol_dummies.columns)} new columns")
    
    # === Drop leakage columns and separate target ===
    X = df_sample.drop(columns=available_leakage + ['is_attack'])
    y = df_sample['is_attack']
    
    # === Handle infinite and NaN values ===
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # === Convert all columns to numeric ===
    X = X.astype(np.float32)
    
    logger.info(f"  Final features: {X.shape[1]} columns")
    logger.info(f"  Target distribution: {y.sum()} attacks ({y.mean()*100:.2f}%)")
    
    # === Train/test split (stratified to preserve class distribution) ===
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    logger.info(f"  Train: {len(X_train)} rows ({y_train.sum()} attacks, {y_train.mean()*100:.2f}% attack rate)")
    logger.info(f"  Test:  {len(X_test)} rows ({y_test.sum()} attacks, {y_test.mean()*100:.2f}% attack rate)")
    
    feature_names = list(X_train.columns)
    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# Step 2 - Random Forest training
# =============================================================================

def train_random_forest(X_train, y_train, random_state: int = 42):
    """
    Train a Random Forest classifier with class weighting for imbalance.
    
    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info("Training Random Forest...")
    logger.info(f"  n_estimators=100, max_depth=15, class_weight='balanced'")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,  # use all CPU cores
        verbose=0
    )
    rf.fit(X_train, y_train)
    logger.info("  Random Forest trained successfully")
    return rf


# =============================================================================
# Step 3 - XGBoost training
# =============================================================================

def train_xgboost(X_train, y_train, random_state: int = 42):
    """
    Train an XGBoost classifier with scale_pos_weight for class imbalance.
    
    scale_pos_weight = (negative samples) / (positive samples)
    This makes the model pay more attention to the minority class.
    """
    from xgboost import XGBClassifier
    
    logger.info("Training XGBoost...")
    
    # Calculate scale_pos_weight for class imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    logger.info(f"  n_estimators=200, max_depth=6, learning_rate=0.1")
    logger.info(f"  scale_pos_weight={scale_pos_weight:.4f} (neg={n_neg}, pos={n_pos})")
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0,
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    logger.info("  XGBoost trained successfully")
    return xgb


# =============================================================================
# Step 4 - Evaluation
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a trained model on the test set.
    Returns a dict with accuracy, precision, recall, F1, AUC, confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    y_pred = model.predict(X_test)
    
    # predict_proba for AUC (XGBoost and RF both support it)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }
    
    logger.info(f"  {model_name} metrics:")
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1-score:  {metrics['f1_score']:.4f}")
    logger.info(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"    Confusion matrix: {metrics['confusion_matrix']}")
    
    return metrics


# =============================================================================
# Step 5 - Feature importance
# =============================================================================

def get_feature_importance(model, feature_names: List[str], model_name: str) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    Works for both Random Forest and XGBoost.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning(f"  {model_name} doesn't support feature_importances_")
        return pd.DataFrame()
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"  {model_name} - Top 10 features:")
    for i, row in df_importance.head(10).iterrows():
        logger.info(f"    {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    return df_importance


# =============================================================================
# Step 6 - Save model to Unity Catalog Volume
# =============================================================================

def save_model_to_volume(model, model_name: str, metrics: Dict, feature_names: List[str]):
    """
    Save the trained model and its metadata to a Unity Catalog Volume.
    
    The model is saved with joblib (pickle), and metrics are saved as JSON.
    """
    import joblib
    import os
    
    # Create a volume for models if not exists
    spark = SparkSession.builder.getOrCreate()
    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.cybersecurity.models")
    
    model_path = f"/Volumes/workspace/cybersecurity/models/{model_name}.joblib"
    metrics_path = f"/Volumes/workspace/cybersecurity/models/{model_name}_metrics.json"
    features_path = f"/Volumes/workspace/cybersecurity/models/{model_name}_features.json"
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"  Model saved: {model_path}")
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved: {metrics_path}")
    
    # Save feature names
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"  Features saved: {features_path}")
    
    return model_path


# =============================================================================
# Main orchestration function
# =============================================================================

def train_models(
    spark: SparkSession,
    sample_size: int = 200000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete ML training pipeline:
    1. Prepare data (load Gold, sample, split)
    2. Train Random Forest
    3. Train XGBoost
    4. Evaluate both models
    5. Compute feature importance
    6. Save best model to Volume
    
    Returns a dict with models, metrics, and feature importances.
    """
    logger.info("=" * 60)
    logger.info("ML Training Pipeline Started")
    logger.info("=" * 60)
    
    # === Step 1 - Data preparation ===
    logger.info("\n--- Step 1: Data Preparation ---")
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(
        spark, sample_size=sample_size, random_state=random_state
    )
    
    # === Step 2 - Train Random Forest ===
    logger.info("\n--- Step 2: Random Forest Training ---")
    rf_model = train_random_forest(X_train, y_train, random_state)
    
    # === Step 3 - Train XGBoost ===
    logger.info("\n--- Step 3: XGBoost Training ---")
    xgb_model = train_xgboost(X_train, y_train, random_state)
    
    # === Step 4 - Evaluate models ===
    logger.info("\n--- Step 4: Model Evaluation ---")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # === Step 5 - Feature importance ===
    logger.info("\n--- Step 5: Feature Importance ---")
    rf_importance = get_feature_importance(rf_model, feature_names, "Random Forest")
    xgb_importance = get_feature_importance(xgb_model, feature_names, "XGBoost")
    
    # === Step 6 - Compare models and save best ===
    logger.info("\n--- Step 6: Models Comparison ---")
    logger.info(f"Random Forest: F1={rf_metrics['f1_score']:.4f}, AUC={rf_metrics['roc_auc']:.4f}")
    logger.info(f"XGBoost:       F1={xgb_metrics['f1_score']:.4f}, AUC={xgb_metrics['roc_auc']:.4f}")
    
    # Compare on F1-score (good metric for class imbalance)
    if xgb_metrics['f1_score'] > rf_metrics['f1_score']:
        best_model = xgb_model
        best_name = "XGBoost"
        best_metrics = xgb_metrics
        best_importance = xgb_importance
        logger.info(f"  -> Winner: XGBoost (F1={xgb_metrics['f1_score']:.4f})")
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_metrics
        best_importance = rf_importance
        logger.info(f"  -> Winner: Random Forest (F1={rf_metrics['f1_score']:.4f})")
    
    # Save best model to Volume
    logger.info(f"\n--- Saving best model ({best_name}) to Volume ---")
    model_path = save_model_to_volume(best_model, best_name, best_metrics, feature_names)
    
    logger.info("\n" + "=" * 60)
    logger.info("ML Training Pipeline Completed")
    logger.info("=" * 60)
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'rf_importance': rf_importance,
        'xgb_importance': xgb_importance,
        'best_model': best_model,
        'best_model_name': best_name,
        'best_metrics': best_metrics,
        'best_importance': best_importance,
        'feature_names': feature_names,
        'model_path': model_path,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_distribution': y_train.value_counts().to_dict(),
        'y_test_distribution': y_test.value_counts().to_dict(),
    }


# === For local CLI usage ===
if __name__ == "__main__":
    import argparse
    from src.ingestion import create_spark_session
    
    parser = argparse.ArgumentParser(description="ML Training for Intrusion Detection")
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    spark = create_spark_session()
    try:
        results = train_models(spark, sample_size=args.sample_size, random_state=args.random_state)
        print(f"\nBest model: {results['best_model_name']}")
        print(f"Best F1: {results['best_metrics']['f1_score']:.4f}")
    finally:
        spark.stop()