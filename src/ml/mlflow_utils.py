"""
MLflow utilities for tracking ML experiments on the CIC-IDS2017 dataset.

Following the critical evaluation performed in J10 (which revealed a strong
discrepancy between stratified random split and temporal split), this module
tracks BOTH evaluation protocols for each model configuration:

- stratified_* metrics (random split — reproducibility)
- temporal_* metrics (Mon-Wed train, Thu-Fri test — true generalization)

The "best model" is selected based on temporal F1 (more informative for
generalization than stratified F1, as demonstrated by J10).

Usage on Databricks:
    from src.ml.mlflow_utils import train_multiple_configs, compare_runs
    results = train_multiple_configs(spark)
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# === Logging ===
logger = logging.getLogger(__name__)


# =============================================================================
# MLflow setup
# =============================================================================

def setup_mlflow(experiment_name: str = None):
    """
    Configure MLflow tracking and create the experiment if not exists.
    
    On Databricks, experiment names must be absolute paths within the workspace,
    e.g. '/Users/<email>/my-experiment'.
    """
    import mlflow
    
    if experiment_name is None:
        experiment_name = "/Users/salmacharki721@gmail.com/mlflow-experiments/cic-ids2017-intrusion-detection"
    
    mlflow.set_experiment(experiment_name)
    
    tracking_uri = mlflow.get_tracking_uri()
    exp = mlflow.get_experiment_by_name(experiment_name)
    
    logger.info(f"✓ MLflow configured")
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Tracking URI: {tracking_uri}")
    logger.info(f"  Experiment ID: {exp.experiment_id}")
    try:
        logger.info(f"  Location: {exp.location}")
    except AttributeError:
        logger.info(f"  Location: (not available on this deployment)")
    
    return exp


# =============================================================================
# Train a single model with MLflow tracking (BOTH stratified AND temporal eval)
# =============================================================================
def train_with_mlflow(
    model_class: str,
    params: Dict[str, Any],
    X_train_strat,
    y_train_strat,
    X_test_strat,
    y_test_strat,
    X_train_temp,
    y_train_temp,
    X_test_temp,
    y_test_temp,
    feature_names: List[str],
    run_name: str,
) -> Dict[str, Any]:
    """
    Train TWO models per configuration:
    1. Stratified model: trained on stratified train, evaluated on stratified test
    2. Temporal model: trained on temporal train (Mon-Wed), evaluated on temporal test (Thu-Fri)
    
    This avoids the J10 evaluation bias where a model trained on stratified data
    (which includes Thu-Fri) would appear to generalize well on temporal test,
    when in fact it has already seen that data.
    
    The temporal model is the one logged as artifact (production-relevant).
    """
    import mlflow
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )
    
    logger.info(f"\n--- Training {run_name} ({model_class}) ---")
    logger.info(f"  Params: {params}")
    
    with mlflow.start_run(run_name=run_name) as run:
        # === Log parameters ===
        mlflow.log_param("model_type", model_class)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("stratified_train_size", len(X_train_strat))
        mlflow.log_param("stratified_test_size", len(X_test_strat))
        mlflow.log_param("temporal_train_size", len(X_train_temp))
        mlflow.log_param("temporal_test_size", len(X_test_temp))
        mlflow.log_param("n_features", len(feature_names))
        
        # === Build model factory (returns a fresh model each time) ===
        def build_model():
            if model_class == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 15),
                    random_state=params.get("random_state", 42),
                    class_weight="balanced",
                    n_jobs=-1,
                )
            elif model_class == "xgboost":
                from xgboost import XGBClassifier
                # Calculate scale_pos_weight based on STRATIFIED train
                # (same for both models since the class ratio is similar)
                n_neg = int((y_train_strat == 0).sum())
                n_pos = int((y_train_strat == 1).sum())
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
                return XGBClassifier(
                    n_estimators=params.get("n_estimators", 200),
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.1),
                    random_state=params.get("random_state", 42),
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    eval_metric="logloss",
                    verbosity=0,
                    use_label_encoder=False,
                )
            else:
                raise ValueError(f"Unknown model_class: {model_class}")
        
        # === MODEL 1: Stratified model ===
        logger.info("  Training stratified model...")
        model_strat = build_model()
        model_strat.fit(X_train_strat, y_train_strat)
        
        y_pred_strat = model_strat.predict(X_test_strat)
        if hasattr(model_strat, "predict_proba"):
            y_proba_strat = model_strat.predict_proba(X_test_strat)[:, 1]
        else:
            y_proba_strat = y_pred_strat
        
        stratified_metrics = {
            "stratified_accuracy": float(accuracy_score(y_test_strat, y_pred_strat)),
            "stratified_precision": float(precision_score(y_test_strat, y_pred_strat)),
            "stratified_recall": float(recall_score(y_test_strat, y_pred_strat)),
            "stratified_f1_score": float(f1_score(y_test_strat, y_pred_strat)),
            "stratified_roc_auc": float(roc_auc_score(y_test_strat, y_proba_strat)),
        }
        
        logger.info(f"    Stratified F1: {stratified_metrics['stratified_f1_score']:.4f}")
        
        # === MODEL 2: Temporal model (trained on Mon-Wed ONLY) ===
        logger.info("  Training temporal model (Mon-Wed only)...")
        model_temp = build_model()
        model_temp.fit(X_train_temp, y_train_temp)
        
        y_pred_temp = model_temp.predict(X_test_temp)
        if hasattr(model_temp, "predict_proba"):
            y_proba_temp = model_temp.predict_proba(X_test_temp)[:, 1]
        else:
            y_proba_temp = y_pred_temp
        
        temporal_metrics = {
            "temporal_accuracy": float(accuracy_score(y_test_temp, y_pred_temp)),
            "temporal_precision": float(precision_score(y_test_temp, y_pred_temp)),
            "temporal_recall": float(recall_score(y_test_temp, y_pred_temp)),
            "temporal_f1_score": float(f1_score(y_test_temp, y_pred_temp)),
            "temporal_roc_auc": float(roc_auc_score(y_test_temp, y_proba_temp)),
        }
        
        logger.info(f"    Temporal F1: {temporal_metrics['temporal_f1_score']:.4f}")
        
        # === Log all metrics ===
        all_metrics = {**stratified_metrics, **temporal_metrics}
        mlflow.log_metrics(all_metrics)
        
        # === Log the TEMPORAL model as artifact (production-relevant) ===
        # The temporal model is the one that generalizes to unseen attacks
        input_example = X_test_temp.iloc[:5].astype('float64').copy()
        
        if model_class == "random_forest":
            mlflow.sklearn.log_model(
                model_temp,
                "model",
                input_example=input_example,
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        else:
            mlflow.xgboost.log_model(
                model_temp,
                "model",
                input_example=input_example,
            )
        
        # === Log feature names ===
        temp_features_path = "/tmp/feature_names.json"
        with open(temp_features_path, "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact(temp_features_path, artifact_path="features")
        
        # === Tags ===
        mlflow.set_tag("model_type", model_class)
        mlflow.set_tag("evaluation_protocols", "stratified,temporal")
        mlflow.set_tag("training_split", "stratified AND temporal (2 models per config)")
        mlflow.set_tag("status", "trained")
        
        run_id = run.info.run_id
        logger.info(f"  ✓ Run logged: {run_id}")
        
        return {
            "run_id": run_id,
            "run_name": run_name,
            "model_type": model_class,
            "params": params,
            "stratified_metrics": stratified_metrics,
            "temporal_metrics": temporal_metrics,
        }


# =============================================================================
# Train multiple configurations
# =============================================================================

def train_multiple_configs(
    spark,
    sample_size: int = 200000,
) -> Dict[str, Any]:
    """
    Train multiple RF and XGBoost configurations with MLflow tracking,
    evaluating each on BOTH stratified random and temporal splits.
    """
    from src.ml.train import prepare_training_data
    from src.ml.evaluate import prepare_temporal_split
    
    logger.info("=" * 60)
    logger.info("MLflow Multi-Configuration Training Pipeline")
    logger.info("(with dual evaluation: stratified + temporal)")
    logger.info("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # === Prepare BOTH splits ===
    logger.info("\n--- Preparing stratified split ---")
    X_train, X_test_strat, y_train, y_test_strat, feature_names = prepare_training_data(
        spark, sample_size=sample_size, random_state=42
    )
    logger.info(f"  Stratified train: {len(X_train)}, test: {len(X_test_strat)}")
    
    logger.info("\n--- Preparing temporal split ---")
    X_train_temp, y_train_temp, X_test_temp, y_test_temp, feature_names_temp, test_metadata_temp = (
        prepare_temporal_split(spark, random_state=42)
    )
    logger.info(f"  Temporal train: {len(X_train_temp)}, test: {len(X_test_temp)}")
    
    # Align columns (in case protocol_name one-hot encoding differs)
    train_cols = set(X_train.columns)
    temp_cols = set(X_test_temp.columns)
    
    # Add missing columns to temporal test
    for col in train_cols - temp_cols:
        X_test_temp[col] = 0
    # Add missing columns to train
    for col in temp_cols - train_cols:
        X_train[col] = 0
    
    # Reorder columns to match
    X_test_temp = X_test_temp[X_train.columns]
    
    # Define configurations to test
    rf_configs = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 20},
    ]
    
    xgb_configs = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05},
    ]
    
    all_results = []
    
    # Train RF configs
    logger.info("\n--- Training Random Forest configurations ---")
    for i, params in enumerate(rf_configs, 1):
        run_name = f"rf_config_{i}"
        result = train_with_mlflow(
            model_class="random_forest",
            params=params,
            X_train_strat=X_train,
            y_train_strat=y_train,
            X_test_strat=X_test_strat,
            y_test_strat=y_test_strat,
            X_train_temp=X_train_temp,
            y_train_temp=y_train_temp,
            X_test_temp=X_test_temp,
            y_test_temp=y_test_temp,
            feature_names=feature_names,
            run_name=run_name,
        )
        all_results.append(result)
    
    # Train XGBoost configs
    logger.info("\n--- Training XGBoost configurations ---")
    for i, params in enumerate(xgb_configs, 1):
        run_name = f"xgb_config_{i}"
        result = train_with_mlflow(
            model_class="xgboost",
            params=params,
            X_train_strat=X_train,
            y_train_strat=y_train,
            X_test_strat=X_test_strat,
            y_test_strat=y_test_strat,
            X_train_temp=X_train_temp,
            y_train_temp=y_train_temp,
            X_test_temp=X_test_temp,
            y_test_temp=y_test_temp,
            feature_names=feature_names,
            run_name=run_name,
        )
        all_results.append(result)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training pipeline completed: {len(all_results)} runs tracked")
    logger.info(f"Each run has BOTH stratified and temporal metrics.")
    logger.info(f"{'=' * 60}")
    
    return {
        "runs": all_results,
        "feature_names": feature_names,
        "train_size": len(X_train),
        "stratified_test_size": len(X_test_strat),
        "temporal_test_size": len(X_test_temp),
    }


# =============================================================================
# Compare all runs (sorted by TEMPORAL F1, not stratified)
# =============================================================================
def compare_runs() -> pd.DataFrame:
    """
    Load all runs from the current experiment and return them as a DataFrame.
    
    Filters to ONLY include runs with the 'training_split' tag (the new dual-eval
    runs). Old runs without this tag are excluded to avoid mixing buggy temporal
    metrics with correct ones.
    
    Runs are sorted by temporal_f1_score DESCENDING (more informative for
    generalization than stratified_f1_score, as demonstrated by J10).
    """
    import mlflow
    
    experiment_name = "/Users/salmacharki721@gmail.com/mlflow-experiments/cic-ids2017-intrusion-detection"
    exp = mlflow.get_experiment_by_name(experiment_name)
    
    if exp is None:
        logger.warning(f"Experiment '{experiment_name}' not found.")
        return pd.DataFrame()
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    
    if runs.empty:
        logger.warning("No runs found in the experiment.")
        return pd.DataFrame()
    
    # === CRITICAL: Filter to only runs with the 'training_split' tag ===
    # This excludes old runs that had buggy temporal evaluation
    if 'tags.training_split' in runs.columns:
        runs = runs[runs['tags.training_split'].notna()].copy()
        logger.info(f"  Filtered to {len(runs)} runs with 'training_split' tag (dual-eval runs only)")
    else:
        logger.warning("  No 'training_split' tag found in any run - showing all runs")
    
    # Select columns for both stratified and temporal metrics
    columns_to_keep = [
        "run_id", "tags.mlflow.runName",
        "params.model_type", "params.n_estimators", "params.max_depth",
        "params.learning_rate",
        # Stratified metrics
        "metrics.stratified_accuracy", "metrics.stratified_precision",
        "metrics.stratified_recall", "metrics.stratified_f1_score",
        "metrics.stratified_roc_auc",
        # Temporal metrics
        "metrics.temporal_accuracy", "metrics.temporal_precision",
        "metrics.temporal_recall", "metrics.temporal_f1_score",
        "metrics.temporal_roc_auc",
    ]
    
    available_cols = [c for c in columns_to_keep if c in runs.columns]
    df = runs[available_cols].copy()
    
    # Rename columns for clarity
    rename_map = {
        "tags.mlflow.runName": "run_name",
        "params.model_type": "model",
        "params.n_estimators": "n_estimators",
        "params.max_depth": "max_depth",
        "params.learning_rate": "learning_rate",
        "metrics.stratified_accuracy": "strat_accuracy",
        "metrics.stratified_precision": "strat_precision",
        "metrics.stratified_recall": "strat_recall",
        "metrics.stratified_f1_score": "strat_f1",
        "metrics.stratified_roc_auc": "strat_auc",
        "metrics.temporal_accuracy": "temp_accuracy",
        "metrics.temporal_precision": "temp_precision",
        "metrics.temporal_recall": "temp_recall",
        "metrics.temporal_f1_score": "temp_f1",
        "metrics.temporal_roc_auc": "temp_auc",
    }
    df = df.rename(columns=rename_map)
    
    # Sort by TEMPORAL F1 descending
    if "temp_f1" in df.columns:
        df = df.sort_values("temp_f1", ascending=False, na_position="last").reset_index(drop=True)
    
    logger.info(f"✓ Found {len(df)} dual-eval runs (sorted by temporal F1)")
    return df


# =============================================================================
# Register the best model (selected by TEMPORAL F1)
# =============================================================================

def register_best_model(
    model_name: str = "cic-ids2017-intrusion-detection-best",
) -> Dict[str, Any]:
    """
    Find the run with the best TEMPORAL F1-score (not stratified) and
    register its model in the MLflow Model Registry.
    
    Selecting on temporal F1 follows the J10 finding that stratified split
    alone is insufficient to characterize model generalization.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    df_runs = compare_runs()
    if df_runs.empty:
        raise ValueError("No runs found to register.")
    
    # Filter out runs without temporal F1 (e.g., test runs)
    df_with_temporal = df_runs[df_runs["temp_f1"].notna()].copy()
    
    if df_with_temporal.empty:
        raise ValueError("No runs with temporal metrics found. Make sure to re-run with the updated train_with_mlflow.")
    
    # Get the best run (highest temporal F1)
    best_row = df_with_temporal.iloc[0]
    best_run_id = best_row["run_id"]
    best_temp_f1 = best_row["temp_f1"]
    best_strat_f1 = best_row.get("strat_f1", None)
    best_model_type = best_row["model"]
    
    logger.info(f"✓ Best run (selected by TEMPORAL F1): {best_row['run_name']}")
    logger.info(f"  Run ID: {best_run_id}")
    logger.info(f"  Model type: {best_model_type}")
    logger.info(f"  Temporal F1: {best_temp_f1:.4f}")
    if best_strat_f1:
        logger.info(f"  Stratified F1: {best_strat_f1:.4f}")
    logger.info(f"  (Note: selected on temporal F1, not stratified — see J10)")
    
    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        logger.info(f"✓ Model registered as '{model_name}' version {result.version}")
        
        return {
            "model_name": model_name,
            "version": result.version,
            "run_id": best_run_id,
            "model_type": best_model_type,
            "temporal_f1_score": float(best_temp_f1),
            "stratified_f1_score": float(best_strat_f1) if best_strat_f1 else None,
            "selection_criterion": "temporal_f1 (per J10 finding)",
            "model_uri": model_uri,
        }
    except Exception as e:
        logger.warning(f"Could not register model: {e}")
        return {
            "model_name": model_name,
            "run_id": best_run_id,
            "model_type": best_model_type,
            "temporal_f1_score": float(best_temp_f1),
            "error": str(e),
        }


# === For CLI usage ===
if __name__ == "__main__":
    import argparse
    from src.ingestion import create_spark_session
    
    parser = argparse.ArgumentParser(description="MLflow tracking pipeline")
    parser.add_argument("--sample-size", type=int, default=200000)
    args = parser.parse_args()
    
    spark = create_spark_session()
    try:
        results = train_multiple_configs(spark, sample_size=args.sample_size)
        print(f"\nTrained {len(results['runs'])} models.")
        
        df = compare_runs()
        print("\n--- Runs comparison (sorted by temporal F1) ---")
        print(df.to_string(index=False))
        
        best_info = register_best_model()
        print(f"\n--- Best model registered ---")
        print(best_info)
    finally:
        spark.stop()