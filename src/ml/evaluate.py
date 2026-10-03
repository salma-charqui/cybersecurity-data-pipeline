"""
ML evaluation and investigation for CIC-IDS2017 intrusion detection.

This module investigates the high performance obtained on a random stratified
split by:

1. Reproducing the train/test split WITH original labels preserved
2. Visualizing the confusion matrix and ROC/PR curves
3. Identifying which attack TYPES are missed (False Negatives) using
   labels preserved through the entire pipeline (no row reconstruction)
4. Detecting feature-identical observations between train and test
   (counted as observation count, not unique hash count)
5. Re-evaluating with a temporal split (Mon-Wed train, Thu-Fri test)
6. Comparing attack type distributions between train and test
7. Per-day performance analysis (with day_of_week preserved in same DataFrame)

Conclusions are deliberately measured. No categorical claims about leakage
or concept drift unless directly verified by the data.

Usage on Databricks:
    from src.ml.evaluate import run_full_evaluation
    results = run_full_evaluation(spark)
"""

import logging
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# === Logging ===
logger = logging.getLogger(__name__)


# =============================================================================
# Helper - reproduce prepare_training_data BUT preserve label + day_of_week
# =============================================================================

def prepare_training_data_with_metadata(
    spark: SparkSession,
    sample_size: int = 200000,
    random_state: int = 42,
):
    """
    Reproduce the same train/test split as prepare_training_data(), BUT
    preserve the original 'label' and 'day_of_week' columns throughout
    the entire pipeline so we can later trace FN rows to their attack type.
    
    IMPORTANT: This function is a reimplementation of prepare_training_data()
    with metadata preservation added. It does NOT prove that the saved XGBoost.joblib
    was trained on this exact split — only that the preprocessing logic is identical.
    To prove the latter, run-time assertions compare the actual feature values
    with the original prepare_training_data() output.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names,
        train_metadata (DataFrame with label + day_of_week for train rows),
        test_metadata (DataFrame with label + day_of_week for test rows)
    """
    logger.info("Loading Gold table with metadata preservation...")
    df_gold = spark.read.table("workspace.cybersecurity.gold")
    
    total_rows = df_gold.count()
    sample_fraction = min(1.0, sample_size / total_rows)
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Sampling fraction: {sample_fraction:.4f}")
    
    # Sample with the same logic as prepare_training_data
    df_sample = (
        df_gold
        .sampleBy("is_attack", fractions={0: sample_fraction, 1: sample_fraction}, seed=random_state)
        .toPandas()
    )
    logger.info(f"  Sampled rows: {len(df_sample)}")
    
    # === STEP 1: PRESERVE label and day_of_week for later analysis ===
    metadata_cols = ['label', 'day_of_week']
    available_metadata = [c for c in metadata_cols if c in df_sample.columns]
    
    # Save metadata BEFORE any transformation
    df_metadata = df_sample[available_metadata].copy().reset_index(drop=True)
    df_metadata['is_attack'] = df_sample['is_attack'].values
    
    # === STEP 2: Same transformations as prepare_training_data ===
    leakage_cols = ['label', 'attack_category', 'day_of_week',
                    '_source_file', '_ingestion_ts', 'timestamp']
    available_leakage = [c for c in leakage_cols if c in df_sample.columns]
    
    # One-hot encode protocol_name
    if 'protocol_name' in df_sample.columns:
        protocol_dummies = pd.get_dummies(df_sample['protocol_name'], prefix='protocol')
        df_sample = pd.concat([df_sample.drop(columns=['protocol_name']), protocol_dummies], axis=1)
    
    # Separate features and target
    X = df_sample.drop(columns=available_leakage + ['is_attack'])
    y = df_sample['is_attack']
    
    # Replace inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    
    # === STEP 3: Train/test split (stratified) ===
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # === STEP 4: Get the indices used for the split (to slice metadata) ===
    train_indices = y_train.index
    test_indices = y_test.index
    
    # === STEP 5: Reset indices on X_train, X_test, y_train, y_test ===
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # === STEP 6: Create train_metadata and test_metadata (using original indices) ===
    train_metadata = df_metadata.iloc[train_indices].reset_index(drop=True)
    test_metadata = df_metadata.iloc[test_indices].reset_index(drop=True)
    
    logger.info(f"  Train: {len(X_train)} rows ({y_train.sum()} attacks, {y_train.mean()*100:.2f}%)")
    logger.info(f"  Test:  {len(X_test)} rows ({y_test.sum()} attacks, {y_test.mean()*100:.2f}%)")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Metadata preserved: {available_metadata}")
    
    # === STEP 7: CRITICAL verification - EXACT match with prepare_training_data ===
    # This ensures our preprocessing is a faithful reimplementation.
    # NOTE: This proves preprocessing reproducibility, NOT that the saved XGBoost.joblib
    # was trained on this exact split (which would require saving row_ids during training).
    logger.info("  Verifying preprocessing EXACTLY matches prepare_training_data()...")
    from src.ml.train import prepare_training_data as _original_prepare
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig, feature_names_orig = _original_prepare(
        spark, sample_size=sample_size, random_state=random_state
    )
    
    # === Verify column ORDER (not just set) ===
    assert list(X_train.columns) == list(X_train_orig.columns), (
        f"Column ORDER mismatch! "
        f"New first 5: {list(X_train.columns)[:5]}, "
        f"Original first 5: {list(X_train_orig.columns)[:5]}"
    )
    assert list(X_test.columns) == list(X_test_orig.columns), (
        f"X_test column ORDER mismatch!"
    )
    
    # === Verify shapes ===
    assert X_train.shape == X_train_orig.shape, (
        f"X_train shape mismatch! New: {X_train.shape}, Original: {X_train_orig.shape}"
    )
    assert X_test.shape == X_test_orig.shape, (
        f"X_test shape mismatch! New: {X_test.shape}, Original: {X_test_orig.shape}"
    )
    
    # === CRITICAL: Verify actual feature VALUES (not just shapes/sums) ===
    assert np.array_equal(
        X_train.to_numpy(),
        X_train_orig.to_numpy()
    ), "X_train feature values differ from original preprocessing!"
    
    assert np.array_equal(
        X_test.to_numpy(),
        X_test_orig.to_numpy()
    ), "X_test feature values differ from original preprocessing!"
    
    # === Verify targets exactly ===
    assert np.array_equal(
        y_train.to_numpy(),
        y_train_orig.to_numpy()
    ), "y_train differs from original preprocessing!"
    
    assert np.array_equal(
        y_test.to_numpy(),
        y_test_orig.to_numpy()
    ), "y_test differs from original preprocessing!"
    
    logger.info(f"  ✓ EXACT preprocessing match verified (values, shapes, columns, targets)")
    logger.info(f"    All assertions passed:")
    logger.info(f"      - Column order: ✓")
    logger.info(f"      - Shapes:       ✓")
    logger.info(f"      - Feature values: ✓ (np.array_equal)")
    logger.info(f"      - Target values:  ✓ (np.array_equal)")
    logger.info(f"    Note: This confirms preprocessing reproducibility,")
    logger.info(f"    not that the saved XGBoost.joblib was trained on this exact split.")
    
    feature_names = list(X_train.columns)
    return X_train, X_test, y_train, y_test, feature_names, train_metadata, test_metadata


# =============================================================================
# Phase 1 - Load model and reproduce data WITH metadata
# =============================================================================

def load_model_and_data(spark: SparkSession, random_state: int = 42):
    """Load saved XGBoost model and reproduce the split WITH metadata preserved."""
    import joblib
    
    logger.info("Loading saved XGBoost model...")
    model_path = "/Volumes/workspace/cybersecurity/models/XGBoost.joblib"
    model = joblib.load(model_path)
    logger.info(f"  Model loaded: {type(model).__name__}")
    
    logger.info("Reproducing train/test split WITH metadata preservation...")
    X_train, X_test, y_train, y_test, feature_names, train_metadata, test_metadata = (
        prepare_training_data_with_metadata(spark, random_state=random_state)
    )
    
    return model, X_train, X_test, y_train, y_test, feature_names, train_metadata, test_metadata


# =============================================================================
# Phase 2 - Confusion Matrix
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name: str = "XGBoost"):
    """Visual confusion matrix as a heatmap."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Count')
    
    classes = ['Benign', 'Attack']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f'Predicted {c}' for c in classes], fontsize=10)
    ax.set_yticklabels([f'Actual {c}' for c in classes], fontsize=10)
    
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = 'white' if value > cm.max() / 2 else 'black'
            ax.text(j, i, f'{value:,}\n({value/cm.sum()*100:.2f}%)',
                    ha='center', va='center', color=color, fontsize=12, fontweight='bold')
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    return fig, cm


# =============================================================================
# Phase 3 - ROC Curve
# =============================================================================

def plot_roc_curve(y_test, y_proba, model_name: str = "XGBoost"):
    """Plot ROC curve with AUC."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, roc_auc


# =============================================================================
# Phase 4 - Precision-Recall Curve
# =============================================================================

def plot_precision_recall_curve(y_test, y_proba, model_name: str = "XGBoost"):
    """Plot Precision-Recall curve."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='darkred', lw=2,
            label=f'{model_name} (AP = {avg_precision:.4f})')
    
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='navy', lw=1, linestyle='--',
               label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, avg_precision


# =============================================================================
# Phase 5 - PROPER False Negatives Analysis (with preserved metadata)
# =============================================================================

def analyze_false_negatives(
    y_test: pd.Series,
    y_pred: np.ndarray,
    test_metadata: pd.DataFrame,
) -> Tuple[Dict[str, int], pd.Series]:
    """
    Identify the False Negatives and their original attack types.
    
    Uses test_metadata preserved through the entire pipeline.
    Includes assertions to guarantee alignment between y_test, y_pred, and metadata.
    """
    logger.info("Analyzing False Negatives (with preserved metadata)...")
    
    # === CRITICAL: Assertions to guarantee alignment ===
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    
    assert len(y_test) == len(y_pred_arr), (
        f"Length mismatch: y_test={len(y_test)}, y_pred={len(y_pred_arr)}"
    )
    assert len(test_metadata) == len(y_test), (
        f"Metadata length mismatch: metadata={len(test_metadata)}, y_test={len(y_test)}"
    )
    
    # Check index alignment (if both have meaningful indices)
    try:
        if not test_metadata.index.equals(y_test.index):
            logger.warning("  Metadata and y_test indices differ - using positional alignment (reset_index was applied)")
    except Exception:
        logger.warning("  Could not verify index alignment, proceeding with positional alignment")
    
    logger.info(f"  ✓ Alignment verified: y_test={len(y_test)}, y_pred={len(y_pred_arr)}, metadata={len(test_metadata)}")
    
    # Convert to numpy arrays for safe masking
    y_true_arr = y_test.values
    
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)
    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    tp_mask = (y_true_arr == 1) & (y_pred_arr == 1)
    tn_mask = (y_true_arr == 0) & (y_pred_arr == 0)
    
    fn_count = int(fn_mask.sum())
    fp_count = int(fp_mask.sum())
    tp_count = int(tp_mask.sum())
    tn_count = int(tn_mask.sum())
    
    fn_summary = {
        'true_positives': tp_count,
        'false_negatives': fn_count,
        'true_negatives': tn_count,
        'false_positives': fp_count,
    }
    
    logger.info(f"  True Positives  (correctly detected attacks): {tp_count:,}")
    logger.info(f"  False Negatives (MISSED attacks):            {fn_count:,}")
    logger.info(f"  True Negatives  (correctly identified benign): {tn_count:,}")
    logger.info(f"  False Positives (false alarms):                {fp_count:,}")
    
    # === Get the original attack types for FN rows ===
    # test_metadata has the SAME length and order as y_test (we preserved it)
    # We use positional indexing (iloc) which is safe after reset_index
    if 'label' in test_metadata.columns and fn_count > 0:
        # Use .iloc with boolean mask for positional indexing (safe)
        fn_metadata = test_metadata.iloc[fn_mask].copy()
        fn_attack_types = fn_metadata['label'].value_counts()
        
        logger.info(f"\n  Attack types MISSED (False Negatives):")
        for attack, count in fn_attack_types.items():
            logger.info(f"    {attack}: {count}")
        
        # Also show day_of_week distribution of FN if available
        if 'day_of_week' in fn_metadata.columns:
            fn_days = fn_metadata['day_of_week'].value_counts()
            logger.info(f"\n  Day of week distribution of False Negatives:")
            for day, count in fn_days.items():
                logger.info(f"    {day}: {count}")
    else:
        logger.warning("  'label' column not in test_metadata or no FN to analyze.")
        fn_attack_types = pd.Series()
    
    return fn_summary, fn_attack_types


# =============================================================================
# Phase 6 - Feature-Identical Observations (properly counted)
# =============================================================================

def check_feature_identical_observations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Count the number of TEST observations that have at least one
    feature-identical counterpart in TRAIN.
    
    This is the proper count (not just unique hash intersection).
    
    IMPORTANT: Feature-identical observations are NOT necessarily data leakage.
    Two different network flows can have identical numerical features,
    especially after dropping identifying columns like _source_file, timestamp.
    """
    logger.info("Checking for feature-identical observations (proper count)...")
    
    # Hash each row's feature values explicitly as float32
    def hash_row(row):
        values = np.asarray(row.values, dtype=np.float32)
        return hashlib.md5(values.tobytes()).hexdigest()
    
    logger.info("  Hashing train rows...")
    train_hashes = X_train.apply(hash_row, axis=1)
    train_hash_set = set(train_hashes)
    
    logger.info("  Hashing test rows...")
    test_hashes = X_test.apply(hash_row, axis=1)
    
    # Count how many TEST rows have a counterpart in train
    test_has_counterpart = test_hashes.isin(train_hash_set)
    test_with_counterpart_count = int(test_has_counterpart.sum())
    
    # Also: how many TRAIN rows have a counterpart in test
    train_has_counterpart = train_hashes.isin(set(test_hashes))
    train_with_counterpart_count = int(train_has_counterpart.sum())
    
    # Unique hash overlap (for comparison with old method)
    unique_overlap = len(train_hash_set & set(test_hashes))
    
    result = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_observations_with_train_counterpart': test_with_counterpart_count,
        'pct_of_test_with_counterpart': test_with_counterpart_count / len(X_test) * 100,
        'train_observations_with_test_counterpart': train_with_counterpart_count,
        'pct_of_train_with_counterpart': train_with_counterpart_count / len(X_train) * 100,
        'unique_hash_overlap': unique_overlap,
        'interpretation': (
            f"{test_with_counterpart_count} test observations (out of {len(X_test)}) "
            f"have at least one feature-identical counterpart in train "
            f"({test_with_counterpart_count / len(X_test) * 100:.2f}% of test). "
            f"This does NOT automatically mean data leakage. "
            f"Two different network flows can have identical numerical features "
            f"(especially after dropping _source_file, timestamp, label). "
            f"To confirm leakage, identical identifying columns would need to be verified."
        )
    }
    
    logger.info(f"  Train size: {result['train_size']:,}")
    logger.info(f"  Test size:  {result['test_size']:,}")
    logger.info(f"  Test observations with feature-identical counterpart in train: {test_with_counterpart_count:,}")
    logger.info(f"  Percentage of test: {result['pct_of_test_with_counterpart']:.2f}%")
    logger.info(f"  Train observations with counterpart in test: {train_with_counterpart_count:,}")
    logger.info(f"  Unique hash overlap (old method): {unique_overlap}")
    logger.info(f"\n  Interpretation: {result['interpretation']}")
    
    return result


# =============================================================================
# Phase 7 - Temporal Split with metadata preserved
# =============================================================================

def prepare_temporal_split(spark: SparkSession, random_state: int = 42):
    """
    Temporal split: train on Mon-Wed, test on Thu-Fri.
    Preserves day_of_week and label in the test set for per-day analysis.
    """
    logger.info("Preparing TEMPORAL split (train: Mon-Wed, test: Thu-Fri)...")
    
    df_gold = spark.read.table("workspace.cybersecurity.gold")
    
    df_train = df_gold.filter(F.col("day_of_week").isin(["Monday", "Tuesday", "Wednesday"]))
    df_test = df_gold.filter(F.col("day_of_week").isin(["Thursday", "Friday"]))
    
    train_count = df_train.count()
    test_count = df_test.count()
    
    logger.info(f"  Train (Mon-Wed): {train_count:,} rows")
    logger.info(f"  Test  (Thu-Fri): {test_count:,} rows")
    
    logger.info("  Converting to pandas (preserving day_of_week and label)...")
    df_train_pd = df_train.toPandas()
    df_test_pd = df_test.toPandas()
    
    # PRESERVE day_of_week and label in test_metadata for per-day analysis
    test_metadata_cols = ['day_of_week', 'label']
    available_test_metadata = [c for c in test_metadata_cols if c in df_test_pd.columns]
    test_metadata = df_test_pd[available_test_metadata].copy().reset_index(drop=True)
    
    # Drop leakage columns
    leakage_cols = ['label', 'attack_category', 'day_of_week',
                    '_source_file', '_ingestion_ts', 'timestamp']
    available_leakage_train = [c for c in leakage_cols if c in df_train_pd.columns]
    available_leakage_test = [c for c in leakage_cols if c in df_test_pd.columns]
    
    # One-hot encode protocol_name
    if 'protocol_name' in df_train_pd.columns:
        train_dummies = pd.get_dummies(df_train_pd['protocol_name'], prefix='protocol')
        df_train_pd = pd.concat([df_train_pd.drop(columns=['protocol_name']), train_dummies], axis=1)
    
    if 'protocol_name' in df_test_pd.columns:
        test_dummies = pd.get_dummies(df_test_pd['protocol_name'], prefix='protocol')
        df_test_pd = pd.concat([df_test_pd.drop(columns=['protocol_name']), test_dummies], axis=1)
    
    # Align columns
    train_cols = set(df_train_pd.columns)
    test_cols = set(df_test_pd.columns)
    
    for col in train_cols - test_cols:
        df_test_pd[col] = 0
    for col in test_cols - train_cols:
        df_train_pd[col] = 0
    
    df_test_pd = df_test_pd[df_train_pd.columns]
    
    # Separate features and target
    X_train = df_train_pd.drop(columns=available_leakage_train + ['is_attack'])
    y_train = df_train_pd['is_attack']
    X_test = df_test_pd.drop(columns=available_leakage_test + ['is_attack'])
    y_test = df_test_pd['is_attack']
    
    # Replace inf/NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    
    # Reset indices for clean alignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    logger.info(f"  Train: {len(X_train):,} rows, {y_train.sum():,} attacks ({y_train.mean()*100:.2f}%)")
    logger.info(f"  Test:  {len(X_test):,} rows, {y_test.sum():,} attacks ({y_test.mean()*100:.2f}%)")
    logger.info(f"  Features: {X_train.shape[1]}")
    
    feature_names = list(X_train.columns)
    return X_train, y_train, X_test, y_test, feature_names, test_metadata


def train_xgboost_temporal(X_train, y_train, random_state: int = 42):
    """Train XGBoost on the temporal split."""
    from xgboost import XGBClassifier
    
    logger.info("Training XGBoost on temporal split...")
    
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
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
    logger.info("  XGBoost trained on temporal split")
    return xgb


# =============================================================================
# Phase 8 - Attack Type Distribution Comparison
# =============================================================================

def compare_attack_type_distribution(
    df_train_pd: pd.DataFrame,
    df_test_pd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare which attack types are in train vs test.
    THE KEY analysis: are attacks in test truly 'unseen'?
    """
    logger.info("Comparing attack type distributions between train and test...")
    
    train_labels = df_train_pd['label'].value_counts() if 'label' in df_train_pd.columns else pd.Series()
    test_labels = df_test_pd['label'].value_counts() if 'label' in df_test_pd.columns else pd.Series()
    
    all_labels = sorted(set(train_labels.index) | set(test_labels.index))
    
    rows = []
    for label in all_labels:
        train_count = int(train_labels.get(label, 0))
        test_count = int(test_labels.get(label, 0))
        
        if train_count > 0 and test_count > 0:
            status = "Both"
        elif train_count > 0:
            status = "Train only"
        elif test_count > 0:
            status = "TEST ONLY (unseen in training)"
        else:
            status = "Neither"
        
        rows.append({
            'attack_type': label,
            'train_count': train_count,
            'test_count': test_count,
            'status': status,
        })
    
    df_comparison = pd.DataFrame(rows).sort_values('train_count', ascending=False)
    
    logger.info("\n  Attack type distribution (train vs test):")
    logger.info(f"  {'Attack Type':<40} {'Train':>10} {'Test':>10}  {'Status'}")
    logger.info(f"  {'-'*40} {'-'*10} {'-'*10}  {'-'*30}")
    for _, row in df_comparison.iterrows():
        logger.info(f"  {row['attack_type']:<40} {row['train_count']:>10} {row['test_count']:>10}  {row['status']}")
    
    train_only = df_comparison[df_comparison['status'] == 'Train only']
    test_only = df_comparison[df_comparison['status'] == 'TEST ONLY (unseen in training)']
    both = df_comparison[df_comparison['status'] == 'Both']
    
    logger.info(f"\n  Summary:")
    logger.info(f"    Attack types in BOTH train and test:    {len(both)}")
    logger.info(f"    Attack types in train only:              {len(train_only)}")
    logger.info(f"    Attack types in TEST ONLY (unseen):      {len(test_only)}")
    
    if len(test_only) > 0:
        logger.info(f"\n  Unseen attack types in test:")
        for _, row in test_only.iterrows():
            logger.info(f"    - {row['attack_type']}: {row['test_count']} occurrences in test")
    
    return df_comparison


# =============================================================================
# Phase 9 - Per-day Performance (with preserved metadata)
# =============================================================================

def evaluate_per_day(
    y_test: pd.Series,
    y_pred: np.ndarray,
    test_metadata: pd.DataFrame,
):
    """
    Evaluate model performance per day.
    
    test_metadata is preserved from the SAME df_test_pd as y_test, so
    the alignment is guaranteed (no separate toPandas() calls).
    """
    logger.info("Per-day performance analysis (with preserved day_of_week)...")
    
    if 'day_of_week' not in test_metadata.columns:
        logger.warning("  day_of_week not preserved, cannot do per-day analysis.")
        return None
    
    # Convert y_pred to numpy array (in case it's a list)
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    
    # test_metadata and y_test have the same length and order
    days = test_metadata['day_of_week'].values
    
    if len(days) != len(y_pred_arr):
        logger.error(f"  Length mismatch: days={len(days)}, y_pred={len(y_pred_arr)}")
        return None
    
    results_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred_arr,
        'day_of_week': days,
    })
    
    per_day_metrics = []
    for day in sorted(results_df['day_of_week'].unique()):
        day_data = results_df[results_df['day_of_week'] == day]
        tp = int(((day_data['y_true'] == 1) & (day_data['y_pred'] == 1)).sum())
        fp = int(((day_data['y_true'] == 0) & (day_data['y_pred'] == 1)).sum())
        fn = int(((day_data['y_true'] == 1) & (day_data['y_pred'] == 0)).sum())
        tn = int(((day_data['y_true'] == 0) & (day_data['y_pred'] == 0)).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        per_day_metrics.append({
            'day': day,
            'total_rows': int(len(day_data)),
            'attacks': int(day_data['y_true'].sum()),
            'benign': int((day_data['y_true'] == 0).sum()),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
        })
    
    df_per_day = pd.DataFrame(per_day_metrics).sort_values('day')
    logger.info("\n" + df_per_day.to_string(index=False))
    
    return df_per_day


# =============================================================================
# Main orchestration
# =============================================================================

def run_full_evaluation(spark: SparkSession, random_state: int = 42) -> Dict[str, Any]:
    """Run the complete ML evaluation pipeline with measured conclusions."""
    logger.info("=" * 70)
    logger.info("ML Evaluation and Investigation Pipeline Started")
    logger.info("=" * 70)
    
    results = {}
    
    # === Phase 1 - Load model and reproduce data WITH metadata ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 - Load Model and Reproduce Data (with metadata preserved)")
    logger.info("=" * 70)
    
    model, X_train, X_test, y_train, y_test, feature_names, train_metadata, test_metadata = (
        load_model_and_data(spark, random_state)
    )
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # === Compute REAL stratified metrics (not hardcoded) ===
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    stratified_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
    }
    
    logger.info(f"\n  Computed stratified split metrics:")
    logger.info(f"    Accuracy:  {stratified_metrics['accuracy']:.4f}")
    logger.info(f"    Precision: {stratified_metrics['precision']:.4f}")
    logger.info(f"    Recall:    {stratified_metrics['recall']:.4f}")
    logger.info(f"    F1-score:  {stratified_metrics['f1_score']:.4f}")
    logger.info(f"    ROC AUC:   {stratified_metrics['roc_auc']:.4f}")
    
    results['stratified_metrics'] = stratified_metrics
    # === Phase 2 - Confusion Matrix ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 - Confusion Matrix (stratified split)")
    logger.info("=" * 70)
    
    cm_fig, cm = plot_confusion_matrix(y_test, y_pred, "XGBoost (stratified split)")
    results['confusion_matrix_stratified'] = cm.tolist()
    
    # === Phase 3 - ROC Curve ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 - ROC Curve (stratified split)")
    logger.info("=" * 70)
    
    roc_fig, roc_auc = plot_roc_curve(y_test, y_proba, "XGBoost (stratified split)")
    results['roc_auc_stratified'] = float(roc_auc)
    
    # === Phase 4 - Precision-Recall Curve ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 - Precision-Recall Curve (stratified split)")
    logger.info("=" * 70)
    
    pr_fig, avg_precision = plot_precision_recall_curve(y_test, y_proba, "XGBoost (stratified split)")
    results['avg_precision_stratified'] = float(avg_precision)
    
    # === Phase 5 - PROPER False Negatives Analysis ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5 - False Negatives Analysis (with preserved metadata)")
    logger.info("=" * 70)
    
    fn_summary, fn_attack_types = analyze_false_negatives(
        y_test, y_pred, test_metadata
    )
    results['fn_summary'] = fn_summary
    results['fn_attack_types'] = fn_attack_types.to_dict() if hasattr(fn_attack_types, 'to_dict') else {}
    
    # === Phase 6 - Feature-Identical Observations (proper count) ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6 - Feature-Identical Observations Check (proper count)")
    logger.info("=" * 70)
    
    overlap_result = check_feature_identical_observations(X_train, X_test)
    results['feature_identical_check'] = overlap_result
    
    # === Phase 7 - Temporal Split ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7 - TEMPORAL SPLIT EVALUATION")
    logger.info("=" * 70)
    
    X_train_temp, y_train_temp, X_test_temp, y_test_temp, feature_names_temp, test_metadata_temp = (
        prepare_temporal_split(spark, random_state)
    )
    
    # === Phase 8 - Attack Type Distribution ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8 - Attack Type Distribution Comparison")
    logger.info("=" * 70)
    
    df_gold = spark.read.table("workspace.cybersecurity.gold")
    df_train_orig = df_gold.filter(F.col("day_of_week").isin(["Monday", "Tuesday", "Wednesday"])).toPandas()
    df_test_orig = df_gold.filter(F.col("day_of_week").isin(["Thursday", "Friday"])).toPandas()
    
    attack_distribution = compare_attack_type_distribution(df_train_orig, df_test_orig)
    results['attack_type_distribution'] = attack_distribution.to_dict('records')
    
    # Train XGBoost on temporal split
    model_temp = train_xgboost_temporal(X_train_temp, y_train_temp, random_state)
    
    y_pred_temp = model_temp.predict(X_test_temp)
    y_proba_temp = model_temp.predict_proba(X_test_temp)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    temporal_metrics = {
        'accuracy': float(accuracy_score(y_test_temp, y_pred_temp)),
        'precision': float(precision_score(y_test_temp, y_pred_temp)),
        'recall': float(recall_score(y_test_temp, y_pred_temp)),
        'f1_score': float(f1_score(y_test_temp, y_pred_temp)),
        'roc_auc': float(roc_auc_score(y_test_temp, y_proba_temp)),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("TEMPORAL SPLIT RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Accuracy:  {temporal_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {temporal_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {temporal_metrics['recall']:.4f}")
    logger.info(f"  F1-score:  {temporal_metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {temporal_metrics['roc_auc']:.4f}")
    
    # === Phase 9 - Per-day Performance (with preserved metadata) ===
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 9 - Per-day Performance (with preserved metadata)")
    logger.info("=" * 70)
    
    df_per_day = evaluate_per_day(y_test_temp, y_pred_temp, test_metadata_temp)
    results['per_day_metrics'] = df_per_day.to_dict('records') if df_per_day is not None else None
    
    # Confusion matrix for temporal split
    cm_temp_fig, _ = plot_confusion_matrix(y_test_temp, y_pred_temp, "XGBoost (temporal split)")
    
    # === Measured Summary ===
    logger.info("\n" + "=" * 70)
    logger.info("MEASURED SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Stratified split F1: {stratified_metrics['f1_score']:.4f}, "
                f"AUC: {stratified_metrics['roc_auc']:.4f}")
    logger.info(f"Temporal split F1:    {temporal_metrics['f1_score']:.4f}, "
                f"AUC: {temporal_metrics['roc_auc']:.4f}")
    logger.info(f"Test observations with feature-identical counterpart in train: "
                f"{overlap_result['test_observations_with_train_counterpart']} "
                f"({overlap_result['pct_of_test_with_counterpart']:.2f}% of test)")
    
    f1_diff = stratified_metrics['f1_score'] - temporal_metrics['f1_score']
    auc_diff = stratified_metrics['roc_auc'] - temporal_metrics['roc_auc']
    
    logger.info(f"\n  F1 difference (stratified - temporal): {f1_diff:.4f}")
    logger.info(f"  AUC difference (stratified - temporal): {auc_diff:.4f}")
    
    # === Neutral interpretation (no hardcoded "substantial") ===
    logger.info("\n  Observed difference between the two evaluation protocols.")
    logger.info("  Possible contributing factors (not mutually exclusive):")
    logger.info("    - Different attack types in train vs test (see Phase 8)")
    logger.info("    - Day-specific traffic patterns")
    logger.info("    - Feature-identical observations in both splits")
    logger.info("    - Other dataset-specific characteristics")
    logger.info("  Further investigation would be needed to isolate the contribution of each factor.")
    
    logger.info("\n" + "=" * 70)
    logger.info("ML Evaluation Pipeline Completed")
    logger.info("=" * 70)
    
    return {
        'stratified_metrics': stratified_metrics,
        'temporal_metrics': temporal_metrics,
        'feature_identical_check': overlap_result,
        'fn_summary': fn_summary,
        'fn_attack_types': results['fn_attack_types'],
        'attack_type_distribution': results['attack_type_distribution'],
        'per_day_metrics': results['per_day_metrics'],
        'figures': {
            'cm_stratified': cm_fig,
            'roc_curve': roc_fig,
            'pr_curve': pr_fig,
            'cm_temporal': cm_temp_fig,
        }
    }