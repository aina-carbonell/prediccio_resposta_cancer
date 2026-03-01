"""
models.py
=========
Funcions d'entrenament, avaluació i comparació de models ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV,
    learning_curve
)
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ── Configuració de models ───────────────────────────────────────────────────

def get_model_configs(class_ratio: float = 2.0) -> Dict:
    """
    Retorna configuració de models amb hiperparàmetres per defecte.
    
    Parameters
    ----------
    class_ratio : ratio n_neg / n_pos (per xgboost scale_pos_weight)
    """
    return {
        'Logistic Regression': LogisticRegression(
            C=0.1,
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_ratio,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }


def get_hyperparameter_grids() -> Dict:
    """Grids d'hiperparàmetres per GridSearchCV."""
    return {
        'Logistic Regression': {
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5, None],
            'clf__min_samples_leaf': [2, 4, 8]
        },
        'XGBoost': {
            'clf__max_depth': [3, 4, 5],
            'clf__learning_rate': [0.05, 0.1],
            'clf__n_estimators': [100, 200],
            'clf__subsample': [0.8, 1.0]
        }
    }


# ── Avaluació ────────────────────────────────────────────────────────────────

SCORING_DICT = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'average_precision': 'average_precision'
}


def evaluate_model_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Avaluació per cross-validation estratificada.
    
    Returns
    -------
    dict amb mètriques (mean ± std)
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    t0 = time.time()
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=SCORING_DICT,
        return_train_score=True,
        n_jobs=-1
    )
    
    elapsed = time.time() - t0
    
    summary = {'training_time': elapsed}
    for metric in SCORING_DICT.keys():
        test_key = f'test_{metric}'
        train_key = f'train_{metric}'
        if test_key in cv_results:
            summary[f'{metric}_mean'] = cv_results[test_key].mean()
            summary[f'{metric}_std'] = cv_results[test_key].std()
            summary[f'{metric}_train'] = cv_results[train_key].mean() if train_key in cv_results else None
    
    if verbose:
        print(f"  AUC: {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")
        print(f"  F1:  {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
        print(f"  Time: {elapsed:.1f}s")
    
    return summary


def compare_models(
    pipelines: Dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5
) -> pd.DataFrame:
    """
    Compara múltiples models i retorna DataFrame de resultats.
    """
    results = []
    print("Comparant models...")
    print("=" * 60)
    
    for name, pipeline in pipelines.items():
        print(f"\n▶ {name}")
        summary = evaluate_model_cv(pipeline, X, y, cv_splits)
        
        row = {'Model': name}
        for metric in ['roc_auc', 'f1', 'accuracy', 'precision', 'recall']:
            mean = summary.get(f'{metric}_mean', np.nan)
            std = summary.get(f'{metric}_std', np.nan)
            row[metric.upper()] = f'{mean:.3f} ± {std:.3f}'
        row['Time (s)'] = f'{summary["training_time"]:.1f}'
        results.append(row)
    
    return pd.DataFrame(results).set_index('Model')


# ── Visualitzacions ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', ax=None):
    """Confusion matrix normalitzada i absoluta."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Resp.', 'Responder'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title, fontweight='bold')
    return ax


def plot_learning_curves(model, X, y, title='Learning Curve', ax=None):
    """Corbes d'aprenentatge per detectar overfitting/underfitting."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1
    )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(train_sizes, train_scores.mean(1), 'b-o', label='Train AUC')
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.2, color='b')
    
    ax.plot(train_sizes, val_scores.mean(1), 'r-o', label='Validation AUC')
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color='r')
    
    ax.set_xlabel('Grandària del Training Set')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    
    return ax


def save_model(model, feature_names, target, output_path, metrics=None):
    """Guarda el model i metadades."""
    package = {
        'model': model,
        'feature_names': feature_names,
        'target': target,
        'metrics': metrics or {}
    }
    joblib.dump(package, output_path)
    print(f"✅ Model guardat: {output_path}")
    return package


def load_model(model_path):
    """Carrega un model guardat."""
    package = joblib.load(model_path)
    return package['model'], package.get('feature_names', [])