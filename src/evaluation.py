"""
evaluation.py
Funcions d'avaluació de models per a predicció de resposta a immunoteràpia.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calcula mètriques rellevants per a context clínic.
    
    Inclou: accuracy, AUC, F1, sensitivitat, especificitat, PPV, NPV.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),          # TPR
        'specificity': tn / (tn + fp),                         # TNR
        'ppv': precision_score(y_true, y_pred, zero_division=0), # Positive Predictive Value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,        # Negative Predictive Value
        'f1': f1_score(y_true, y_pred),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = '', ax=None,
                           class_names: List[str] = ['No-Resposta', 'Resposta']) -> plt.Axes:
    """Visualitza la matriu de confusió amb anotacions clíniques."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Percentatges
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Heatmap
    sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor='white',
        vmin=0
    )
    
    # Anotacions manuals amb percentatge
    for i in range(2):
        for j in range(2):
            label = {(0,0): 'TN', (0,1): 'FP', (1,0): 'FN', (1,1): 'TP'}[(i,j)]
            ax.text(j + 0.5, i + 0.4, str(cm[i,j]),
                    ha='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
            ax.text(j + 0.5, i + 0.65, f'{cm_pct[i,j]:.1f}%',
                    ha='center', fontsize=10, color='gray')
            ax.text(j + 0.5, i + 0.85, label,
                    ha='center', fontsize=10, color='gray', fontstyle='italic')
    
    ax.set_xlabel('Predicció', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.set_title(f'Matriu de Confusió — {model_name}', fontweight='bold')
    
    return ax


def plot_roc_comparison(models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         ax=None, title: str = 'Corbes ROC') -> plt.Axes:
    """
    Compara corbes ROC de múltiples models.
    
    Args:
        models_data: {nom_model: (y_true, y_proba)}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models_data)))
    
    for (name, (y_true, y_proba)), color in zip(models_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    
    ax.set_xlabel('1 - Especificitat (FPR)', fontsize=12)
    ax.set_ylabel('Sensitivitat (TPR)', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    return ax


def calibration_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                          model_name: str = '', ax=None) -> plt.Axes:
    """Analitza la calibració del model (fiabilitat de les probabilitats)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=8)
    
    ax.plot(prob_pred, prob_true, 'o-', color='#4361EE', linewidth=2.5,
            markersize=8, label=f'{model_name}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Calibraci perfecta')
    
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='#4361EE')
    
    ax.set_xlabel('Probabilitat Predicta', fontsize=12)
    ax.set_ylabel('Fracció de Positius Reals', fontsize=12)
    ax.set_title(f'Corba de Calibraci — {model_name}', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return ax


def generate_clinical_report(y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray, model_name: str) -> str:
    """
    Genera un informe textual orientat a clinicians.
    """
    metrics = compute_clinical_metrics(y_true, y_pred, y_proba)
    n = len(y_true)
    
    report = f"""
{'='*60}
INFORME DE RENDIMENT DEL MODEL: {model_name.upper()}
{'='*60}

RESUM:
  Pacients avaluats: {n}
  Positius reals (resposta): {y_true.sum()} ({y_true.mean():.1%})
  Positius predits: {y_pred.sum()} ({y_pred.mean():.1%})

MÈTRIQUES PRINCIPALS:
  AUC-ROC:        {metrics.get('auc_roc', 'N/A'):.3f}
  F1 Score:       {metrics['f1']:.3f}
  Accuracy:       {metrics['accuracy']:.3f}

MÈTRIQUES CLÍNIQUES:
  Sensitivitat:   {metrics['sensitivity']:.3f}  (probabilitat detectar responedors)
  Especificitat:  {metrics['specificity']:.3f}  (probabilitat detectar no-responedors)
  PPV (precision):{metrics['ppv']:.3f}  (% prediccions positives correctes)
  NPV:            {metrics['npv']:.3f}  (% prediccions negatives correctes)

MATRIU DE CONFUSIÓ:
  TP (resposta correctament identificada): {metrics['tp']}
  TN (no-resposta correctament identificada): {metrics['tn']}
  FP (fals positiu - tractament innecessari): {metrics['fp']}
  FN (fals negatiu - resposta perduda): {metrics['fn']}

INTERPRETACIÓ CLÍNICA:
  Per cada 100 pacients avaluats:
  - El model identificaria correctament ~{int(metrics['sensitivity']*y_true.mean()*100)} responedors
  - Derivaria innecessàriament ~{int(metrics['fp']/n*100)} no-responedors
  - Perdria ~{int(metrics['fn']/n*100)} responedors potencials

{'='*60}
"""
    return report
