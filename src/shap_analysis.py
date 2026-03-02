"""
shap_analysis.py
Funcions SHAP per a interpretabilitat global i local del model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Classe per a l'anàlisi SHAP de models de predicció.
    Suporta TreeExplainer (RF, XGBoost) i LinearExplainer (LR).
    """

    def __init__(self, model, feature_names: List[str],
                 feature_labels: Optional[List[str]] = None):
        """
        Args:
            model: Model sklearn/XGBoost entrenat
            feature_names: Noms de les columnes de features
            feature_labels: Etiquetes llegibles per humans (opcional)
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_labels = feature_labels or feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def fit(self, X: np.ndarray, model_type: str = 'tree'):
        """
        Inicialitza l'explainer SHAP.
        
        Args:
            X: Dades de referència (background)
            model_type: 'tree' per RF/XGB, 'linear' per LR
        """
        # Extreure el classificador del pipeline si cal
        clf = self.model
        if hasattr(self.model, 'named_steps'):
            clf = self.model.named_steps.get('classifier', self.model)
            # Transformar X amb el preprocessador
            if 'preprocessor' in self.model.named_steps:
                X = self.model.named_steps['preprocessor'].transform(X)
            elif 'imputer' in self.model.named_steps:
                X = self.model.named_steps['imputer'].transform(X)

        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(clf)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(clf, X)
        else:
            self.explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X, 50))

        self.shap_values = self.explainer.shap_values(X)
        self.expected_value = self.explainer.expected_value

        # Per classificadors binaris, agafar la classe positiva
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

        logger.info(f'SHAP values calculats: {self.shap_values.shape}')
        logger.info(f'Expected value (base rate): {self.expected_value:.3f}')

        return self

    def global_importance(self) -> pd.DataFrame:
        """Retorna la importància global de features (|SHAP| mean)."""
        if self.shap_values is None:
            raise ValueError("Crida fit() primer.")

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'label': self.feature_labels,
            'mean_abs_shap': mean_abs,
        }).sort_values('mean_abs_shap', ascending=False)

        return importance_df

    def plot_summary(self, X: np.ndarray, max_display: int = 15,
                     plot_type: str = 'dot',
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Genera el SHAP summary plot (beeswarm o bar).
        """
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.5)))
        plt.sca(ax)

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_labels,
            plot_type=plot_type,
            show=False,
            max_display=max_display,
            alpha=0.7
        )

        ax.set_title('SHAP Summary Plot — Importància Global de Features',
                     fontweight='bold', fontsize=13, pad=12)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_dependence(self, feature_idx: int,
                        X: np.ndarray,
                        interaction_idx: Optional[int] = None,
                        ax=None,
                        save_path: Optional[str] = None) -> plt.Axes:
        """
        Genera un SHAP dependence plot per a una feature.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        x_vals = X[:, feature_idx]
        y_shap = self.shap_values[:, feature_idx]

        if interaction_idx is not None:
            colors = X[:, interaction_idx]
            scatter = ax.scatter(x_vals, y_shap, c=colors,
                                 cmap='coolwarm', alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax,
                         label=self.feature_labels[interaction_idx])
        else:
            ax.scatter(x_vals, y_shap, c='#4361EE', alpha=0.7, s=50)

        # Línea de tendència
        z = np.polyfit(x_vals, y_shap, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.6)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(self.feature_labels[feature_idx], fontsize=12)
        ax.set_ylabel('Valor SHAP', fontsize=12)
        ax.set_title(f'SHAP Dependence: {self.feature_labels[feature_idx]}',
                     fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_waterfall(self, patient_idx: int, X: np.ndarray,
                       max_display: int = 10,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Genera el SHAP waterfall plot per a un pacient individual.
        """
        patient_shap = self.shap_values[patient_idx]
        patient_features = X[patient_idx]
        base = float(self.expected_value)

        # Top features per impacte absolut
        top_idx = np.argsort(np.abs(patient_shap))[::-1][:max_display]

        feats = [self.feature_labels[i] for i in top_idx]
        vals_shap = patient_shap[top_idx]
        feat_vals = patient_features[top_idx]

        # Ordenar per valor SHAP
        sort_order = np.argsort(vals_shap)
        feats = [feats[i] for i in sort_order]
        vals_shap = vals_shap[sort_order]
        feat_vals = feat_vals[sort_order]

        fig, ax = plt.subplots(figsize=(10, 8))

        bar_colors = ['#27AE60' if v > 0 else '#E74C3C' for v in vals_shap]
        bars = ax.barh(
            [f'{f}\n(val: {v:.2f})' for f, v in zip(feats, feat_vals)],
            vals_shap,
            color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5
        )

        # Anotació de valors
        for bar, val in zip(bars, vals_shap):
            ax.text(
                val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold'
            )

        ax.axvline(0, color='black', linewidth=1.5)

        # Predicció final
        pred_prob = base + patient_shap.sum()
        ax.set_title(
            f'SHAP Waterfall — Pacient {patient_idx}\n'
            f'Base rate: {base:.3f} → Predicció: {pred_prob:.3f}',
            fontweight='bold', fontsize=12
        )
        ax.set_xlabel('Valor SHAP (contribució a la predicció)', fontsize=11)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def compare_patients(self, responder_idx: int, nonresponder_idx: int,
                          X: np.ndarray,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Compara els SHAP waterfall de dos pacients (responedor vs no-responedor).
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        cases = [
            (responder_idx, 'RESPONEDOR', '#27AE60'),
            (nonresponder_idx, 'NO-RESPONEDOR', '#E74C3C'),
        ]

        for ax, (idx, label, color) in zip(axes, cases):
            patient_shap = self.shap_values[idx]
            patient_features = X[idx]
            pred = float(self.expected_value) + patient_shap.sum()

            top_idx = np.argsort(np.abs(patient_shap))[::-1][:10]
            feats = [self.feature_labels[i] for i in top_idx]
            shap_v = patient_shap[top_idx]
            feat_v = patient_features[top_idx]

            sort_ord = np.argsort(shap_v)
            feats = [feats[i] for i in sort_ord]
            shap_v = shap_v[sort_ord]
            feat_v = feat_v[sort_ord]

            bar_colors = ['#27AE60' if v > 0 else '#E74C3C' for v in shap_v]
            ax.barh(
                [f'{f}\n({v:.2f})' for f, v in zip(feats, feat_v)],
                shap_v,
                color=bar_colors, alpha=0.85, edgecolor='white'
            )
            ax.axvline(0, color='black', linewidth=1.5)
            ax.set_title(
                f'Pacient {idx} — {label}\nP(resposta) = {pred:.2f}',
                fontweight='bold', fontsize=11, color=color
            )
            ax.set_xlabel('Valor SHAP', fontsize=10)

        plt.suptitle('Comparació SHAP: Responedor vs. No-Responedor',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def shap_interaction_matrix(self, X: np.ndarray,
                                 top_n: int = 8) -> pd.DataFrame:
        """
        Calcula la matriu d'interacció entre features (correlació SHAP).
        """
        top_feats = np.argsort(
            np.abs(self.shap_values).mean(axis=0)
        )[::-1][:top_n]

        shap_subset = self.shap_values[:, top_feats]
        labels = [self.feature_labels[i] for i in top_feats]

        corr_matrix = pd.DataFrame(shap_subset, columns=labels).corr()
        return corr_matrix
