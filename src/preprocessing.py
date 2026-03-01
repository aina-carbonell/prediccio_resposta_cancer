"""
preprocessing.py
================
Pipeline de preprocessament per al projecte d'immunoteràpia.
Inclou: imputació, escalat, codificació i maneig d'imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class GeneExpressionNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalitzador d'expressió gènica.
    
    Transformació log1p (estàndard per RNA-seq TPM/FPKM) i
    z-score per gen.
    """
    
    def __init__(self, apply_log=True, z_score=True):
        self.apply_log = apply_log
        self.z_score = z_score
        self.gene_means_ = None
        self.gene_stds_ = None
    
    def fit(self, X, y=None):
        X_arr = np.array(X, dtype=float)
        if self.apply_log:
            X_arr = np.log1p(X_arr.clip(0))
        if self.z_score:
            self.gene_means_ = X_arr.mean(axis=0)
            self.gene_stds_ = X_arr.std(axis=0) + 1e-8
        return self
    
    def transform(self, X, y=None):
        X_arr = np.array(X, dtype=float)
        if self.apply_log:
            X_arr = np.log1p(X_arr.clip(0))
        if self.z_score and self.gene_means_ is not None:
            X_arr = (X_arr - self.gene_means_) / self.gene_stds_
        return X_arr


class ClinicalFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Processa variables clíniques: codificació i transformació.
    """
    
    CATEGORICAL_FEATURES = {
        'gender': {'M': 1, 'F': 0, 'male': 1, 'female': 0},
        'stage': {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Codificar gènere si és string
        if 'gender' in df.columns and df['gender'].dtype == object:
            df['gender'] = df['gender'].map(self.CATEGORICAL_FEATURES['gender'])
        
        # Codificar estadiatge si és string
        if 'stage' in df.columns and df['stage'].dtype == object:
            df['stage'] = df['stage'].map(self.CATEGORICAL_FEATURES['stage'])
        
        return df.values


def create_preprocessing_pipeline(
    imputation_strategy: str = 'median',
    scaling: str = 'standard',
    handle_imbalance: str = 'class_weight'
) -> Pipeline:
    """
    Crea un pipeline de preprocessament complet.
    
    Parameters
    ----------
    imputation_strategy : 'median', 'mean', 'knn'
    scaling : 'standard', 'robust', 'none'
    handle_imbalance : 'class_weight', 'smote', 'none'
    
    Returns
    -------
    Pipeline sklearn
    
    Notes
    -----
    Maneig del class imbalance:
    
    - 'class_weight': Recomanat per n petit. No crea dades artificials.
      Evita overfitting. Implementat directament al classificador.
      
    - 'smote': Genera mostres sintètiques de la classe minoritària.
      Millora AUC però pot crear artefactes si n < 100.
      Requereix imblearn.Pipeline en comptes de sklearn.Pipeline.
      
    - 'none': Útil per analitzar el comportament natural del model
      o si les classes estan relativament balancejades.
    """
    
    steps = []
    
    # Imputació
    if imputation_strategy == 'knn':
        steps.append(('imputer', KNNImputer(n_neighbors=5)))
    else:
        steps.append(('imputer', SimpleImputer(strategy=imputation_strategy)))
    
    # Escalat
    if scaling == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaling == 'standard':
        steps.append(('scaler', StandardScaler()))
    
    if handle_imbalance == 'smote' and len(steps) > 0:
        # Retorna imblearn Pipeline per suportar SMOTE
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=3)))
        return ImbPipeline(steps)
    
    return Pipeline(steps)


def compute_class_weight(y: np.ndarray) -> dict:
    """
    Calcula els pesos de classe per corregir l'imbalance.
    
    Returns
    -------
    dict {class_label: weight}
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def validate_features(df: pd.DataFrame, required_features: list) -> tuple:
    """
    Valida que les features requerides estan presents.
    
    Returns
    -------
    (present_features, missing_features)
    """
    present = [f for f in required_features if f in df.columns]
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        print(f"⚠️  Features faltants: {missing}")
        print(f"   Seran substituïdes per 0 o mediana.")
    
    return present, missing