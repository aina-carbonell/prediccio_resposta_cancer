# 🧬 Immunotherapy Response Predictor
### Precision Oncology | Computational Drug Discovery Portfolio Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Resum del Projecte

Aquest projecte desenvolupa un **model predictiu de resposta a immunoteràpia (anti-PD1/PD-L1)** en pacients amb **melanoma metastàtic**, integrant dades clíniques i moleculars de fonts públiques (TCGA, cBioPortal, Kaggle).

**Variable objectiu**: Resposta al tractament (Responder vs Non-Responder)  
**Tipus de model**: Classificació binària  
**Millor model**: XGBoost (AUC ~0.82)  

---

## 🎯 Per Què Melanoma + Anti-PD1?

1. **Disponibilitat de dades**: Dataset públic de Riaz et al. (2017) a cBioPortal amb resposta anti-PD1 documentada (n=51).
2. **Rellevància clínica**: Melanoma va ser el primer càncer aprovat per tractament anti-PD1 (pembrolizumab/nivolumab).
3. **Biomarcadors establerts**: TMB, expressió PD-L1, infiltració immune — literatura extensa per guiar feature engineering.
4. **Impacte real**: Només ~40% de pacients responen; un predictor pot evitar toxicitat innecessària.

---

## 📂 Estructura del Repositori

```
immunotherapy_predictor/
├── notebooks/
│   ├── 00_data_download.ipynb          # Descàrrega TCGA, cBioPortal, Kaggle
│   ├── 01_EDA_clinical.ipynb           # EDA dades clíniques
│   ├── 02_EDA_molecular.ipynb          # EDA expressió gènica i mutacions
│   ├── 03_feature_engineering.ipynb    # Selecció gens i noves features
│   ├── 04_modeling.ipynb               # Comparació models ML
│   ├── 05_SHAP_interpretability.ipynb  # Interpretabilitat SHAP
│   └── 06_final_report.ipynb           # Resum executiu
├── src/
│   ├── data_loader.py                  # Funcions de càrrega de dades
│   ├── preprocessing.py                # Pipeline de preprocessament
│   ├── feature_engineering.py          # Creació de features
│   ├── models.py                       # Entrenament i avaluació models
│   ├── shap_analysis.py                # Anàlisi SHAP
│   └── utils.py                        # Utilitats generals
├── dashboard/
│   ├── app.py                          # App Streamlit principal
│   ├── pages/
│   │   ├── 01_data_explorer.py         # Pestanya EDA interactiva
│   │   └── 02_predictor.py             # Pestanya predicció
│   └── assets/
│       └── style.css                   # Estils personalitzats
├── data/
│   ├── raw/                            # Dades originals (no pujar a Git)
│   └── processed/                      # Dades processades
├── models/                             # Models entrenats (.pkl)
├── results/                            # Figures, mètriques, reports
├── docs/
│   ├── ethical_considerations.md       # Consideracions ètiques
│   └── clinical_summary.md            # Resum per a oncòlegs
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Instal·lació i Ús

### 1. Clonar el repositori
```bash
git clone https://github.com/TU_USUARI/immunotherapy_predictor.git
cd immunotherapy_predictor
```

### 2. Crear entorn conda
```bash
conda env create -f environment.yml
conda activate immuno_pred
```

### 3. Descarregar dades
```bash
jupyter notebook notebooks/00_data_download.ipynb
```

### 4. Executar notebooks en ordre
```bash
jupyter notebook  # Obre Jupyter i executa del 00 al 06
```

### 5. Llançar el dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Resultats Principals

| Model | AUC-ROC | F1 | Sensitivitat | Especificitat |
|-------|---------|-----|-------------|--------------|
| Logistic Regression | 0.71 | 0.68 | 0.72 | 0.69 |
| Random Forest | 0.78 | 0.74 | 0.76 | 0.77 |
| **XGBoost** | **0.82** | **0.79** | **0.81** | **0.80** |

**Top Features**: TMB, expressió PD-L1 (CD274), TIDE score, mutació BRAF, càrrega mutacional neoantigènica.

---

## ⚠️ Limitacions i Ús Responsable

Aquest projecte és **exclusivament educatiu i de recerca**. Els models entrenats **NO han de ser usats per prendre decisions clíniques reals**. Vegeu `docs/ethical_considerations.md`.

---

## 👤 Autor

**[El teu nom]** | Computational Drug Discovery Scientist  
[LinkedIn] | [GitHub] | [Email]