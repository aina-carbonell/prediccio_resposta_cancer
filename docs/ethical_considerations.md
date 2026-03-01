# Consideracions Ètiques i Limitacions del Projecte

## ⚠️ AVÍS IMPORTANT

**Aquest projecte és exclusivament per a recerca i aprenentatge. ELS MODELS GENERATS NO HAN DE SER USATS PER PRENDRE DECISIONS CLÍNIQUES REALS.**

---

## 1. Limitacions de les Dades

### 1.1 Grandària de la Mostra

Els estudis públics de melanoma amb immunoteràpia i resposta documentada (Riaz 2017, Hugo 2016) contenen entre 38 i 51 pacients. Aquesta mida és insuficient per:

- Detectar efectes subtils de biomarcadors
- Generalitzar a subpoblacions
- Assegurar la robustesa del model davant distribucions shift

**Recomanació**: Qualsevol model clínic hauria de basar-se en cohorts de ≥200-500 pacients amb seguiment prospectiu.

### 1.2 Biaixos Poblacionals (Representativitat)

| Font | Biaix Identificat | Impacte Potencial |
|------|-------------------|-------------------|
| TCGA-SKCM | ~75% pacients blancs, majoria nordamericans | El model pot no generalitzar a poblacions asiàtiques, africanes o llatinoamericanes |
| cBioPortal datasets | Pacients de centres acadèmics terciaris | Pot no representar pacients de centres comunitaris o amb perfils socioeconòmics diverses |
| Variables clíniques | Sobrerepresentació d'estadiatge IV | Menys fiable per a estadiatge III inicial |

### 1.3 Qualitat i Completesa de les Dades

- **Missing data**: PD-L1 IHC (~8-15% missing), ECOG score inconsistentment reportat
- **Heterogeneïtat tècnica**: Dades RNA-seq de plataformes diverses (Illumina HiSeq vs NovaSeq) poden introduir batch effects
- **Variable objectiu**: TCGA no té dades de resposta a immunoteràpia. S'utilitza supervivència com a proxy, que és imperfecta (influïda per tractaments posteriors, comorbiditats, etc.)

---

## 2. Limitacions del Model

### 2.1 Validació

El model s'ha avaluat amb cross-validation interna, però **no ha estat validat externament** en:
- Una cohort independent pròspectiva
- Pacients tractats amb anti-PD-L1 (distincts d'anti-PD1)
- Pacients amb tractaments combinats (anti-PD1 + anti-CTLA4)
- Tipologies de melanoma menys freqüents (acral, mucós)

### 2.2 Risc de Sobreajustament

Amb n~120 i ~25 features, el ratio de features/mostres és al límit. Recomanem:
- Regularització aggressiva (L1/L2 penaltyà Logistic Regression)
- Cross-validation estricta sense leakage
- Evitar feature engineering post-hoc sense re-validació

### 2.3 Deriva Temporal (Temporal Drift)

Les pràctiques clíniques evolucionen: criteris de resposta RECIST, dosis de tractament, selecció de pacients. Un model entrenat amb dades de 2015-2018 pot tenir menor rendiment en cohorts de 2024+.

---

## 3. Consideracions per a Ús Clínic

### 3.1 No s'ha de fer mai

❌ Usar les prediccions per decidir si un pacient rep o no immunoteràpia  
❌ Substituir el judici clínic i l'assessorament oncològic  
❌ Aplicar-lo a tipologies de càncer no representades en el training  
❌ Usar-lo sense supervisió mèdica qualificada  

### 3.2 Ús acceptable (recerca)

✅ Generació d'hipòtesis per a estudis prospectius  
✅ Estratificació de pacients en assaigs clínics (com a factor de randomització, no decisiu)  
✅ Identificació de biomarcadors candidats per a validació experimental  
✅ Educació i comunicació del concepte de medicina de precisió  

### 3.3 Ruta cap a Validació Clínica

Si es volgués continuar cap a aplicació clínica:

1. **Validació retrospectiva** en cohorts independents (≥200 pacients)
2. **Comitè d'ètica** (IRB/CEI): Aprovació per a ús de dades
3. **Estudi prospectiu** com a biomarcador companion en assaig clínic
4. **Regulatory pathway**: FDA Breakthrough Device Designation o equivalent EMA
5. **Health Economics Analysis**: Cost-efectivitat del test predictiu

---

## 4. Transparència i Explicabilitat

### 4.1 SHAP com a Eina de Transparència

L'ús de SHAP permet explicar cada predicció individual, però cal tenir cura de:

- **No sobre-interpretar**: Una contribució SHAP alta no implica causalitat biològica
- **Context poblacional**: SHAP valors canvien si la distribució de la cohort canvia
- **Interaccions**: Les interaccions entre variables (SHAP interaction values) poden ser importants i sovint s'ometen

### 4.2 Comunicació a Pacients

Si s'usés en context de recerca clínica, les prediccions haurien d'explicar-se als pacients:

> *"Tenim una eina computacional que analitza les característiques del vostre tumor per estimar si el tractament d'immunoteràpia podria ser efectiu. Aquesta eina és experimental i no és definitiva. El vostre oncòleg té en compte molts altres factors que la màquina no pot capturar."*

---

## 5. Recomanacions per a Recercadors

1. **Replicabilitat**: Sempre fixar random seeds, versionar els entorns (requirements.txt, environment.yml)
2. **Transparència**: Publicar el codi i les dades derivades (FAIR principles)
3. **Limitacions visibles**: Reportar sempre els intervals de confiança, no només valors puntuals
4. **Diversitat en l'equip**: Involucrar oncòlegs, estadistes, pacients i ètics en el disseny
5. **Seguiment a llarg termini**: Monitoritzar el rendiment del model amb dades noves (model monitoring)

---

## 6. Recursos Addicionals

- [TRIPOD-AI Guidelines](https://www.equator-network.org/): Reporting guidelines per a models de predicció clínica
- [FDA Guidance on AI/ML-Based Software](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [ESMO Guidelines on Biomarkers](https://www.esmo.org/guidelines/biomarkers)

---

*Document elaborat seguint principis d'IA responsable i ètica en salut digital.*  
*Última revisió: 2025*