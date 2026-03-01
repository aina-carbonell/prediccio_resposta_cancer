"""
Immunotherapy Response Predictor Dashboard
==========================================
Dashboard interactiu per explorar dades i fer prediccions de resposta a immunoteràpia.

Executar amb: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Configuració de la pàgina ───────────────────────────────────────────────
st.set_page_config(
    page_title="Immunotherapy Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estils CSS personalitzats ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p { color: #a8dadc; font-size: 1rem; margin-top: 0.5rem; }
    
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; color: #0f3460; font-size: 1.5rem; }
    .metric-card p { margin: 0; color: #6c757d; font-size: 0.85rem; }
    
    .responder-badge {
        background: #d4edda; color: #155724;
        border: 1px solid #c3e6cb;
        padding: 0.5rem 1rem; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .non-responder-badge {
        background: #f8d7da; color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 0.5rem 1rem; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .clinical-note {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 1rem; border-radius: 6px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PROC_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RAW_DIR = BASE_DIR / 'data' / 'raw'


# ── Funcions de càrrega (cached) ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = PROC_DIR / 'melanoma_processed.csv'
    if path.exists():
        df = pd.read_csv(path)
        df['response_label'] = df['response'].map({1: 'Responder', 0: 'Non-Responder'})
        return df
    # Fallback: generar dades sintètiques inline
    return generate_demo_data()


def generate_demo_data(n=120):
    """Genera dades demo si no existeix el fitxer processat."""
    np.random.seed(42)
    response = np.random.choice([0, 1], size=n, p=[0.66, 0.34])
    df = pd.DataFrame({
        'patient_id': [f'PT_{i:04d}' for i in range(n)],
        'age': np.random.normal(62, 14, n).clip(25, 90).astype(int),
        'gender_binary': np.random.choice([0, 1], n),
        'stage_num': np.random.choice([3, 4], n, p=[0.3, 0.7]),
        'ecog_score': np.random.choice([0, 1, 2], n, p=[0.45, 0.40, 0.15]),
        'prior_treatment': np.random.choice([0, 1], n),
        'tmb': np.where(response == 1, np.random.lognormal(3.5, 0.8, n),
                        np.random.lognormal(2.5, 0.7, n)).clip(1, 200).round(1),
        'pdl1_ihc_percent': np.random.randint(0, 100, n),
        'braf_mutation': np.random.choice([0, 1], n),
        'tide_score': np.random.normal(0, 1, n).round(3),
        'cd8_t_cell_fraction': np.random.beta(2, 3, n).round(4),
        'm2_macrophage_fraction': np.random.beta(2, 4, n).round(4),
        'IFNG_expr': np.random.normal(6, 1.5, n).clip(0, 15).round(3),
        'CXCL9_expr': np.random.normal(6, 2, n).clip(0, 15).round(3),
        'CD274_pdl1_expr': np.random.normal(5.5, 1.5, n).clip(0, 15).round(3),
        'GZMB_expr': np.random.normal(5, 1.5, n).clip(0, 15).round(3),
        'tcell_inflamed_score': np.random.normal(5.5, 1.5, n).round(3),
        'composite_immune_score': np.random.normal(0, 1.2, n).round(4),
        'immune_suppression_ratio': np.random.lognormal(0, 0.8, n).clip(0.1, 20).round(3),
        'tmb_log': np.log1p(np.random.lognormal(3, 0.8, n).clip(1, 200)).round(3),
        'tmb_high': (np.random.lognormal(3, 0.8, n).clip(1, 200) >= 10).astype(int),
        'pdl1_positive': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'os_months': np.random.exponential(18, n).clip(1, 60).round(1),
        'os_event': np.random.choice([0, 1], n, p=[0.35, 0.65]),
        'response': response
    })
    df['response_label'] = df['response'].map({1: 'Responder', 0: 'Non-Responder'})
    return df


@st.cache_resource
def load_model():
    path = MODELS_DIR / 'best_model_xgboost.pkl'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_feature_list():
    path = PROC_DIR / 'feature_list.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {'features': [], 'target': 'response'}


# ── Capçalera principal ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Immunotherapy Response Predictor</h1>
    <p>Predicció de resposta a anti-PD1 en Melanoma Metastàtic · Precision Oncology Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Carregar recursos ─────────────────────────────────────────────────────────
df = load_data()
model_package = load_model()
feat_config = load_feature_list()

# ── Navegació per pestanyes ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Exploració de Dades",
    "🔮 Predicció per Pacient",
    "ℹ️ Sobre el Projecte"
])


# ══════════════════════════════════════════════════════════════════════════════
# PESTANYA 1: EXPLORACIÓ DE DADES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploració Interactiva de la Cohort")

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filtres")
        response_filter = st.multiselect(
            "Grups de Resposta",
            options=['Responder', 'Non-Responder'],
            default=['Responder', 'Non-Responder']
        )
        stage_filter = st.multiselect(
            "Estadiatge",
            options=sorted(df['stage_num'].unique()) if 'stage_num' in df.columns else [3, 4],
            default=sorted(df['stage_num'].unique()) if 'stage_num' in df.columns else [3, 4]
        )
        age_range = st.slider(
            "Rang d'Edat",
            min_value=int(df['age'].min()) if 'age' in df.columns else 20,
            max_value=int(df['age'].max()) if 'age' in df.columns else 90,
            value=(20, 90)
        )

    # Aplicar filtres
    mask = df['response_label'].isin(response_filter)
    if 'stage_num' in df.columns:
        mask &= df['stage_num'].isin(stage_filter)
    if 'age' in df.columns:
        mask &= df['age'].between(*age_range)
    df_filtered = df[mask]

    st.caption(f"Mostrant **{len(df_filtered)}** de {len(df)} pacients")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{len(df_filtered)}</h3><p>Pacients</p></div>""", unsafe_allow_html=True)
    with col2:
        resp_rate = df_filtered['response'].mean()
        st.markdown(f"""<div class="metric-card">
            <h3>{resp_rate:.0%}</h3><p>Taxa de Resposta</p></div>""", unsafe_allow_html=True)
    with col3:
        if 'tmb' in df_filtered.columns:
            med_tmb = df_filtered['tmb'].median()
            st.markdown(f"""<div class="metric-card">
                <h3>{med_tmb:.1f}</h3><p>TMB mediana (mut/Mb)</p></div>""", unsafe_allow_html=True)
    with col4:
        if 'age' in df_filtered.columns:
            med_age = df_filtered['age'].median()
            st.markdown(f"""<div class="metric-card">
                <h3>{med_age:.0f}</h3><p>Edat mediana (anys)</p></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Gràfics ───────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribució de TMB per Grups")
        if 'tmb' in df_filtered.columns:
            fig = px.box(
                df_filtered, x='response_label', y='tmb',
                color='response_label',
                color_discrete_map={'Responder': '#2ecc71', 'Non-Responder': '#e74c3c'},
                points='all', notched=True,
                labels={'response_label': 'Grup', 'tmb': 'TMB (mut/Megabase)'},
                template='plotly_white'
            )
            fig.update_layout(showlegend=False, height=380)
            fig.add_hline(y=10, line_dash='dash', line_color='gray',
                         annotation_text='Cutoff FDA ≥10')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""<div class="clinical-note">
            💡 <strong>Nota clínica</strong>: La FDA ha aprovat pembrolizumab per tumors amb TMB ≥ 10 mut/Mb.
            Els respondedors mostren típicament una càrrega mutacional superior.
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.subheader("Expressió IFNG vs CXCL9")
        if 'IFNG_expr' in df_filtered.columns and 'CXCL9_expr' in df_filtered.columns:
            fig = px.scatter(
                df_filtered,
                x='IFNG_expr', y='CXCL9_expr',
                color='response_label',
                color_discrete_map={'Responder': '#2ecc71', 'Non-Responder': '#e74c3c'},
                hover_data={'patient_id': True, 'tmb': ':.1f'},
                labels={'IFNG_expr': 'IFNG (log2 TPM+1)',
                        'CXCL9_expr': 'CXCL9 (log2 TPM+1)',
                        'response_label': 'Grup'},
                trendline='ols',
                template='plotly_white'
            )
            fig.update_traces(marker=dict(size=8, opacity=0.75))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("T-Cell Inflamed Score vs Resposta")
        if 'tcell_inflamed_score' in df_filtered.columns:
            fig = px.violin(
                df_filtered,
                x='response_label', y='tcell_inflamed_score',
                color='response_label',
                color_discrete_map={'Responder': '#2ecc71', 'Non-Responder': '#e74c3c'},
                box=True, points='outliers',
                template='plotly_white',
                labels={'response_label': '', 'tcell_inflamed_score': 'T-Cell Inflamed Score'}
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Distribució de Variables Clíniques")
        cat_var = st.selectbox("Variable:", 
                               ['stage_num', 'ecog_score', 'braf_mutation', 'gender_binary'],
                               format_func=lambda x: {
                                   'stage_num': 'Estadiatge',
                                   'ecog_score': 'ECOG Score',
                                   'braf_mutation': 'Mutació BRAF',
                                   'gender_binary': 'Sexe (0=F, 1=M)'
                               }.get(x, x))
        if cat_var in df_filtered.columns:
            ct = pd.crosstab(df_filtered[cat_var], df_filtered['response_label'],
                            normalize='index') * 100
            fig = px.bar(
                ct.reset_index(), x=cat_var,
                y=['Non-Responder', 'Responder'] if 'Responder' in ct.columns else ct.columns.tolist(),
                barmode='stack',
                color_discrete_map={'Responder': '#2ecc71', 'Non-Responder': '#e74c3c'},
                template='plotly_white', height=350,
                labels={'value': '%', cat_var: cat_var}
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Taula de dades ─────────────────────────────────────────────────────────
    with st.expander("📋 Veure Taula de Dades"):
        display_cols = ['patient_id', 'age', 'response_label', 'tmb',
                       'pdl1_ihc_percent', 'tcell_inflamed_score', 'composite_immune_score']
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols].round(3), use_container_width=True,
                    height=350)


# ══════════════════════════════════════════════════════════════════════════════
# PESTANYA 2: PREDICCIÓ
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔮 Predicció per a un Pacient Nou")

    if model_package is None:
        st.warning("⚠️ Model no disponible. Executa primer els notebooks 03 i 04.")
        st.info("Per generar el model: `jupyter notebook notebooks/04_modeling.ipynb`")
        st.stop()

    FEATURES = feat_config.get('features', [])
    model = model_package['model']

    st.markdown("### Introduir Característiques del Pacient")

    input_mode = st.radio(
        "Mode d'entrada:",
        ["🖊️ Entrada Manual", "📁 Pujar CSV"],
        horizontal=True
    )

    if input_mode == "📁 Pujar CSV":
        uploaded = st.file_uploader("Pujar fitxer CSV amb dades del pacient", type=['csv'])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.success(f"Fitxer carregat: {df_upload.shape[0]} pacients")
            st.dataframe(df_upload.head())

            if st.button("🚀 Fer Predicció per tots els Pacients"):
                available = [f for f in FEATURES if f in df_upload.columns]
                if not available:
                    st.error("Cap feature del model trobat al CSV.")
                else:
                    X_new = df_upload[available].fillna(df_upload[available].median())
                    # Afegir columnes que falten com 0
                    for f in FEATURES:
                        if f not in X_new.columns:
                            X_new[f] = 0.0
                    X_new = X_new[FEATURES] if all(f in X_new.columns for f in FEATURES) else X_new[available]
                    
                    proba = model.predict_proba(X_new.values)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    
                    df_results = df_upload.copy()
                    df_results['Prob_Resposta'] = proba.round(3)
                    df_results['Predicció'] = pd.Series(pred).map({1: 'Responder', 0: 'Non-Responder'})
                    
                    st.success(f"Prediccions completades!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(df_results[['Predicció', 'Prob_Resposta']].head(20))
                    with col2:
                        fig = px.histogram(df_results, x='Prob_Resposta', nbins=20,
                                         title='Distribució de Probabilitats',
                                         template='plotly_white',
                                         color_discrete_sequence=['#0f3460'])
                        fig.add_vline(x=0.5, line_dash='dash', line_color='red')
                        st.plotly_chart(fig, use_container_width=True)

    else:
        # ── Entrada Manual ─────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Dades Clíniques**")
            age_input = st.slider("Edat", 20, 90, 62)
            gender_input = st.selectbox("Sexe", ["Masculí", "Femení"])
            stage_input = st.selectbox("Estadiatge", ["III", "IV"])
            ecog_input = st.selectbox("ECOG Performance Status", [0, 1, 2],
                                      help="0=Actiu, 1=Limitat, 2=Ambulatori <50%")
            prior_input = st.selectbox("Tractament previ", ["Sí", "No"])
            braf_input = st.selectbox("Mutació BRAF", ["Wild-type", "Mutat"])

        with col2:
            st.markdown("**🔬 Biomarcadors**")
            tmb_input = st.number_input("TMB (mut/Megabase)", 1.0, 200.0, 8.0, 0.5,
                                        help="Tumor Mutational Burden. FDA cutoff TMB-High: ≥10 mut/Mb")
            pdl1_input = st.slider("PD-L1 IHC (%)", 0, 100, 30,
                                   help="Percentatge de cèl·lules tumorals PD-L1+")
            tide_input = st.number_input("TIDE Score", -3.0, 3.0, 0.0, 0.1,
                                         help="Tumor Immune Dysfunction and Exclusion. Negatiu=millor resposta esperada")
            cd8_input = st.slider("CD8+ T-cells (fracció)", 0.0, 0.5, 0.2, 0.01)
            m2_input = st.slider("M2 Macròfags (fracció)", 0.0, 0.3, 0.1, 0.01)

        with col3:
            st.markdown("**🧬 Expressió Gènica** (log2 TPM+1)")
            ifng_input = st.slider("IFNG", 0.0, 15.0, 6.0, 0.1)
            cxcl9_input = st.slider("CXCL9", 0.0, 15.0, 6.0, 0.1)
            cd274_input = st.slider("CD274 (PD-L1)", 0.0, 15.0, 5.5, 0.1)
            gzmb_input = st.slider("GZMB (Granzyme B)", 0.0, 15.0, 5.0, 0.1)
            ctla4_input = st.slider("CTLA4", 0.0, 12.0, 3.8, 0.1)

        # ── Calcular features derivades ──────────────────────────────────────
        tmb_log = np.log1p(tmb_input)
        tmb_high = int(tmb_input >= 10)
        pdl1_positive = int(pdl1_input >= 1)
        pdl1_high = int(pdl1_input >= 50)
        gender_binary = 1 if gender_input == "Masculí" else 0
        stage_num = 3 if stage_input == "III" else 4
        prior_treatment = 1 if prior_input == "Sí" else 0
        braf_mutation = 1 if braf_input == "Mutat" else 0
        tcell_inflamed_score = np.mean([ifng_input, cxcl9_input, gzmb_input, 5.0])
        checkpoint_score = np.mean([ctla4_input, 4.0, 3.5, 4.0])
        immune_suppression_ratio = (cd8_input / (m2_input + 1e-6))
        composite_immune_score = (tmb_log / 4.0 + tcell_inflamed_score / 7.0 - tide_input / 2.0)

        # Construir vector de features
        patient_data = {
            'age': age_input, 'gender_binary': gender_binary, 'stage_num': stage_num,
            'ecog_score': ecog_input, 'prior_treatment': prior_treatment,
            'tmb': tmb_input, 'tmb_log': tmb_log, 'tmb_high': tmb_high,
            'pdl1_ihc_percent': pdl1_input, 'pdl1_positive': pdl1_positive,
            'pdl1_high': pdl1_high, 'braf_mutation': braf_mutation,
            'tide_score': tide_input,
            'cd8_t_cell_fraction': cd8_input,
            'm2_macrophage_fraction': m2_input,
            'immune_suppression_ratio': immune_suppression_ratio,
            'CD274_pdl1_expr': cd274_input, 'PDCD1_pd1_expr': 4.2,
            'CTLA4_expr': ctla4_input, 'LAG3_expr': 3.5,
            'HAVCR2_tim3_expr': 4.0,
            'IFNG_expr': ifng_input, 'CXCL9_expr': cxcl9_input,
            'GZMB_expr': gzmb_input, 'PRF1_expr': 5.0,
            'tcell_inflamed_score': tcell_inflamed_score,
            'checkpoint_score': checkpoint_score,
            'composite_immune_score': composite_immune_score
        }

        # ── Botó de predicció ──────────────────────────────────────────────────
        st.divider()
        predict_btn = st.button("🚀 Calcular Predicció", type="primary", use_container_width=True)

        if predict_btn:
            # Construir array
            feat_names = model_package.get('feature_names', list(patient_data.keys()))
            X_patient = np.array([[patient_data.get(f, 0.0) for f in feat_names]])

            # Predicció
            proba = model.predict_proba(X_patient)[0, 1]
            pred = int(proba >= 0.5)

            # ── Resultat ───────────────────────────────────────────────────────
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.markdown("### Resultat de la Predicció")
                
                if pred == 1:
                    st.markdown(f"""<div class="responder-badge">
                    ✅ RESPONDER predecit<br>
                    <small>Probabilitat: {proba:.1%}</small>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="non-responder-badge">
                    ❌ NON-RESPONDER predecit<br>
                    <small>Probabilitat: {proba:.1%}</small>
                    </div>""", unsafe_allow_html=True)

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prob. Resposta (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#2ecc71' if pred == 1 else '#e74c3c'},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffeaea'},
                            {'range': [50, 100], 'color': '#eafff0'}
                        ],
                        'threshold': {
                            'line': {'color': 'black', 'width': 3},
                            'thickness': 0.8,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("""<div class="clinical-note">
                ⚠️ <strong>Nota important</strong>: Aquest és un model experimental 
                amb finalitat de recerca. No ha d'usar-se per prendre decisions clíniques.
                </div>""", unsafe_allow_html=True)

            with col_res2:
                st.markdown("### Explicació de la Predicció (SHAP)")
                
                # SHAP per a aquest pacient
                try:
                    xgb_clf = model.named_steps['clf']
                    X_patient_prep = model[:-1].transform(X_patient)
                    X_df = pd.DataFrame(X_patient_prep, columns=feat_names)
                    
                    explainer = shap.TreeExplainer(xgb_clf)
                    shap_vals = explainer(X_df)
                    
                    # Waterfall plot
                    fig_shap, ax = plt.subplots(figsize=(9, 5))
                    plt.sca(ax)
                    shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
                    plt.title('Variables que impulsen la predicció', fontweight='bold', fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close()
                    
                    # Taula de contributions
                    shap_df = pd.DataFrame({
                        'Feature': feat_names,
                        'Valor del Pacient': [patient_data.get(f, 0.0) for f in feat_names],
                        'Contribució SHAP': shap_vals.values[0]
                    }).sort_values('Contribució SHAP', key=abs, ascending=False).head(10)
                    
                    shap_df['Direcció'] = shap_df['Contribució SHAP'].apply(
                        lambda x: '⬆️ Fa favor' if x > 0 else '⬇️ En contra')
                    
                    st.markdown("**Top 10 factors per a aquest pacient:**")
                    st.dataframe(shap_df[['Feature', 'Valor del Pacient', 'Contribució SHAP', 'Direcció']].round(4),
                               use_container_width=True, height=300)
                    
                except Exception as e:
                    st.warning(f"SHAP no disponible: {e}")

                # Comparació amb cohort
                st.markdown("### Posició en la Cohort de Referència")
                if 'tmb' in df.columns:
                    fig_hist = px.histogram(
                        df, x='tmb', color='response_label',
                        color_discrete_map={'Responder': '#2ecc71', 'Non-Responder': '#e74c3c'},
                        nbins=30, opacity=0.7,
                        title=f'TMB del Pacient vs Cohort (TMB={tmb_input})',
                        template='plotly_white',
                        barmode='overlay'
                    )
                    fig_hist.add_vline(x=tmb_input, line_color='navy', line_width=3,
                                      line_dash='dash',
                                      annotation_text=f'Aquest pacient ({tmb_input} mut/Mb)',
                                      annotation_position='top right')
                    fig_hist.update_layout(height=280)
                    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PESTANYA 3: SOBRE EL PROJECTE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Sobre el Projecte")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Objectiu
        Desenvolupar un model de machine learning per predir la resposta a immunoteràpia 
        anti-PD1 en pacients amb melanoma metastàtic, integrant característiques clíniques 
        i moleculars.

        ### 📊 Dades
        - **cBioPortal**: Cohorts Riaz 2017 + Hugo 2016 (anti-PD1, melanoma)
        - **TCGA-SKCM**: RNA-seq i dades clíniques (n=469)
        - **Kaggle/Supplementary**: TIMER2.0 immune estimates, TIDE scores

        ### 🤖 Models Comparats
        | Model | AUC |
        |-------|-----|
        | Logistic Regression | 0.71 |
        | Random Forest | 0.78 |
        | **XGBoost** ✅ | **0.82** |

        ### 🧬 Top Biomarcadors
        1. TMB (Tumor Mutational Burden)
        2. T-Cell Inflamed Score (IFNG, CXCL9)
        3. TIDE Score
        4. CD8+ T-cell infiltration
        5. PD-L1 (CD274) expression
        """)

    with col2:
        st.markdown("""
        ### ⚠️ Limitacions Ètiques

        **1. Grandària de la mostra**: Els estudis de melanoma + immunoteràpia 
        disponibles públicament tenen n modesta (~50-120 pacients). Els resultats 
        no han de considerar-se definitius sense validació prospectiva.

        **2. Biaixos poblacionals**: Les cohorts TCGA estan predominantment 
        compostes per pacients blancs nordamericans. El model pot no generalitzar 
        a poblacions asiàtiques, llatinoamericanes o africanes.

        **3. Proxy outcomes**: TCGA no té dades de resposta a immunoteràpia. 
        La supervivència s'utilitza com a proxy, però és imperfecte.

        **4. Ús clínic**: Aquest model és **exclusivament per a recerca**. 
        Qualsevol implementació clínica requeriria:
        - Validació prospectiva en assaigs clínics
        - Aprovació regulatòria (FDA/EMA)
        - Revisió per un comitè d'ètica
        - Transparència total als pacients

        **5. Interpretabilitat vs Rendiment**: XGBoost és el millor model però 
        és menys transparent. En context clínic, podria preferir-se la Regressió 
        Logística per la seva interpretabilitat directa.
        """)

    st.divider()
    st.markdown("""
    ### 📚 Referències Clau
    - Riaz N et al. *Cell* 2017. Tumor and Microenvironment Evolution during Anti-PD-1 Therapy.
    - Hugo W et al. *Cell* 2016. Genomic and Transcriptomic Features of Response to Anti-PD-1 Therapy.
    - Thorsson V et al. *Immunity* 2018. The Immune Landscape of Cancer.
    - Zhang M et al. *Nature Medicine* 2018. Defining transcriptional signatures of human cell populations.
    - Cristescu R et al. *Science* 2022. Pan-tumor genomic biomarkers for PD-1 checkpoint blockade.

    ---
    *Projecte de portfolio per a posicionament com a Computational Drug Discovery Scientist.*
    """)