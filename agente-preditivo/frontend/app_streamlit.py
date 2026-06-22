"""
Etapa C — Interface Web (Streamlit)
Agente Preditivo: Impacto das Redes Sociais na Saúde Mental dos Adolescentes
"""

import streamlit as st
import requests
import json

# ─── Configuração da página ─────────────────────────────
st.set_page_config(
    page_title="Agente Preditivo — Saúde Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:5000"

# ─── CSS personalizado ──────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e2e8f0; font-size: 2rem; margin: 0; }
    .main-header p  { color: #94a3b8; margin: 0.4rem 0 0; }

    .result-positive {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 5px solid #ef4444;
        padding: 1.2rem 1.5rem; border-radius: 10px; color: white;
    }
    .result-negative {
        background: linear-gradient(135deg, #14532d, #166534);
        border-left: 5px solid #22c55e;
        padding: 1.2rem 1.5rem; border-radius: 10px; color: white;
    }
    .metric-card {
        background: #1e293b; border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid #334155;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    .explanation-box {
        background: #0f172a; border: 1px solid #1e3a5f;
        border-radius: 10px; padding: 1.5rem;
        line-height: 1.7; color: #e2e8f0;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #60a5fa;
        border-bottom: 1px solid #1e3a5f; padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cabeçalho ──────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Agente Preditivo de Saúde Mental</h1>
    <p>Impacto das Redes Sociais na Saúde Mental dos Adolescentes</p>
    <p style="font-size:0.8rem; color:#64748b;">Modelo: Naive Bayes · AUC-ROC: 0.9922 · Base: Kaggle (1.200 registros)</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar — informações ───────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ Sobre o Agente")
    st.info(
        "Este agente combina **Machine Learning** (Naive Bayes) com "
        "**IA Generativa** (Google Gemini) para prever o risco de "
        "depressão em adolescentes e explicar o resultado em linguagem natural."
    )
    st.markdown("### 📊 Desempenho do Modelo")
    st.metric("AUC-ROC",        "0.9922")
    st.metric("Acurácia",       "97.50%")
    st.metric("Especificidade", "99.15%")
    st.metric("Sensibilidade",  "33.33%")
    st.caption("⚠️ Modelo balanceado via SMOTE (classe positiva muito rara)")
    st.markdown("---")
    st.markdown("### 🔗 Repositório")
    st.markdown("[GitHub — Ver código-fonte](#)", unsafe_allow_html=True)

# ─── Formulário de entrada ───────────────────────────────
# Buscar opções válidas do backend
try:
    opcoes_resp = requests.get(f"{BACKEND_URL}/opcoes", timeout=5)
    opcoes = opcoes_resp.json()
    backend_ok = True
except Exception:
    opcoes = {
        "gender":                   ["female", "male"],
        "platform_usage":           ["Both", "Instagram", "TikTok"],
        "social_interaction_level": ["high", "low", "medium"],
    }
    backend_ok = False

if not backend_ok:
    st.warning("⚠️ Backend não encontrado. Inicie com `python backend/app.py`")

st.markdown('<p class="section-title">📋 Dados do Adolescente</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Perfil**")
    age    = st.slider("Idade",  min_value=13, max_value=19, value=16)
    gender = st.selectbox("Gênero", opcoes["gender"],
                          format_func=lambda x: "Feminino" if x == "female" else "Masculino")
    platform = st.selectbox("Plataforma Principal", opcoes["platform_usage"],
                             format_func=lambda x: {"Both": "Ambas", "Instagram": "Instagram",
                                                     "TikTok": "TikTok"}.get(x, x))

with col2:
    st.markdown("**Uso Digital & Sono**")
    horas_sm    = st.slider("Horas/dia em Redes Sociais", 1.0, 8.0, 4.5, 0.1)
    sleep       = st.slider("Horas de Sono",              4.0, 10.0, 7.0, 0.1)
    screen_bed  = st.slider("Tela antes de dormir (h)",   0.0, 4.0,  1.5, 0.1)
    social_int  = st.selectbox("Interação Social",
                                opcoes["social_interaction_level"],
                                format_func=lambda x: {"high": "Alta", "medium": "Média",
                                                        "low": "Baixa"}.get(x, x))

with col3:
    st.markdown("**Indicadores de Saúde**")
    acad_perf   = st.slider("Desempenho Acadêmico (0–5)", 0.0, 5.0, 3.0, 0.01)
    phys_act    = st.slider("Atividade Física (h/sem)",   0.0, 5.0, 1.5, 0.1)
    stress      = st.slider("Nível de Estresse (0–10)",   0,   10,  5)
    anxiety     = st.slider("Nível de Ansiedade (0–10)",  0,   10,  4)
    addiction   = st.slider("Dependência Digital (0–10)", 0,   10,  5)

st.markdown("---")
predict_btn = st.button("🔍 Analisar e Gerar Explicação", type="primary", use_container_width=True)

# ─── Predição ───────────────────────────────────────────
if predict_btn:
    payload = {
        "age":                      age,
        "gender":                   gender,
        "daily_social_media_hours": horas_sm,
        "platform_usage":           platform,
        "sleep_hours":              sleep,
        "screen_time_before_sleep": screen_bed,
        "academic_performance":     acad_perf,
        "physical_activity":        phys_act,
        "social_interaction_level": social_int,
        "stress_level":             stress,
        "anxiety_level":            anxiety,
        "addiction_level":          addiction,
    }

    with st.spinner("🤖 Consultando modelo e gerando explicação..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
            resultado = resp.json()

            if "erro" in resultado:
                st.error(f"Erro na predição: {resultado['erro']}")
            else:
                predicao  = resultado["predicao"]
                prob      = resultado["probabilidade"]
                label     = resultado["label"]
                explicacao = resultado["explicacao"]

                st.markdown("---")
                st.markdown('<p class="section-title">📊 Resultado do Modelo</p>',
                            unsafe_allow_html=True)

                # Cards de métricas
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">{"🔴" if predicao == 1 else "🟢"}</div>
                        <div class="label">{label}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">{prob:.1%}</div>
                        <div class="label">Prob. de Depressão</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">0.99</div>
                        <div class="label">AUC-ROC do Modelo</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value">NB</div>
                        <div class="label">Naive Bayes</div>
                    </div>""", unsafe_allow_html=True)

                # Banner de resultado
                st.markdown("<br>", unsafe_allow_html=True)
                if predicao == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <b>⚠️ Resultado: {label}</b><br>
                        O modelo identificou indicadores compatíveis com risco de depressão.
                        Probabilidade calculada: <b>{prob:.1%}</b>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <b>✅ Resultado: {label}</b><br>
                        O modelo não identificou indicadores significativos de depressão.
                        Probabilidade calculada: <b>{prob:.1%}</b>
                    </div>""", unsafe_allow_html=True)

                # Explicação do agente
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-title">🤖 Explicação do Agente Inteligente (Gemini)</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="explanation-box">{explicacao}</div>',
                            unsafe_allow_html=True)

                # Dados enviados (expansível)
                with st.expander("🔎 Ver dados enviados ao modelo"):
                    st.json(payload)

        except requests.exceptions.ConnectionError:
            st.error("❌ Não foi possível conectar ao backend. Verifique se o servidor Flask está rodando.")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

# ─── Rodapé ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Aviso:** Este sistema tem fins acadêmicos e não substitui avaliação médica profissional. "
    "Em caso de dúvidas sobre saúde mental, consulte um psicólogo ou psiquiatra."
)
