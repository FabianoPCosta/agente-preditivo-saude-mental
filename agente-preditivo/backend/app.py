"""
Etapa B — Backend Flask + Agente Inteligente (Gemini API)
"""

import os
import json
import joblib
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # carrega o arquivo .env automaticamente

app = Flask(__name__)
CORS(app)

# ─── Carregamento dos artefatos de ML ───────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

best_model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le_gender   = joblib.load(os.path.join(MODEL_DIR, "le_gender.pkl"))
le_platform = joblib.load(os.path.join(MODEL_DIR, "le_platform.pkl"))
le_social   = joblib.load(os.path.join(MODEL_DIR, "le_social.pkl"))

with open(os.path.join(MODEL_DIR, "model_meta.json"), encoding="utf-8") as f:
    model_meta = json.load(f)

FEATURES = model_meta["features"]

# ─── Configuração do Gemini ─────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

SYSTEM_PROMPT = """Você é um especialista em saúde mental adolescente e ciência de dados.
Sua tarefa é interpretar o resultado de um modelo preditivo de Machine Learning
(Naive Bayes, AUC-ROC = 0.99) que avalia o risco de depressão em adolescentes
com base no uso de redes sociais e hábitos de vida.

Regras obrigatórias:
1. Apresente o resultado de forma clara, empática e sem jargões técnicos excessivos.
2. Explique brevemente quais variáveis mais contribuem para o resultado, com base nos dados fornecidos.
3. Se o risco for POSITIVO (depressão detectada): forneça 3 recomendações práticas e incentive buscar apoio profissional.
4. Se o risco for NEGATIVO (sem depressão): parabenize os hábitos saudáveis e dê 2 dicas preventivas.
5. NUNCA invente dados, diagnósticos médicos ou afirme certezas além do que o modelo indica.
6. Finalize com um aviso: "Este resultado é orientativo e não substitui avaliação médica profissional."
7. Responda sempre em português brasileiro, com linguagem acolhedora e positiva.
"""

def preprocessar(dados: dict) -> np.ndarray:
    """Recebe dict com inputs do usuário e retorna array pronto para predição."""
    gender_enc   = int(le_gender.transform([dados["gender"]])[0])
    platform_enc = int(le_platform.transform([dados["platform_usage"]])[0])
    social_enc   = int(le_social.transform([dados["social_interaction_level"]])[0])

    vetor = [
        float(dados["age"]),
        float(dados["daily_social_media_hours"]),
        float(dados["sleep_hours"]),
        float(dados["screen_time_before_sleep"]),
        float(dados["academic_performance"]),
        float(dados["physical_activity"]),
        float(dados["stress_level"]),
        float(dados["anxiety_level"]),
        float(dados["addiction_level"]),
        gender_enc,
        platform_enc,
        social_enc,
    ]
    return np.array(vetor).reshape(1, -1)


def gerar_explicacao(dados: dict, predicao: int, probabilidade: float) -> str:
    """Chama Gemini para explicar o resultado em linguagem natural."""
    if gemini_model is None:
        return (
            "⚠️ API do Gemini não configurada. Configure a variável de ambiente "
            "GEMINI_API_KEY para ativar as explicações do agente inteligente."
        )

    label_texto = "POSITIVO — risco de depressão identificado" if predicao == 1 \
                  else "NEGATIVO — sem indicativo de depressão"

    prompt_usuario = f"""
Analise os seguintes dados de um adolescente e o resultado do modelo preditivo:

DADOS DO ADOLESCENTE:
- Idade: {dados['age']} anos
- Gênero: {dados['gender']}
- Horas diárias em redes sociais: {dados['daily_social_media_hours']}h
- Plataforma principal: {dados['platform_usage']}
- Horas de sono por noite: {dados['sleep_hours']}h
- Tempo de tela antes de dormir: {dados['screen_time_before_sleep']}h
- Desempenho acadêmico (0–5): {dados['academic_performance']}
- Atividade física (horas/semana): {dados['physical_activity']}
- Nível de interação social: {dados['social_interaction_level']}
- Nível de estresse (0–10): {dados['stress_level']}
- Nível de ansiedade (0–10): {dados['anxiety_level']}
- Nível de dependência digital (0–10): {dados['addiction_level']}

RESULTADO DO MODELO (Naive Bayes, AUC-ROC = 0.99):
- Classificação: {label_texto}
- Probabilidade de depressão: {probabilidade:.1%}

Forneça uma explicação completa seguindo as regras do seu sistema.
"""

    try:
        chat = gemini_model.start_chat(history=[])
        # Injeta o system prompt como primeira mensagem de contexto
        resposta = chat.send_message(
            f"{SYSTEM_PROMPT}\n\n{prompt_usuario}"
        )
        return resposta.text
    except Exception as e:
        return f"Erro ao consultar o agente inteligente: {str(e)}"


# ─── ROTAS ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modelo": model_meta["nome"],
        "auc_roc": model_meta["auc"],
        "gemini_configurado": gemini_model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json(force=True)

    # Validação dos campos obrigatórios
    campos = [
        "age", "gender", "daily_social_media_hours", "platform_usage",
        "sleep_hours", "screen_time_before_sleep", "academic_performance",
        "physical_activity", "social_interaction_level",
        "stress_level", "anxiety_level", "addiction_level"
    ]
    faltando = [c for c in campos if c not in dados]
    if faltando:
        return jsonify({"erro": f"Campos faltando: {faltando}"}), 400

    try:
        X = preprocessar(dados)
        X_sc = scaler.transform(X)

        predicao     = int(best_model.predict(X_sc)[0])
        probabilidade = float(best_model.predict_proba(X_sc)[0][1])

        explicacao = gerar_explicacao(dados, predicao, probabilidade)

        return jsonify({
            "predicao":      predicao,
            "label":         model_meta["classes"][str(predicao)],
            "probabilidade": round(probabilidade, 4),
            "explicacao":    explicacao,
            "modelo_info": {
                "nome":        model_meta["nome"],
                "auc_roc":     model_meta["auc"],
                "acuracia":    model_meta["acuracia"],
                "sensibilidade": model_meta["sensibilidade"],
            }
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/opcoes", methods=["GET"])
def opcoes():
    """Retorna as opções válidas para campos categóricos."""
    return jsonify({
        "gender":                   list(le_gender.classes_),
        "platform_usage":           list(le_platform.classes_),
        "social_interaction_level": list(le_social.classes_),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Backend rodando em http://localhost:{port}")
    print(f"   Modelo: {model_meta['nome']}  |  AUC-ROC: {model_meta['auc']}")
    print(f"   Gemini: {'✓ configurado' if gemini_model else '✗ sem API key'}")
    app.run(debug=True, host="0.0.0.0", port=port)
