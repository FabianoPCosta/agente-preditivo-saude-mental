"""
Etapa A — Preparação, Modelagem e Avaliação de Modelos
Base: Impacto das Redes Sociais na Saúde Mental dos Adolescentes
Target: depression_label (0 = Sem depressão, 1 = Com depressão)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"Distribuição do target:\n{df['depression_label'].value_counts()}\n")

# Codificação de variáveis categóricas
le_gender   = LabelEncoder()
le_platform = LabelEncoder()
le_social   = LabelEncoder()

df["gender_enc"]   = le_gender.fit_transform(df["gender"])
df["platform_enc"] = le_platform.fit_transform(df["platform_usage"])
df["social_enc"]   = le_social.fit_transform(df["social_interaction_level"])

FEATURES = [
    "age", "daily_social_media_hours", "sleep_hours",
    "screen_time_before_sleep", "academic_performance",
    "physical_activity", "stress_level", "anxiety_level",
    "addiction_level", "gender_enc", "platform_enc", "social_enc"
]

X = df[FEATURES]
y = df["depression_label"]

# Salvar encoders para uso no backend
joblib.dump(le_gender,   os.path.join(MODEL_DIR, "le_gender.pkl"))
joblib.dump(le_platform, os.path.join(MODEL_DIR, "le_platform.pkl"))
joblib.dump(le_social,   os.path.join(MODEL_DIR, "le_social.pkl"))

# Train/Test split (80/20) estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE para balancear classes (1169 vs 31)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Após SMOTE — treino: {pd.Series(y_train_res).value_counts().to_dict()}")

# Normalização
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ─────────────────────────────────────────────
# 2. GRÁFICOS EXPLORATÓRIOS (Seaborn)
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

## 2a. Mapa de Correlação
plt.figure(figsize=(12, 9))
numeric_cols = df[FEATURES + ["depression_label"]].copy()
corr = numeric_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, cbar_kws={"shrink": .8},
    annot_kws={"size": 8}
)
plt.title("Mapa de Correlação entre Variáveis", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlacao.png"), dpi=150)
plt.close()
print("✓ Gráfico de correlação salvo")

## 2b. Box Plot — variáveis-chave por label
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
box_vars = [
    ("daily_social_media_hours", "Horas/dia em Redes Sociais"),
    ("sleep_hours",              "Horas de Sono"),
    ("stress_level",             "Nível de Estresse"),
    ("anxiety_level",            "Nível de Ansiedade"),
    ("addiction_level",          "Nível de Dependência"),
    ("academic_performance",     "Desempenho Acadêmico"),
]
for ax, (col, label) in zip(axes.flat, box_vars):
    sns.boxplot(
        data=df, x="depression_label", y=col, ax=ax,
        palette=["#5B8DB8", "#E07B54"],
        width=0.5, linewidth=1.2
    )
    ax.set_xlabel("Depressão (0=Não, 1=Sim)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
plt.suptitle("Box Plot: Variáveis por Diagnóstico de Depressão",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Box plot salvo")

## 2c. Gráfico de Frequência
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cat_vars = [
    ("gender",                   "Gênero"),
    ("platform_usage",           "Plataforma"),
    ("social_interaction_level", "Interação Social"),
]
for ax, (col, label) in zip(axes, cat_vars):
    order = df[col].value_counts().index
    sns.countplot(
        data=df, x=col, hue="depression_label", ax=ax,
        order=order, palette={0: "#5B8DB8", 1: "#E07B54"}
    )
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Depressão", labels=["Não", "Sim"], fontsize=8)
plt.suptitle("Frequência de Variáveis Categóricas por Diagnóstico",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "frequencia.png"), dpi=150)
plt.close()
print("✓ Gráfico de frequência salvo\n")

# ─────────────────────────────────────────────
# 3. TREINAMENTO DOS MODELOS
# ─────────────────────────────────────────────
def metricas(nome, y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], 0, 0, cm[1,1])
    acc  = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0   # Sensibilidade / Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0   # Especificidade
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0   # Precisão
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    print(f"\n{'='*48}")
    print(f"  {nome}")
    print(f"{'='*48}")
    print(f"  Acurácia      : {acc:.4f}")
    print(f"  Sensibilidade : {sens:.4f}")
    print(f"  Especificidade: {spec:.4f}")
    print(f"  Precisão      : {prec:.4f}")
    if auc: print(f"  AUC-ROC       : {auc:.4f}")
    print(f"  Matriz de Confusão:\n{cm}")
    return {"nome": nome, "acuracia": acc, "sensibilidade": sens,
            "especificidade": spec, "precisao": prec, "auc": auc,
            "modelo": None}

resultados = []

# 3a. Regressão Logística (substitui Regressão Linear Múltipla para classificação)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sc, y_train_res)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
r = metricas("Regressão Logística (Linear)", y_test, y_pred_lr, y_prob_lr)
r["modelo"] = lr
resultados.append(r)

# 3b. KNN
knn = KNeighborsClassifier(n_neighbors=7, metric="minkowski")
knn.fit(X_train_sc, y_train_res)
y_pred_knn = knn.predict(X_test_sc)
y_prob_knn = knn.predict_proba(X_test_sc)[:, 1]
r = metricas("K-Nearest Neighbors (KNN)", y_test, y_pred_knn, y_prob_knn)
r["modelo"] = knn
resultados.append(r)

# 3c. MLP (Multi-Layer Perceptron)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu",
    max_iter=500, random_state=42, early_stopping=True
)
mlp.fit(X_train_sc, y_train_res)
y_pred_mlp = mlp.predict(X_test_sc)
y_prob_mlp = mlp.predict_proba(X_test_sc)[:, 1]
r = metricas("MLP (Multi-Layer Perceptron)", y_test, y_pred_mlp, y_prob_mlp)
r["modelo"] = mlp
resultados.append(r)

# 3d. Naive Bayes
nb = GaussianNB()
nb.fit(X_train_sc, y_train_res)
y_pred_nb = nb.predict(X_test_sc)
y_prob_nb = nb.predict_proba(X_test_sc)[:, 1]
r = metricas("Naive Bayes (Gaussiano)", y_test, y_pred_nb, y_prob_nb)
r["modelo"] = nb
resultados.append(r)

# ─────────────────────────────────────────────
# 4. COMPARAÇÃO E EXPORTAÇÃO DO MELHOR MODELO
# ─────────────────────────────────────────────
# Critério: AUC-ROC (mais adequado para dados desbalanceados)
melhor = max(resultados, key=lambda x: (x["auc"] or 0))
print(f"\n{'*'*48}")
print(f"  MELHOR MODELO: {melhor['nome']}")
print(f"  AUC-ROC: {melhor['auc']:.4f}")
print(f"{'*'*48}\n")

joblib.dump(melhor["modelo"], os.path.join(MODEL_DIR, "best_model.pkl"))
print(f"✓ Modelo exportado: models/best_model.pkl")

# Salvar metadados do melhor modelo
import json
meta = {
    "nome": melhor["nome"],
    "acuracia": round(melhor["acuracia"], 4),
    "sensibilidade": round(melhor["sensibilidade"], 4),
    "especificidade": round(melhor["especificidade"], 4),
    "precisao": round(melhor["precisao"], 4),
    "auc": round(melhor["auc"], 4) if melhor["auc"] else None,
    "features": FEATURES,
    "classes": {0: "Sem depressão", 1: "Com depressão"}
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("✓ Metadados salvos: models/model_meta.json")

# ─────────────────────────────────────────────
# 5. GRÁFICO COMPARATIVO DE MÉTRICAS
# ─────────────────────────────────────────────
nomes   = [r["nome"].split("(")[0].strip() for r in resultados]
metricas_cols = ["acuracia", "sensibilidade", "especificidade", "precisao"]
labels  = ["Acurácia", "Sensibilidade", "Especificidade", "Precisão"]
cores   = ["#5B8DB8", "#E07B54", "#6AAF6E", "#9B72B0"]

x = np.arange(len(nomes))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 6))
for i, (col, label, cor) in enumerate(zip(metricas_cols, labels, cores)):
    vals = [r[col] for r in resultados]
    bars = ax.bar(x + i * width, vals, width, label=label, color=cor, alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Comparação de Métricas entre Algoritmos", fontsize=13, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(nomes, fontsize=10)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "comparacao_modelos.png"), dpi=150)
plt.close()
print("✓ Gráfico comparativo salvo\n")
print("Treinamento concluído com sucesso!")
