# Agente Preditivo - Saude Mental dos Adolescentes

Projeto da disciplina de Inteligencia Artificial e Sistemas Inteligentes  
UNOESC Chapeco  
Professor: Jacson Luiz Matte  
Alunos: Fabiano Costa e Leonardo Felipe Zani

---

## Sobre o Projeto

O objetivo e construir um agente preditivo capaz de estimar o risco de depressao em adolescentes com base no uso de redes sociais e indicadores de saude. O modelo de machine learning realiza a predicao e um agente inteligente (Google Gemini) explica o resultado em linguagem natural.

Base de dados: Social Media & Mental Health (Kaggle) - 1.200 registros, 13 variaveis  
Variavel alvo: depression_label (0 = sem depressao / 1 = com depressao)  
Melhor modelo: Naive Bayes Gaussiano (AUC-ROC = 0.9922)

---

## Estrutura do Repositorio

```
agente-preditivo/
    data/
        dataset.csv
    ml/
        train_models.py
    backend/
        app.py
    frontend/
        app_streamlit.py
    models/
        best_model.pkl
        scaler.pkl
        le_gender.pkl
        le_platform.pkl
        le_social.pkl
        model_meta.json
    plots/
        correlacao.png
        boxplot.png
        frequencia.png
        comparacao_modelos.png
    requirements.txt
    README.md
```

---

## Como Rodar

### Requisitos

- Python 3.10 ou superior
- Chave de API do Google Gemini (instrucoes abaixo)

### 1. Clone o repositorio

```bash
git clone https://github.com/SEU_USUARIO/agente-preditivo.git
cd agente-preditivo
```

### 2. Instale as dependencias

```bash
pip install -r requirements.txt
```

### 3. Treine os modelos

```bash
python ml/train_models.py
```

Isso gera os arquivos em models/ e os graficos em plots/.

### 4. Configure a chave do Gemini

Renomeie o arquivo .env.example para .env e substitua o valor pela chave fornecida junto com este projeto:

```
GEMINI_API_KEY=sua_chave_aqui
```

O backend carrega esse arquivo automaticamente ao iniciar.

### 5. Inicie o backend

```bash
python backend/app.py
```

Servidor disponivel em http://localhost:5000

### 6. Inicie o frontend

Em outro terminal:

```bash
streamlit run frontend/app_streamlit.py
```

Interface disponivel em http://localhost:8501

---

## Endpoints da API

| Metodo | Rota      | Descricao                            |
|--------|-----------|--------------------------------------|
| GET    | /health   | Status do servidor e modelo carregado |
| POST   | /predict  | Predicao + explicacao via Gemini     |
| GET    | /opcoes   | Valores validos para campos categoricos |

Exemplo de body para POST /predict:

```json
{
  "age": 16,
  "gender": "female",
  "daily_social_media_hours": 7.5,
  "platform_usage": "Instagram",
  "sleep_hours": 5.5,
  "screen_time_before_sleep": 3.0,
  "academic_performance": 2.5,
  "physical_activity": 0.5,
  "social_interaction_level": "low",
  "stress_level": 9,
  "anxiety_level": 8,
  "addiction_level": 9
}
```

---

## Resultados dos Modelos

| Algoritmo            | Acuracia | Sensibilidade | Especificidade | Precisao | AUC-ROC |
|----------------------|----------|---------------|----------------|----------|---------|
| Regressao Logistica  | 0.9625   | 0.6667        | 0.9701         | 0.3636   | 0.9829  |
| KNN (k=7)            | 0.8958   | 0.6667        | 0.9017         | 0.1481   | 0.9113  |
| MLP (64-32 neuronios)| 0.9625   | 0.5000        | 0.9744         | 0.3333   | 0.9651  |
| Naive Bayes          | 0.9750   | 0.3333        | 0.9915         | 0.5000   | 0.9922  |

Criterio de selecao: AUC-ROC, por ser mais adequado para dados desbalanceados.  
Balanceamento: SMOTE aplicado no conjunto de treino (31 positivos expandidos para 935).

---

## Diario de Bordo de Contribuicoes

### Fabiano Costa

**Dias 1 a 5**
Selecao e exploracao inicial da base de dados no Kaggle. Analise das variaveis, identificacao do desbalanceamento da classe alvo (31 positivos em 1.200 registros) e definicao da estrategia de balanceamento com SMOTE. Geracao dos graficos exploratOrios com Seaborn: mapa de correlacao, boxplot por diagnostico e grafico de frequencia das variaveis categoricas.

**Dias 6 a 10**
Implementacao do pipeline de pre-processamento (codificacao de variaveis categoricas com LabelEncoder, normalizacao com StandardScaler). Treinamento dos quatro algoritmos: Regressao Logistica, KNN, MLP e Naive Bayes. Calculo das metricas de avaliacao e selecao do melhor modelo com base no AUC-ROC.

**Dias 11 a 15**
Exportacao do modelo Naive Bayes como arquivo .pkl. Geracao do grafico comparativo de metricas. Revisao do codigo e documentacao do script de treinamento.

---

### Leonardo Felipe Zani

**Dias 1 a 5**
Estudo da API do Google Gemini e definicao do System Prompt do agente, com foco em respostas empaticas, fundamentadas e sem alucinacoes. Prototipagem da arquitetura backend/frontend.

**Dias 6 a 10**
Desenvolvimento do servidor Flask com os endpoints /health, /predict e /opcoes. Integracao dos artefatos de ML com a API. Desenvolvimento da interface Streamlit com formulario de entrada, exibicao do resultado do modelo e da explicacao gerada pelo agente.

**Dias 11 a 15**
Integracao do backend com o Gemini 1.5 Flash. Tratamento de erros de conexao e validacao de inputs no frontend. Testes com diferentes combinacoes de dados e ajustes finais de layout. Organizacao do repositorio e escrita do README.

---

## Aviso

Este sistema tem fins academicos. Os resultados nao constituem diagnostico medico. Em caso de preocupacoes com saude mental, consulte um profissional qualificado.
