# Agente ADK — Previsão de IDEB de Escolas Brasileiras

Agente construído em **Google ADK (Agent Development Kit)** que usa um modelo
clássico de Machine Learning (Random Forest) para prever o **IDEB — Índice
de Desenvolvimento da Educação Básica** de escolas brasileiras, integrado
com a API pública do IBGE para enriquecimento de dados.

**Tema do desafio:** Educação
**Tipo de modelo clássico:** Regressão (RandomForestRegressor)
**Fonte de dados pública integrada em tempo real:** API IBGE Localidades
(`servicodados.ibge.gov.br`)

---

## Arquitetura

```
desafio-ml-adk/
├── README.md                    ← este arquivo
├── requirements.txt             ← dependências Python
├── .env.example                 ← template para variáveis de ambiente
│
├── data/
│   ├── generate_dataset.py      ← gera dataset sintético (estrutura INEP)
│   └── ideb_schools.csv         ← dataset gerado (4000 escolas)
│
├── ml/
│   ├── train.py                 ← treina RandomForestRegressor
│   ├── ideb_model.joblib        ← modelo treinado (entregável opcional)
│   ├── feature_info.json        ← metadados das features
│   └── metrics.json             ← métricas de avaliação
│
└── edu_agent/                   ← pacote ADK (descoberto pelo `adk` CLI)
    ├── __init__.py
    ├── agent.py                 ← root_agent
    └── tools.py                 ← ferramentas do agente
```

### Fluxo de execução

```
usuário ──► ADK runtime ──► root_agent (Gemini 2.5 Flash)
                                │
                                ├── prever_ideb ──► RandomForest joblib
                                ├── explicar_fatores_ideb ──► feature_info.json
                                ├── estatisticas_regiao ──► dataset CSV
                                ├── buscar_municipio_ibge ──► API IBGE (HTTPS)
                                └── listar_campos_esperados ──► metadados
```

---

## Pré-requisitos

- Python 3.10+
- Chave de API do Google Gemini (grátis em https://aistudio.google.com/apikey)

---

## Setup

### 1. Clonar / extrair o projeto

```bash
cd desafio-ml-adk
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# ou
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha sua chave da API:

```bash
cp .env.example edu_agent/.env
```

Edite `edu_agent/.env` e defina:

```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=sua_chave_do_google_ai_studio_aqui
```

> O ADK busca o `.env` dentro da pasta do agente (`edu_agent/.env`).

### 5. Gerar o dataset e treinar o modelo

Se você recebeu o projeto **sem** os arquivos `data/ideb_schools.csv` e
`ml/ideb_model.joblib`, rode:

```bash
python data/generate_dataset.py
python ml/train.py
```

Isso produz:

- `data/ideb_schools.csv` (4000 escolas sintéticas com estrutura do Censo Escolar)
- `ml/ideb_model.joblib` (modelo Random Forest treinado)
- `ml/feature_info.json` (metadados das features)
- `ml/metrics.json` (R², MAE, RMSE)

Métricas típicas obtidas:

| Métrica   | Treino | Teste  |
|-----------|--------|--------|
| R²        | 0,93   | 0,76   |
| MAE       | 0,16   | 0,29   |
| RMSE (teste) | —   | 0,37   |

---

## Como executar o agente

### Opção A — Interface web do ADK (recomendada para demo)

A partir do diretório **raiz do projeto** (`desafio-ml-adk/`):

```bash
adk web
```

Abra http://localhost:8000 no navegador, selecione `edu_agent` no menu
lateral e converse.

### Opção B — CLI interativa

```bash
adk run edu_agent
```

### Opção C — Como servidor API (para integrar em outra aplicação)

```bash
adk api_server
```

---

## Exemplos de uso

Experimente estas perguntas com o agente:

1. **Previsão direta**
   > *"Uma escola municipal urbana no Nordeste (SE), anos iniciais, com 200 alunos, 12 professores, 70% com ensino superior, biblioteca e internet mas sem laboratórios, taxa de aprovação 85%, NSE 3.5. Qual IDEB previsto?"*

2. **Interpretabilidade**
   > *"Quais fatores mais influenciam o IDEB segundo o modelo?"*

3. **Benchmark regional**
   > *"Qual o IDEB médio das escolas municipais do Nordeste no dataset?"*

4. **Consulta a dado público real (IBGE)**
   > *"Consulta o código IBGE de Aracaju-SE e me diga a mesorregião."*

5. **Análise comparativa (o agente encadeia ferramentas)**
   > *"Simula duas escolas iguais, uma urbana e outra rural, no mesmo
   > município, e compara o IDEB previsto com a média da região."*

---

## O modelo de Machine Learning

- **Algoritmo:** `sklearn.ensemble.RandomForestRegressor`
- **Hiperparâmetros:** 300 árvores, profundidade máxima 15, mínimo 3
  amostras por folha
- **Pré-processamento:** `OneHotEncoder` para variáveis categóricas
  (região, UF, dependência administrativa, localização, etapa) +
  `StandardScaler` para variáveis numéricas
- **Target:** IDEB (0 a 10)
- **18 features** cobrindo contexto geográfico, infraestrutura física,
  perfil do corpo docente, taxa de aprovação e nível socioeconômico

É um modelo **clássico / não-generativo**, conforme exigido pelo desafio.

### Top features por importância

Consistente com a literatura educacional brasileira (Soares & Andrade, INEP):

1. Nível socioeconômico do alunado (~60%)
2. Taxa de aprovação (~11%)
3. Localização (urbana/rural) (~11% combinando as duas categorias)
4. Percentual de professores com ensino superior (~4%)

---

## Fontes de dados públicas relevantes

O agente integra **em tempo real** com a API do IBGE. As demais fontes
abaixo foram usadas como referência para a estrutura do dataset e podem
ser usadas para substituir o dataset sintético por dados reais:

- **IBGE Localidades API** — `https://servicodados.ibge.gov.br/api/v1/localidades/municipios`
  *(usada online pelo agente)*
- **INEP — Microdados do Censo Escolar e IDEB** — https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos
- **Dados Abertos do MEC** — https://dadosabertos.mec.gov.br/
- **Portal de Dados Abertos Federal** — https://dados.gov.br/
- **Catálogo de APIs do Governo Federal** — https://www.gov.br/conecta/catalogo/

---

## Substituindo pelo dataset real do INEP (opcional)

Para usar microdados oficiais do IDEB:

1. Baixe a planilha em https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/ideb
2. Normalize os nomes de colunas para o schema em `data/ideb_schools.csv`
3. Substitua o CSV e rode `python ml/train.py` novamente

As features do dataset sintético foram escolhidas para serem
diretamente mapeáveis para colunas do Censo Escolar INEP.

---

## Testando as tools isoladamente (sem chave Gemini)

Útil para desenvolvimento:

```bash
python -c "
from edu_agent.tools import prever_ideb
print(prever_ideb(
    regiao='Nordeste', uf='SE', dependencia_administrativa='Municipal',
    localizacao='Urbana', etapa_ensino='Anos Iniciais',
    numero_alunos=200, numero_professores=12,
    pct_prof_ensino_superior=0.7,
    tem_biblioteca=1, tem_lab_informatica=0, tem_lab_ciencias=0,
    tem_quadra_esportes=1, tem_internet=1, tem_agua_potavel=1,
    tem_energia_eletrica=1, tem_jornada_estendida=0,
    taxa_aprovacao=0.85, nivel_socioeconomico=3.5
))
"
```

---

## Entregáveis (conforme o desafio)

| Item | Localização |
|------|-------------|
| 1. Código-fonte do agente em ADK | `edu_agent/` |
| 2. README.md com instruções | `README.md` (este arquivo) |
| 3. Modelo treinado (opcional) | `ml/ideb_model.joblib` |

---

## Troubleshooting

**`adk: command not found`**
→ Verifique se instalou `google-adk` e se o virtualenv está ativo.

**`FileNotFoundError: ideb_model.joblib`**
→ Rode `python ml/train.py` antes de iniciar o agente.

**`API key not valid`**
→ Confira se `edu_agent/.env` existe e tem `GOOGLE_API_KEY=...` preenchido.

**Agente não encontra as tools**
→ Execute o `adk web` a partir do diretório **pai** de `edu_agent/`
(ou seja, da raiz `desafio-ml-adk/`).

---

## Licença

Projeto desenvolvido como resposta ao Desafio de ML. Dataset sintético
gerado internamente; modelo livre para uso em avaliação.
