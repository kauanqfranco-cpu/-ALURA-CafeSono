# ☕ Café e Qualidade do Sono: análise exploratória, DAG e modelos de classificação

Projeto de análise de dados que investiga a relação entre o consumo de café/cafeína e a qualidade do sono, combinando **raciocínio causal (DAG)**, **análise exploratória** e **modelos de machine learning** (Random Forest, Regressão Logística, Regressão Ordinal e XGBoost) com interpretação via **valores SHAP**.

> **Status:** em andamento. O notebook é um *checkpoint* de aprendizado de uma série de cursos da Alura, com conteúdo adicional de causalidade trazido da formação em Economia. A seção [Achados e limitações](#achados-e-limitações) documenta o que a validação do projeto revelou e o que está planejado para a próxima versão.

> **Aviso importante:** o dataset é **sintético** (gerado por código). Qualquer relação encontrada descreve o processo que gerou os dados, e não o mundo real. Os resultados não devem ser lidos como evidência sobre saúde ou sobre o efeito real da cafeína.

---

## Sumário

- [Objetivo](#objetivo)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Dados](#dados)
- [Metodologia](#metodologia)
- [Achados e limitações](#achados-e-limitações)
- [Próximos passos](#próximos-passos)
- [Como executar](#como-executar)
- [Tecnologias](#tecnologias)

---

## Objetivo

Avaliar o impacto do consumo de cafeína na qualidade do sono, considerando também outras variáveis (estresse, saúde, atividade física, idade, hábitos etc.), e discutir **o que cada técnica consegue ou não afirmar** sobre causalidade.

Etapas do estudo (conforme o notebook):

1. Análise do panorama geral (DAG)
2. Obtenção dos dados
3. Análise exploratória
4. Preparação dos dados para os modelos
5. Seleção e treinamento de modelos
6. Aperfeiçoamento (GridSearchCV)
7. Apresentação dos resultados
8. Deploy *(ainda não realizado)*

## Estrutura do repositório

```
.
├── cafeSono.ipynb                          # Notebook principal (análise completa)
├── synthetic_coffee_health_10000_in_.csv   # Dataset (10.000 registros)
├── DAG.jpeg                                # Grafo causal assumido no projeto
└── README.md
```

## Dados

Dataset do Kaggle (*Global Coffee Health*), **gerado sinteticamente**, com 10.000 observações e 16 colunas, fornecido pela escola no contexto do curso.

| Coluna | Descrição |
|---|---|
| `ID` | Identificador do registro (1–10000) |
| `Age` | Idade (18 a 80 anos) |
| `Gender` | Male, Female ou Other |
| `Country` | País de residência (20 países) |
| `Coffee_Intake` | Consumo diário de café, em xícaras (0 a 10) |
| `Caffeine_mg` | Cafeína estimada por dia, em mg (≈ 95 mg por xícara) |
| `Sleep_Hours` | Horas de sono |
| `Sleep_Quality` | Qualidade do sono: Poor, Fair, Good, Excellent |
| `BMI` | Índice de massa corporal |
| `Heart_Rate` | Frequência cardíaca |
| `Stress_Level` | Estresse: Low, Medium, High |
| `Physical_Activity_Hours` | Horas de atividade física |
| `Health_Issues` | Problemas de saúde: None, Mild, Moderate, Severe |
| `Occupation` | Office, Service, Student, Healthcare, Other |
| `Smoking` | Fumante (0/1) |
| `Alcohol_Consumption` | Consome álcool (0/1) |

> **Atenção ao carregar o CSV:** o valor `"None"` em `Health_Issues` é interpretado como `NaN` pelo pandas. No notebook isso é tratado com `fillna('Nenhum')`. Alternativamente, use `pd.read_csv(..., keep_default_na=False, na_values=[''])`.

## Metodologia

### 1. DAG (grafo causal)

O projeto parte de um grafo causal assumido, com as seguintes relações:

- Ocupação → Ingestão de Café → Cafeína → Qualidade do Sono
- Nível de Estresse → Qualidade do Sono
- Atividade Física → Qualidade do Sono
- Idade, Consumo de Cigarros, Consumo de Álcool, IMC → Problemas de Saúde → Frequência Cardíaca → Qualidade do Sono

![DAG](DAG.jpeg)

### 2. Análise exploratória

Distribuições, boxplots de horas de sono por categoria, proporções de estresse por hábitos e ocupação, mapa de correlações (paleta acessível a daltônicos) e avaliação do balanceamento das classes.

### 3. Pré-processamento

- `OrdinalEncoder` com ordem explícita para variáveis ordinais (qualidade do sono, estresse, problemas de saúde)
- Divisão treino/teste estratificada
- `StandardScaler` para variáveis contínuas
- Balanceamento de classes com SMOTENC dentro do pipeline

### 4. Modelagem

Quatro modelos, com busca de hiperparâmetros via `GridSearchCV` (métrica: F1 macro):

1. Random Forest
2. Regressão Logística
3. Regressão Ordinal (`mord`)
4. XGBoost

### 5. Interpretação

Valores SHAP por classe de qualidade do sono, para avaliar a contribuição de cada variável nas previsões.

---

## Achados e limitações

Uma validação detalhada do projeto (execução de diagnósticos sobre os dados e réplica da modelagem) levou às conclusões abaixo. Elas orientam a próxima versão e **substituem as conclusões preliminares do notebook** nos pontos em que divergem.

### O alvo é derivado de outras variáveis (vazamento de alvo)

| Relação verificada | Resultado |
|---|---|
| `Sleep_Quality` vs. corte de `Sleep_Hours` em 5, 6 e 8 h | 97,07% de acordo; as divergências restantes são empates nas fronteiras por arredondamento das horas |
| `Stress_Level` vs. `Sleep_Quality` | 100% determinístico (Poor→High, Fair→Medium, Good/Excellent→Low) |
| `Health_Issues = None` | Nunca ocorre com menos de 6 horas de sono |

**Consequências:**

- Os F1 = 1,00 para as classes Ruim e Moderada vêm do estresse, que já contém a resposta, e não de capacidade preditiva do modelo.
- Uma regra de uma linha (estresse → qualidade) atinge **macro-F1 = 0,723** e acurácia de 0,865 no mesmo teste. Os modelos do notebook (macro-F1 ≈ 0,75) pouco a superam.
- Os quatro modelos empatam em macro-F1; a diferença está só na troca entre as classes Boa e Excelente.

### Desempenho por conjunto de variáveis (validação cruzada, 5 folds)

| Conjunto de variáveis | Macro-F1 (faixa entre 4 modelos) | R² em horas de sono (OLS) |
|---|---|---|
| Original do notebook (inclui estresse e saúde) | 0,74–0,75 | 0,635 |
| Sem estresse (mantém saúde) | 0,42–0,48 | 0,388 |
| Somente variáveis exógenas | 0,19–0,25 | 0,03–0,04 |
| Somente café | 0,19–0,22 | 0,036 |

O nível de acaso para 4 classes é da ordem de 0,2. Com variáveis que não derivam do desfecho, nenhum dos modelos supera o acaso de forma relevante.

### Efeito do café nas horas de sono

Com `Horas_de_sono` como desfecho contínuo:

| Especificação | Coeficiente do café (h por xícara) | IC 95% |
|---|---|---|
| Bruto | −0,160 | [−0,177; −0,144] |
| + idade, FC, atividade física, saúde | −0,101 | [−0,114; −0,088] |
| + estresse | −0,061 | [−0,071; −0,051] |

O efeito estimado é de cerca de **−10 minutos de sono por xícara** neste dataset sintético. O encolhimento do coeficiente ao adicionar saúde e estresse **não indica efeito menor**: são variáveis que decorrem do próprio desfecho (viés de mediador/colisor). Há também uma distinção importante: o efeito é estatisticamente claro, mas o café explica só ~3,6% da variância das horas, por isso é um mau preditor individual.

### O DAG precisa de revisão

- Setas bidirecionais (feedback) não cabem em um DAG, que é acíclico por definição. Em um grafo, `↔` representaria causa comum não observada.
- No grafo atual não há *backdoor path* entre café e sono (Ocupação só afeta o café), então estresse, atividade física e FC seriam covariáveis de precisão, e não confundidoras.
- Arestas **sem suporte** nos dados: atividade física → sono, cigarro → saúde, álcool → saúde.
- Relação **ausente** no grafo: café → frequência cardíaca (ρ ≈ 0,056).
- Relação mais forte do dataset (saúde ↔ sono, ρ ≈ −0,63) não está representada, e o sentido estresse → sono é incompatível com a derivação determinística descrita acima.

### Problemas técnicos conhecidos no notebook

- O bloco "Regressão Ordinal" usa `XGBClassifier` no pipeline, então o relatório rotulado como ordinal é na verdade de um XGBoost.
- Métricas manuais (`.score`) avaliam modelos treinados com dados padronizados em dados não padronizados.
- `ct.fit_transform` é aplicado ao conjunto de teste (deveria ser `transform`).
- O SHAP recebe dados crus de um modelo treinado com dados padronizados, o que torna não confiáveis os valores das variáveis contínuas.
- Hiperparâmetros dos modelos finais divergem dos melhores encontrados no grid.
- O mapa de correlações inclui `ID` e exclui variáveis categóricas (`numeric_only=True`).
- A seleção do melhor modelo foi feita no conjunto de teste.

---

## Próximos passos

- [ ] Usar `Sleep_Hours` (contínua) como desfecho principal, mantendo `Sleep_Quality` apenas como análise secundária
- [ ] Remover `Stress_Level`, `Health_Issues` e `Sleep_Quality` dos preditores (vazamento)
- [ ] Estimar o efeito do café com regressão (OLS com erros robustos; logit ordinal para a versão categórica) e checar não linearidade com splines
- [ ] Revisar o DAG com base nas relações testadas e tratá-lo como hipótese
- [ ] Corrigir os pontos técnicos listados acima e incluir baselines triviais e validação cruzada
- [ ] Incluir um aviso explícito sobre a natureza sintética dos dados nos resultados finais
- [ ] Deploy *(etapa original ainda pendente)*

## Como executar

```bash
# 1. Clone o repositório
git clone <url-do-seu-repositorio>
cd <nome-do-repositorio>

# 2. (Opcional) crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn mord xgboost shap jupyter

# 4. Abra o notebook
jupyter notebook cafeSono.ipynb
```

Verifique, na célula de leitura dos dados, se o caminho aponta para `synthetic_coffee_health_10000_in_.csv`.

## Tecnologias

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn · mord · XGBoost · SHAP · Jupyter

## Referências

- Cunningham, S. *Causal Inference: The Mixtape*, capítulo sobre grafos acíclicos dirigidos: <https://mixtape.scunning.com/03-directed_acyclical_graphs>
- Paleta de cores acessível a daltônicos: <https://davidmathlogic.com/colorblind/>

## Autoria

[Seu nome] · [LinkedIn / e-mail]

Projeto desenvolvido como checkpoint de aprendizado em uma série de cursos da Alura.
