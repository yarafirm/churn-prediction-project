# Simulador de Churn — Análise de Cancelamento de Clientes SaaS

## Sumário

- [Objetivo](#objetivo)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [O que foi utilizado? ](#o-que-foi-utilizado)
- [Passo a Passo da Análise](#passo-a-passo-da-análise)
  - [1. Geração da Base de Dados](#1-geração-da-base-de-dados)
  - [2. Análise Exploratória](#2-análise-exploratória)
  - [3. Preparação dos Dados](#3-preparação-dos-dados)
  - [4. Treinamento do Modelo](#4-treinamento-do-modelo)
  - [5. Avaliação de Performance](#5-avaliação-de-performance)
  - [6. Distribuição das Probabilidades](#6-distribuição-das-probabilidades)
  - [7. Análise de Impacto Financeiro](#7-análise-de-impacto-financeiro)
  - [8. Preditor Individual](#8-preditor-individual)
- [Resultados Obtidos](#resultados-obtidos)
- [Conclusões](#conclusões)
- [Como Executar](#como-executar)

---

## Objetivo

Construir um simulador completo de churn (cancelamento) para dados amostrais de uma empresa, passando por todas as etapas de um projeto de ciência de dados: geração dos dados, exploração, modelagem preditiva com Random Forest e análise do impacto financeiro dos cancelamentos.

---

## Estrutura do Projeto

```
simulador-churn/
├── simulador_churn.py          # Código principal
├── README.md                   # Este documento
└── graficos/                   # Pasta para salvar os gráficos gerados
    ├── 01_analise_exploratoria.png
    ├── 02_matriz_importancia.png
    ├── 03_distribuicao_probabilidades.png
```

---

## O que foi utilizado? 

| Biblioteca | Uso |
|---|---|
| pandas | Manipulação de dados |
| numpy | Operações numéricas |
| matplotlib | Gráficos |
| seaborn | Visualizações estatísticas |
| scikit-learn | Modelagem preditiva (Random Forest) |

---

## Passo a Passo da Análise

### 1. Geração da Base de Dados

Foram gerados **10.000 registros** simulando clientes de uma plataforma SaaS com três planos.

**Variáveis criadas:**

| Variável | Descrição |
|---|---|
| `ID` | Identificador único do cliente |
| `Plano` | Basic, Pro ou Enterprise |
| `Valor_Mensal` | R$ 29.90 / R$ 79.90 / R$ 249.90 |
| `Logins` | Quantidade de logins no último mês (0 a 30) |
| `Tickets` | Chamados de suporte abertos (0 a 10) |
| `Meses_Permanencia` | Tempo como cliente (1 a 24 meses) |
| `Churn` | 0 = ficou, 1 = cancelou |
| `LTV` | Lifetime Value = Meses × Valor Mensal |

**Regras de negócio para o churn:**

- Probabilidade base: **15%**
- Menos de 5 logins: **+35%** (cliente não usa o produto)
- Mais de 6 tickets: **+25%** (cliente insatisfeito)
- 10% dos casos têm resultado aleatório (simula imprevisibilidade real)

---

### 2. Análise Exploratória

Antes de treinar qualquer modelo, os dados foram explorados para entender padrões.

**Tabela gerada:** estatísticas agrupadas por plano (quantidade de clientes, churn médio, logins médio, tickets médio e LTV médio).

**Gráficos gerados (3 subplots lado a lado):**

<img width="800" src="https://github.com/user-attachments/assets/8a4bea88-edd6-4e62-b43d-19f0dcc17e68" alt="Matriz de Confusão e Importância das Variáveis" />

<img width="800" src="https://github.com/user-attachments/assets/c4a6e588-5e26-488d-8180-8ca332e5cd70" alt="Dashboard de Métricas de Marketing" />

<img width="800" src="https://github.com/user-attachments/assets/e3b8f1ce-38c7-436b-9370-a31ec934a79f" alt="Análise de Funil de Conversão" />




| Subplot | O que mostra | Como interpretar |
|---|---|---|
| Esquerdo | Boxplot de Logins por Churn | Clientes que saíram têm mediana de logins muito mais baixa |
| Centro | Boxplot de Tickets por Churn | Clientes que saíram abriram mais chamados de suporte |
| Direito | Taxa de churn por plano | Compara a proporção de cancelamento entre Basic, Pro e Enterprise |

**O que se observa:**
- Clientes com poucos logins cancelam mais — o desuso é o principal fator de risco
- Muitos tickets indicam atrito com o produto
- A taxa de churn é semelhante entre os planos porque o plano sozinho não entra na regra de probabilidade, apenas o comportamento de uso

---

### 3. Preparação dos Dados

Os dados foram preparados para alimentar o modelo:

- A variável categórica `Plano` foi convertida para código numérico (`Plano_Cod`)
- O dataset foi dividido em **80% treino** e **20% teste**
- Foi aplicada **estratificação** (`stratify=y`) para manter a mesma proporção de churn no treino e no teste

**Features utilizadas:**

```
Valor_Mensal | Logins | Tickets | Meses_Permanencia | Plano_Cod
```

---

### 4. Treinamento do Modelo

Foi treinado um **Random Forest Classifier** com os seguintes hiperparâmetros:

| Parâmetro | Valor | Motivo |
|---|---|---|
| `n_estimators` | 200 | Mais árvores para estabilizar as previsões |
| `max_depth` | 10 | Limita a profundidade para evitar overfitting |
| `min_samples_split` | 20 | Exige no mínimo 20 amostras para dividir um nó |
| `min_samples_leaf` | 10 | Cada folha precisa ter ao menos 10 amostras |

Esses parâmetros foram escolhidos para equilibrar capacidade preditiva com generalização — o modelo não decora os dados de treino.

---

### 5. Avaliação de Performance

O modelo foi avaliado no conjunto de teste (2.000 registros).

**Métricas geradas:**

| Métrica | O que significa |
|---|---|
| Precision | Dos que o modelo disse que iam sair, quantos realmente saíram |
| Recall | Dos que realmente saíram, quantos o modelo conseguiu identificar |
| F1-Score | Média harmônica entre precision e recall |
| Accuracy | Taxa geral de acerto |

**Gráficos gerados (2 subplots lado a lado):**

<img width="800" src="https://github.com/user-attachments/assets/48ab2cd1-c1d0-4e4d-802b-d08cfc81655f" alt="Dashboard de Performance de Vendas e Marketing" />


| Subplot | O que mostra | Como interpretar |
|---|---|---|
| Esquerdo | **Matriz de Confusão** | Quadrante superior-esquerdo = acertou quem ficou. Inferior-direito = acertou quem saiu. Os outros dois quadrantes são erros |
| Direito | **Importância das Variáveis** | Ranking de quais features mais contribuíram para as previsões do modelo |

**O que se observa:**
- `Logins` é a variável mais importante — faz sentido, já que a principal regra de churn é o desuso
- `Tickets` aparece como segundo fator mais relevante
- `Meses_Permanencia` e `Valor_Mensal` têm importância menor
- `Plano_Cod` tem pouca influência direta (o plano por si só não determina churn)

---

### 6. Distribuição das Probabilidades

Cada cliente do teste recebe uma probabilidade contínua de churn (0% a 100%), não apenas um rótulo binário.

**Gráfico gerado:**

<img width="800" src="https://github.com/user-attachments/assets/3354e6ef-4df4-4e91-908b-f08c5d72854a" alt="Dashboard de métricas complementares" />

| Elemento | O que mostra |
|---|---|
| Barras azuis | Distribuição de probabilidade dos clientes que **ficaram** |
| Barras corais | Distribuição de probabilidade dos clientes que **saíram** |
| Linha tracejada | Limiar de decisão (0.5) |

**O que se observa:**
- Quanto mais separadas as duas distribuições, melhor o modelo discrimina entre as classes
- Clientes que ficaram tendem a ter probabilidades baixas (concentrados à esquerda)
- Clientes que saíram tendem a ter probabilidades altas (concentrados à direita)
- A zona de sobreposição no meio é onde o modelo tem mais dificuldade de decisão

---

### 7. Análise de Impacto Financeiro

Além da previsão, foi calculado o impacto financeiro do churn.

**Definição de alto risco:** probabilidade de churn ≥ 60%.

**Métricas calculadas:**

| Métrica | Descrição |
|---|---|
| Total de clientes de alto risco | Quantos clientes têm probabilidade ≥ 60% |
| % da base em risco | Proporção do total |
| LTV total em risco | Soma do Lifetime Value dos clientes de alto risco |
| LTV médio em risco | Média do LTV desse grupo |

**Detalhamento por plano:** tabela mostrando quantidade de clientes de alto risco, probabilidade média e LTV total em risco para cada plano (Basic, Pro, Enterprise).

Essa análise permite estimar quanto dinheiro a empresa perderia se não agisse sobre os clientes identificados como alto risco.

---

### 8. Preditor Individual

Uma função permite testar perfis específicos de clientes e obter:

- Probabilidade exata de churn
- Classificação de risco (baixo / moderado / alto)
- LTV estimado

**Exemplos testados:**

| Perfil | Plano | Logins | Tickets | Meses | Risco Esperado |
|---|---|---|---|---|---|
| Cliente desengajado | Pro | 2 | 9 | 3 | Alto |
| Cliente fiel | Enterprise | 22 | 1 | 18 | Baixo |
| Cliente intermediário | Basic | 8 | 5 | 6 | Moderado |

---

## Resultados Obtidos

### Performance do Modelo

O modelo alcançou boa capacidade de separação entre clientes que ficam e clientes que saem, considerando que os dados contêm 10% de ruído proposital.

### Principais Fatores de Churn

1. **Logins** — fator dominante. Clientes que não usam o produto cancelam
2. **Tickets** — segundo fator. Muitas reclamações indicam insatisfação
3. **Permanência e valor** — influência menor na decisão de churn
4. **Plano** — pouco relevante isoladamente

### Impacto Financeiro

O grupo de alto risco (≥ 60% de probabilidade) concentra uma parcela significativa do LTV total da base, indicando que ações de retenção direcionadas a esse grupo teriam retorno elevado.

---

## Conclusões

- O desuso do produto é o sinal mais forte de cancelamento futuro. Monitorar a frequência de logins é a ação mais direta para identificar risco
- O volume de chamados de suporte é o segundo indicador. Clientes que abrem muitos tickets precisam de atenção antes que decidam sair
- O modelo consegue separar bem os perfis de risco mesmo com ruído nos dados, o que indica robustez
- A análise financeira mostra que prever churn tem valor direto: permite priorizar ações de retenção nos clientes que representam maior perda potencial

---

## Como Executar

```bash
# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn

# Executar
python simulador_churn.py
```

Os gráficos serão exibidos durante a execução. Para salvá-los na pasta `graficos/`, adicione `plt.savefig("graficos/nome.png")` antes de cada `plt.show()` no código.
