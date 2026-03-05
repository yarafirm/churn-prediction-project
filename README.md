# SaaS Churn Intelligence System

Este repositório contém uma solução ponta a ponta para identificação e prevenção de cancelamentos em empresas de software (SaaS).

## Objetivo
Fornecer uma ferramenta capaz de prever quais clientes estão propensos a abandonar o serviço (**Churn**), baseando-se em métricas de engajamento (logins) e satisfação (tickets de suporte).

## Metodologia
1. **Dataset:** Gerado via script para simular 10.000 perfis de usuários com variabilidade realista e ruído estatístico.
2. **Modelo:** Utilização do algoritmo **Random Forest Classifier**, escolhido por sua capacidade de lidar com dados não lineares e fornecer a importância de cada variável.
3. **Métricas de Sucesso:** Focamos não apenas em acurácia, mas no impacto financeiro (LTV - Lifetime Value) que o churn causa na operação.

## Funcionalidades
* **Geração automática de dados:** Criação de um CSV realista para testes.
* **Análise de variáveis:** Identificação de que o baixo uso da plataforma é o maior vilão da retenção.
* **Ferramenta de predição:** Interface de código que permite testar clientes individualmente para tomada de decisão imediata.

## Como instalar
1. Clone o repositório.
2. Instale as dependências: `pip install pandas scikit-learn matplotlib seaborn`
3. Execute o arquivo `main.py`.
