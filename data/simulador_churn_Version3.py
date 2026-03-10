# =============================================================================
# Simulador de Churn — Geração de dados, modelagem e análise
# =============================================================================

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. GERAÇÃO DA BASE DE DADOS
# =============================================================================

def gerar_base_clientes(n=10000):
    planos = ['Basic', 'Pro', 'Enterprise']
    valores = {'Basic': 29.90, 'Pro': 79.90, 'Enterprise': 249.90}

    dados = []
    for i in range(1, n + 1):
        plano = random.choice(planos)
        valor = valores[plano]
        logins = random.randint(0, 30)
        tickets = random.randint(0, 10)
        meses = random.randint(1, 24)

        # Probabilidade base de churn
        prob = 0.15

        # Clientes com poucos logins tendem a abandonar mais
        if logins < 5:
            prob += 0.35

        # Muitos tickets de suporte indicam insatisfação
        if tickets > 6:
            prob += 0.25

        # Ruído: ~10% dos casos têm resultado aleatório (comportamento imprevisível)
        if random.random() < 0.10:
            churn = random.choice([0, 1])
        else:
            churn = 1 if random.random() < prob else 0

        dados.append({
            'ID': f'CUST-{i:05d}',
            'Plano': plano,
            'Valor_Mensal': valor,
            'Logins': logins,
            'Tickets': tickets,
            'Meses_Permanencia': meses,
            'Churn': churn,
            'LTV': round(meses * valor, 2)
        })

    return pd.DataFrame(dados)

print("Gerando base de dados com 10.000 clientes...")
df = gerar_base_clientes()
print(f"Base gerada: {df.shape[0]} registros, {df.shape[1]} colunas")
print(f"Taxa de churn na base: {df['Churn'].mean()*100:.1f}%\n")
print(df.head(10))

# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# =============================================================================

print("\n--- Distribuição por Plano ---")
print(df.groupby('Plano').agg(
    clientes=('ID', 'count'),
    churn_medio=('Churn', 'mean'),
    logins_medio=('Logins', 'mean'),
    tickets_medio=('Tickets', 'mean'),
    ltv_medio=('LTV', 'mean')
).round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribuição de logins por situação de churn
sns.boxplot(data=df, x='Churn', y='Logins', ax=axes[0], palette='coolwarm')
axes[0].set_xticklabels(['Ficou', 'Saiu'])
axes[0].set_title('Logins por Situação de Churn')

# Distribuição de tickets por situação de churn
sns.boxplot(data=df, x='Churn', y='Tickets', ax=axes[1], palette='coolwarm')
axes[1].set_xticklabels(['Ficou', 'Saiu'])
axes[1].set_title('Tickets por Situação de Churn')

# Taxa de churn por plano
taxa_plano = df.groupby('Plano')['Churn'].mean().sort_values()
taxa_plano.plot(kind='barh', ax=axes[2], color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[2].set_title('Taxa de Churn por Plano')
axes[2].set_xlabel('Taxa de Churn')

plt.tight_layout()
plt.show()

# =============================================================================
# 3. PREPARAÇÃO DOS DADOS
# =============================================================================

print("\nPreparando os dados para modelagem...")

df_mod = df.copy()
df_mod['Plano_Cod'] = df_mod['Plano'].astype('category').cat.codes

features = ['Valor_Mensal', 'Logins', 'Tickets', 'Meses_Permanencia', 'Plano_Cod']
X = df_mod[features]
y = df_mod['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} registros")
print(f"Teste:  {X_test.shape[0]} registros")

# =============================================================================
# 4. TREINAMENTO DO MODELO
# =============================================================================

print("\nTreinando Random Forest...")

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)[:, 1]

# =============================================================================
# 5. AVALIAÇÃO DE PERFORMANCE
# =============================================================================

print("\n--- RELATÓRIO DE PERFORMANCE ---")
print(classification_report(y_test, previsoes, target_names=['Ficou', 'Saiu']))

# Matriz de confusão + Importância das variáveis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(y_test, previsoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou', 'Saiu'])
disp.plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão')

importancias = pd.Series(modelo.feature_importances_, index=features).sort_values()
cores = plt.cm.viridis(np.linspace(0.3, 0.9, len(importancias)))
importancias.plot(kind='barh', color=cores, ax=axes[1])
axes[1].set_title('Importância das Variáveis para o Churn')
axes[1].set_xlabel('Importância')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. DISTRIBUIÇÃO DAS PROBABILIDADES PREVISTAS
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(probabilidades[y_test == 0], bins=30, alpha=0.6, label='Ficou', color='steelblue', edgecolor='white')
ax.hist(probabilidades[y_test == 1], bins=30, alpha=0.6, label='Saiu', color='coral', edgecolor='white')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Limiar 0.5')
ax.set_title('Distribuição das Probabilidades Previstas')
ax.set_xlabel('Probabilidade de Churn')
ax.set_ylabel('Frequência')
ax.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 7. ANÁLISE DE PERDA FINANCEIRA (LTV EM RISCO)
# =============================================================================

df_teste = X_test.copy()
df_teste['Churn_Real'] = y_test.values
df_teste['Prob_Churn'] = probabilidades
df_teste['LTV'] = df_mod.loc[X_test.index, 'LTV'].values
df_teste['Plano'] = df_mod.loc[X_test.index, 'Plano'].values

# Clientes de alto risco: probabilidade >= 60%
alto_risco = df_teste[df_teste['Prob_Churn'] >= 0.60]

print("\n--- IMPACTO FINANCEIRO ---")
print(f"Clientes no teste:       {len(df_teste)}")
print(f"Clientes de alto risco:  {len(alto_risco)} ({len(alto_risco)/len(df_teste)*100:.1f}%)")
print(f"LTV total em risco:      R$ {alto_risco['LTV'].sum():,.2f}")
print(f"LTV médio em risco:      R$ {alto_risco['LTV'].mean():,.2f}")

print("\nAlto risco por plano:")
print(alto_risco.groupby('Plano').agg(
    clientes=('Prob_Churn', 'count'),
    prob_media=('Prob_Churn', 'mean'),
    ltv_total=('LTV', 'sum')
).round(2))

# =============================================================================
# 8. PREDITOR INDIVIDUAL
# =============================================================================

def prever_churn(plano, logins, tickets, meses):
    """Calcula a probabilidade de churn para um cliente específico."""
    valores = {'Basic': 29.90, 'Pro': 79.90, 'Enterprise': 249.90}
    mapa_plano = {'Basic': 0, 'Enterprise': 1, 'Pro': 2}

    valor = valores[plano]
    entrada = pd.DataFrame(
        [[valor, logins, tickets, meses, mapa_plano[plano]]],
        columns=features
    )

    prob = modelo.predict_proba(entrada)[0][1]
    ltv = round(meses * valor, 2)

    print(f"\n--- Previsão Individual ---")
    print(f"  Plano:        {plano} (R$ {valor}/mês)")
    print(f"  Logins:       {logins}")
    print(f"  Tickets:      {tickets}")
    print(f"  Permanência:  {meses} meses")
    print(f"  LTV:          R$ {ltv:,.2f}")
    print(f"  Risco:        {prob*100:.1f}%", end="")
    if prob >= 0.60:
        print("  [ALTO RISCO]")
    elif prob >= 0.35:
        print("  [RISCO MODERADO]")
    else:
        print("  [BAIXO RISCO]")

    return prob

# --- Exemplos ---
print("\n" + "="*50)
print("TESTES COM PERFIS DE CLIENTE")
print("="*50)

# Cliente de alto risco: poucos logins, muitos tickets, pouco tempo
prever_churn('Pro', logins=2, tickets=9, meses=3)

# Cliente estável: bastante uso, poucos problemas, cliente antigo
prever_churn('Enterprise', logins=22, tickets=1, meses=18)

# Cliente intermediário
prever_churn('Basic', logins=8, tickets=5, meses=6)