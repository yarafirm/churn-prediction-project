# Churn Simulator: Customer Cancellation Analysis for a SaaS Business

*English version. A versão em português está em [README.pt-BR.md](README.pt-BR.md).*

## Contents

- [Goal](#goal)
- [Project Structure](#project-structure)
- [Libraries](#libraries)
- [Step by Step](#step-by-step)
  - [1. Data Generation](#1-data-generation)
  - [2. Exploratory Analysis](#2-exploratory-analysis)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Model Training](#4-model-training)
  - [5. Performance Evaluation](#5-performance-evaluation)
  - [6. Predicted Probability Distribution](#6-predicted-probability-distribution)
  - [7. Financial Impact](#7-financial-impact)
  - [8. Individual Predictor](#8-individual-predictor)
- [Results](#results)
- [Conclusions](#conclusions)
- [How to Run](#how-to-run)

---

## Goal

Build an end-to-end churn simulator on sample data for a subscription business, covering every stage of a data science project: data generation, exploration, predictive modeling with a Random Forest classifier and an estimate of the revenue at risk from the customers most likely to cancel.

---

## Project Structure

```
churn-prediction-project/
├── README.md                         # This document (English)
├── README.pt-BR.md                   # Portuguese version
└── data/
    ├── simulador_churn_Version3.py   # Main script: generation, EDA, model, financial impact
    ├── saas_customer_churn.csv       # Sample of the synthetic customer base
    └── processed/                    # Folder for generated charts
```

---

## Libraries

| Library | Use |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib | Charts |
| seaborn | Statistical visualizations |
| scikit-learn | Predictive modeling (Random Forest) |

---

## Step by Step

### 1. Data Generation

The script generates **10,000 records** simulating customers of a SaaS platform with three plans.

**Variables:**

| Variable | Description |
|---|---|
| `ID` | Unique customer identifier |
| `Plano` | Plan: Basic, Pro or Enterprise |
| `Valor_Mensal` | Monthly fee: R$ 29.90 / R$ 79.90 / R$ 249.90 |
| `Logins` | Logins in the last month (0 to 30) |
| `Tickets` | Support tickets opened (0 to 10) |
| `Meses_Permanencia` | Tenure in months (1 to 24) |
| `Churn` | 0 = stayed, 1 = cancelled |
| `LTV` | Lifetime value = tenure × monthly fee |

**Business rules behind churn:**

- Base probability: **15%**
- Fewer than 5 logins: **+35%** (the customer does not use the product)
- More than 6 tickets: **+25%** (the customer is unhappy)
- 10% of the cases get a random outcome, to mimic the unpredictability of real data

---

### 2. Exploratory Analysis

Before training any model, the data was explored to understand its patterns.

**Table:** statistics grouped by plan (number of customers, average churn, average logins, average tickets and average LTV).

**Charts (three subplots side by side):**

<img width="800" src="https://github.com/user-attachments/assets/8a4bea88-edd6-4e62-b43d-19f0dcc17e68" alt="Exploratory analysis: logins and tickets by churn status, churn rate by plan" />

| Subplot | What it shows | How to read it |
|---|---|---|
| Left | Boxplot of logins by churn status | Customers who left have a much lower median of logins |
| Center | Boxplot of tickets by churn status | Customers who left opened more support tickets |
| Right | Churn rate by plan | Compares the cancellation share across Basic, Pro and Enterprise |

**What stands out:**
- Customers with few logins cancel more often, so disuse is the main risk factor
- Many tickets point to friction with the product
- Churn rate is similar across plans because the plan itself is not part of the probability rule, only usage behavior is

---

### 3. Data Preparation

- The categorical variable `Plano` was encoded as a numeric code (`Plano_Cod`)
- The dataset was split into **80% training** and **20% test**
- The split was **stratified** (`stratify=y`) to keep the same churn proportion in both sets

**Features used:**

```
Valor_Mensal | Logins | Tickets | Meses_Permanencia | Plano_Cod
```

---

### 4. Model Training

A **Random Forest Classifier** was trained with the following hyperparameters:

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | More trees to stabilize the predictions |
| `max_depth` | 10 | Limits depth to avoid overfitting |
| `min_samples_split` | 20 | Requires at least 20 samples to split a node |
| `min_samples_leaf` | 10 | Each leaf must hold at least 10 samples |

These values balance predictive power and generalization, so the model does not memorize the training data.

---

### 5. Performance Evaluation

The model was evaluated on the test set (2,000 records).

**Metrics:**

| Metric | Meaning |
|---|---|
| Precision | Of the customers the model flagged as leaving, how many actually left |
| Recall | Of the customers who actually left, how many the model caught |
| F1-Score | Harmonic mean of precision and recall |
| Accuracy | Overall hit rate |

**Charts (two subplots side by side):**

<img width="800" src="https://github.com/user-attachments/assets/48ab2cd1-c1d0-4e4d-802b-d08cfc81655f" alt="Confusion matrix and feature importance" />

| Subplot | What it shows | How to read it |
|---|---|---|
| Left | **Confusion matrix** | Top-left quadrant = correctly predicted stays. Bottom-right = correctly predicted cancellations. The other two quadrants are errors |
| Right | **Feature importance** | Ranking of the features that contributed most to the model's predictions |

**What stands out:**
- `Logins` is the most important variable, which makes sense since the main churn rule is disuse
- `Tickets` comes second
- `Meses_Permanencia` and `Valor_Mensal` matter less
- `Plano_Cod` has little direct influence (the plan alone does not drive churn)

---

### 6. Predicted Probability Distribution

Each test customer receives a continuous churn probability (0% to 100%) rather than only a binary label.

<img width="800" src="https://github.com/user-attachments/assets/3354e6ef-4df4-4e91-908b-f08c5d72854a" alt="Distribution of predicted churn probabilities" />

| Element | What it shows |
|---|---|
| Blue bars | Probability distribution of customers who **stayed** |
| Coral bars | Probability distribution of customers who **left** |
| Dashed line | Decision threshold (0.5) |

**What stands out:**
- The further apart the two distributions, the better the model separates the classes
- Customers who stayed concentrate at low probabilities (left side)
- Customers who left concentrate at high probabilities (right side)
- The overlap in the middle is where the model struggles most

---

### 7. Financial Impact

Beyond prediction, the script estimates the financial impact of churn.

**High-risk definition:** churn probability ≥ 60%.

**Metrics:**

| Metric | Description |
|---|---|
| High-risk customers | How many customers have probability ≥ 60% |
| % of the base at risk | Share of the total |
| Total LTV at risk | Sum of the lifetime value of high-risk customers |
| Average LTV at risk | Average LTV within that group |

**Breakdown by plan:** a table with the number of high-risk customers, their average probability and the total LTV at risk for each plan (Basic, Pro, Enterprise).

This gives an estimate of how much revenue the company would lose if it took no action on the customers flagged as high risk.

---

### 8. Individual Predictor

A function scores specific customer profiles and returns:

- The exact churn probability
- A risk label (low / moderate / high)
- The estimated LTV

**Profiles tested:**

| Profile | Plan | Logins | Tickets | Months | Expected risk |
|---|---|---|---|---|---|
| Disengaged customer | Pro | 2 | 9 | 3 | High |
| Loyal customer | Enterprise | 22 | 1 | 18 | Low |
| In-between customer | Basic | 8 | 5 | 6 | Moderate |

---

## Results

### Model Performance

The model separates customers who stay from customers who leave well, considering that 10% of the data carries deliberate noise.

### Main Churn Drivers

1. **Logins**, the dominant factor. Customers who do not use the product cancel
2. **Tickets**, the second factor. Many complaints signal dissatisfaction
3. **Tenure and fee**, with a smaller influence on the churn decision
4. **Plan**, with little relevance on its own

### Financial Impact

The high-risk group (≥ 60% probability) concentrates a significant share of the base's total LTV, so retention actions aimed at this group would have a high return.

---

## Conclusions

- Product disuse is the strongest signal of future cancellation. Monitoring login frequency is the most direct way to spot risk
- Support ticket volume is the second indicator. Customers who open many tickets need attention before they decide to leave
- The model separates risk profiles well even with noisy data, which indicates robustness
- The financial analysis shows that predicting churn has direct value: it lets the company prioritize retention on the customers who represent the largest potential loss

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run
cd data
python simulador_churn_Version3.py
```

Charts are displayed during execution. To save them to `data/processed/`, add `plt.savefig("processed/<name>.png")` before each `plt.show()` in the script.
