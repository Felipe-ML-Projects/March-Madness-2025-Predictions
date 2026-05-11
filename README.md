# March-Madness-Predictions
March Madness predictions using historical tournament results, team statistics, and machine learning models to estimate win probabilities and identify potential upsets.

# Key Steps
1. Sourcing external data: Integrating reputable third-party data (e.g., ESPN metrics) and cleaning it with tools like VLOOKUP, TRIM, and GenAI significantly improved predictive performance. Indicators such as BPI and Strength of Schedule demonstrated relevance.
2. Preprocessing drives results: Feature engineering had the biggest impact, such as building a fatigue score based on travel distance and time-zone shifts and scaling inputs with MinMax or Z-score normalization.
3. Feature selection + validation: We used Random Forests to identify high-impact variables, explored PCA for dimensionality reduction, and backtested against historical tournaments to account for time-series effects.
4. Model experimentation: Tested six different algorithms (Logistic Regression, XGBoost, RandomForest, Gaussian Naive Bayes, Support Vector Machine, Bayesian Logistic Regression) to compare predictive performance and understand how different modeling approaches influenced outcomes.

# Prediction Results
The model correctly predicted Florida as the 2025 national champion and nailed 3 of the 4 Final Four teams, including the Auburn–Florida semifinal. The main miss was projecting Tennessee instead of Houston, and expecting a Florida–Duke final instead of Florida–Houston.

# March Data Crunch Madness — NCAA Tournament Final Four Predictor

> Predicting the Final Four of the 2025 NCAA Men's Basketball Tournament using machine learning and engineered features derived from historical game data.

**Team Maroon 4** | Gael Mayanza-ouamba · Felipe Chen · Yichen Yang · Yanbing Ren


# Table of Contents

- [Business Case](#-business-case)
- [Problem Statement](#-problem-statement)
- [Methodology Overview](#-methodology-overview)
- [Feature Engineering](#-feature-engineering)
- [Variable Selection](#-variable-selection)
- [Models Built](#-models-built)
- [Validation Strategy](#-validation-strategy)
- [Model Evaluation](#-model-evaluation)
- [Final Four Predictions](#-final-four-predictions)
- [Getting Started](#-getting-started)
- [Team](#-team)

# Business Case

March Madness is one of the most-watched sporting events in the United States, generating over **$14 billion in bracket contest entries** annually. Despite this, predicting tournament outcomes is notoriously difficult — even expert analysts and seasoned fans routinely fail to pick a perfect bracket.

### Why Machine Learning?

Traditional bracketology relies on seedings, eye-test assessments, and narrative-driven picks. These approaches are:

- **Inconsistent** — human bias towards popular teams skews predictions.
- **Non-scalable** — manual analysis of all 68 teams across dozens of metrics is impractical.
- **Reactive** — pundit opinions often follow media narratives rather than on-court performance data.

A data-driven model can:

1. **Systematically process** dozens of team performance metrics simultaneously.
2. **Quantify matchup advantages** by computing the head-to-head difference in key statistics.
3. **Account for hidden factors** like coaching experience, player fatigue, and travel stress.
4. **Produce calibrated probabilities** for each matchup, enabling principled bracket construction.

### Who Benefits?

| Stakeholder | Use Case |

| Sports analysts & media | Supplement expert opinion with probabilistic predictions |
| Casual fans | Build smarter brackets backed by data |
| Sportsbooks & fantasy platforms | Model market probabilities and identify mispriced lines |
| Athletic departments | Analyse competitive weaknesses and opponent strengths |

---

# Problem Statement

**Binary Classification:** For any given NCAA Tournament matchup between Team 1 and Team 2, predict which team wins.

- **Target Variable:** Game outcome (Team 1 wins / Team 2 wins)
- **Task Type:** Supervised Binary Classification
- **Goal:** Correctly predict the Final Four teams for the 2025 NCAA Tournament
- **Evaluation Metrics:** Log Loss (primary), Accuracy

---

# Methodology Overview

The project followed a four-stage pipeline:

```
1. Variable Transformation
        ↓
2. Variable Selection (Random Forest feature importance)
        ↓
3. Model Building (5 ML models + 1 statistical model)
        ↓
4. Model Evaluation (Log Loss + Accuracy via Expanding Window Backtest)
```

### Stage 1 — Variable Transformation
Team statistics were converted into **head-to-head difference features** (Team 1 minus Team 2) to capture the relative advantage in each matchup. The only exception was `num_over_time`, which is a game-level attribute shared by both teams.

### Stage 2 — Variable Selection
Random Forest feature importance scores were used to identify the **top 10 most predictive variables** from the full feature space.

### Stage 3 — Model Building
Six models were trained: five machine learning algorithms and one Bayesian statistical model.

### Stage 4 — Model Evaluation
Models were compared using **Log Loss** and **Accuracy** across expanding window backtests. The best-performing model was used to generate Final Four predictions.


# Feature Engineering

Beyond raw team statistics, five custom features were engineered to capture factors not present in standard box-score data:

### 1. Basketball Power Index (BPI)
ESPN's **BPI** measures overall team strength and is designed specifically to be the best forward-looking predictor of NCAA performance. Used as-is from ESPN's public data.

### 2. W/L % Coach
A coaching-adjusted win-loss percentage that reflects the **head coach's historical record**, capturing the accumulated strategic experience and performance impact of coaching quality across seasons.

### 3. Top Player Influence
A weighted score measuring how heavily a team **relies on its best player**. Calculated by summing feature importances across player-level metrics (assist-to-turnover ratio, blocks per foul, defensive plays per foul, defensive rebounds, etc.) for players in the **Top 100 player list**.

```
Score = Σ (feature_importance_i × feature_value_i)  for i = 1 to n

Where:
  feature_value_i = 1  if team has a player in the Top 100 list
  feature_value_i = 0  otherwise
```

### 4. Adjusted Shooting %
A dynamically weighted shooting efficiency metric that accounts for a team's **3-point tendency** (`f3grate`):

```
Adjusted Shooting % =
  fg2pct × (0.4 if high-3P-team else 0.3) +
  fg3pct × (0.6 if high-3P-team else 0.5) +
  ftpct  × 0.1

Where "high-3P-team" = f3grate > league average f3grate
```

This better reflects how teams actually score compared to raw field goal percentages.

### 5. Combined Fatigue
Captures the **physical toll of travel** across tournament rounds:

```
Combined_Fatigue = (travel_distance / 1000) + |jet_lag|
```

Teams playing back-to-back games with long travel or significant time zone shifts face a quantifiable fatigue penalty.

# Variable Selection

Random Forest feature importance was used to select the **top 10 variables** with the most explanatory power for game outcomes:

| Rank | Feature | Description |

| 1 | `BPI_diff` | Basketball Power Index difference |
| 2 | `adjoe_diff` | Adjusted offensive efficiency difference |
| 3 | `W-L%_Coach` | Head coach career win-loss percentage |
| 4 | `seed_diff` | Tournament seeding difference |
| 5 | `Strength_Schedule` | Strength of schedule |
| 6 | `adjde_diff` | Adjusted defensive efficiency difference |
| 7 | `adjusted_shooting_%` | Engineered shooting efficiency (see above) |
| 8 | `oppstlrate` | Opponent steal rate |
| 9 | `arate_diff` | Assist rate difference |
| 10 | `combined_fatigue` | Engineered travel/fatigue metric (see above) |


## Models Built

Six models were developed — five ML classifiers and one Bayesian statistical model:

### 1. Gaussian Naive Bayes
- Assumes feature independence and Gaussian-distributed predictors.
- Fast, interpretable baseline.
- Well-suited to small feature sets.

### 2. XGBoost
- Gradient boosting algorithm with strong out-of-the-box performance.
- Handles non-linear interactions and missing values natively.
- Hyperparameter-tuned for this dataset.

### 3. Random Forest
- Ensemble of decision trees using bagging.
- Also used in the variable selection stage (feature importance).
- **Best-performing model** across Log Loss and Accuracy.

### 4. Logistic Regression
- Linear probabilistic classifier.
- Provides well-calibrated probability outputs — important for Log Loss scoring.
- Strong interpretability.

### 5. Support Vector Machine (SVM)
- Finds the optimal hyperplane separating wins and losses in feature space.
- Effective in high-dimensional settings with proper regularisation.

### 6. Bayesian Logistic Regression *(Statistical Model)*
- Logistic regression with Bayesian priors — produces posterior distributions over predictions.
- Naturally accounts for parameter uncertainty.
- Useful for producing credible probability intervals.


## Validation Strategy

### Expanding Window Cross-Validation

A standard random train-test split was **intentionally avoided** because tournament game outcomes are time-dependent — patterns from older seasons may not fully hold in newer ones.

Instead, an **Expanding Window** backtest was used:

```
Training 1 → Test 1
Training 1 + Test 1 → Training 2 → Test 2
Training 2 + Test 2 → Training 3 → Test 3
Training 3 + Test 3 → Training 4 → Test 4
```

Each fold uses all data up to that point for training, and tests on the immediate next time period. This mirrors real-world deployment conditions.

### Assumptions

| # | Assumption |

| 1 | Performance variables in NCAA Tournament games are **time-dependent** |
| 2 | Patterns within variables from **2024** are similar to those in **2025** |
| 3 | Patterns from **2023 and 2024** combined are similar to **2025** |

### Model Selection Steps

1. **Hyperparameter tuning** — each model fine-tuned via grid/random search.
2. **Metric comparison** — models compared by Log Loss and Accuracy across all folds.
3. **Best window selection** — the optimal expanding window size was chosen per model based on held-out performance.

# Model Evaluation

Six models were evaluated on **Log Loss** (primary) and **Accuracy**:

| Model | Log Loss | Accuracy |
|---|---|---|
| Gaussian Naive Bayes | — | — |
| XGBoost | — | — |
| Logistic Regression | — | — |
| **Random Forest** | **Lowest** | **Highest** |
| Support Vector Machine | — | — |
| Bayesian Logistic Regression | — | — |

> **Random Forest achieved the lowest log loss and highest accuracy** across all six models and was selected as the final prediction model.

> **Note:** Exact numeric scores from the evaluation run can be added to the table above after re-running the notebook.

---

## Final Four Predictions

Using the best-performing **Random Forest** model, the predicted 2025 NCAA Tournament Final Four:

```
         South/East Region          Midwest/West Region
         ──────────────────          ───────────────────
             Florida                      Duke
                ↓                           ↓
         [58% win prob]               [76% win prob]
             Auburn                      Tennessee

         Florida advances             Duke advances
```

| Matchup | Predicted Winner | Win Probability |

| Florida vs. Auburn | **Florida** | 58% |
| Duke vs. Tennessee | **Duke** | 76% |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/march-madness-predictor.git
cd march-madness-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook March_Madness.ipynb
```

---

# Dependencies

```
pandas
numpy
scikit-learn
xgboost
pymc  # for Bayesian Logistic Regression
matplotlib
seaborn
jupyter
```

# Project Structure

```
march-madness-predictor/
│
├── March_Madness.ipynb        # Main analysis and modelling notebook
├── data/                      # Raw and processed data files
│   ├── team_stats_2023.csv
│   ├── team_stats_2024.csv
│   └── tournament_results.csv
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```
# Future Work

- [ ] Incorporate **real-time injury reports** and **player availability** data
- [ ] Add **momentum features** — recent game performance in the weeks before the tournament
- [ ] Extend predictions to **full bracket simulation** (all 63 games, not just Final Four)
- [ ] Build an **interactive bracket visualiser** for end users
- [ ] Calibrate predicted probabilities using **Platt Scaling** or **Isotonic Regression**
- [ ] Experiment with **neural network** architectures for matchup prediction

# Team

**Maroon 4** — March Data Crunch Madness 2025

| Name | Role |

| Gael Mayanza-ouamba | Presentation |
| Felipe Chen | Modelling & Data Processing & Validation  |
| Yichen Yang | Feature Engineering & Analysis |
| Yanbing Ren | Modelling & Model Evaluation & Presentation |
---

> *Let's Go Gators! 🐊*
