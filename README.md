# March-Madness-Predictions
March Madness predictions using historical tournament results, team statistics, and machine learning models to estimate win probabilities and identify potential upsets.

# Key Steps
1. Sourcing external data: Integrating reputable third-party data (e.g., ESPN metrics) and cleaning it with tools like VLOOKUP, TRIM, and GenAI significantly improved predictive performance. Indicators such as BPI and Strength of Schedule demonstrated relevance.
2. Preprocessing drives results: Feature engineering had the biggest impact—such as building a fatigue score based on travel distance and time-zone shifts—and scaling inputs with MinMax or Z-score normalization.
3. Feature selection + validation: We used Random Forests to identify high-impact variables, explored PCA for dimensionality reduction, and backtested against historical tournaments to account for time-series effects.
4. Model experimentation: Tested six different algorithms (Logistic Regression, XGBoost, RandomForest, Gaussian Naive Bayes, Support Vector Machine, Bayesian Logistic Regression) to compare predictive performance and understand how different modeling approaches influenced outcomes.

# Prediction Results
The model correctly predicted Florida as the 2025 national champion and nailed 3 of the 4 Final Four teams, including the Auburn–Florida semifinal. The main miss was projecting Tennessee instead of Houston, and expecting a Florida–Duke final instead of Florida–Houston.

