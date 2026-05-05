# Steam Top Games 2026 Research Project

## Project Description

This project is a machine learning research project based on the **Steam Top Games 2026** dataset.

The goal of the project is to predict the **peak number of concurrent users** of a Steam game (`peak_ccu`) using game-related features such as price, reviews, playtime, platforms, release year, estimated owners, and genre.

The project was prepared as a homework assignment for the course **Artificial Intelligence Systems**.

---

## Dataset

The dataset contains information about popular Steam games.

### Dataset Size

- Rows: 1495 games
- Columns: 29 original columns

### Main Columns

The dataset includes:

- `name` — game title
- `release_date` — release date of the game
- `price_usd` — game price in USD
- `is_free` — whether the game is free
- `discount_pct` — discount percentage
- `genres` — game genres
- `platforms_win`, `platforms_mac`, `platforms_linux` — supported platforms
- `metacritic_score` — Metacritic score
- `recommendations` — number of Steam recommendations
- `positive_reviews` — number of positive reviews
- `negative_reviews` — number of negative reviews
- `estimated_owners` — estimated number of owners
- `avg_playtime_forever` — average total playtime
- `peak_ccu` — peak concurrent users

---

## Research Goal

The main research question is:

> Can we predict the peak number of concurrent users of a Steam game based on its metadata, reviews, playtime, and genre?

This is a **regression task**, because the target variable `peak_ccu` is a numerical value.

---

## Data Preprocessing

Several preprocessing steps were applied:

### 1. Handling missing values

The dataset contains missing values, especially in `metacritic_score`.

- Numerical missing values were filled with the **median**
- Categorical missing values were filled with the **most frequent value**

### 2. Feature engineering

Additional features were created:

- `release_year` — extracted from `release_date`
- `game_age_years` — calculated as `2026 - release_year`
- `owners_midpoint` — converted from ranges like `200,000 .. 500,000`
- `total_reviews` — positive reviews + negative reviews
- `positive_review_ratio` — share of positive reviews
- `main_genre` — first genre from the genre list

### 3. Encoding categorical variables

Categorical features were converted into numerical format using:

- `OneHotEncoder`

This was necessary because machine learning models cannot directly work with text categories.

### 4. Target transformation

The target variable `peak_ccu` is highly skewed: a small number of games have extremely high player counts.

Because of this, the model used:

- `log1p` transformation before training
- `expm1` transformation after prediction

This makes the model more stable and reduces the influence of extreme outliers.

---

## Models Used

Three models were compared:

| Model | Purpose |
|---|---|
| Baseline Median Model | Simple reference model |
| Decision Tree Regressor | Simple tree-based model |
| Random Forest Regressor | Main model with better generalization |

The main model was:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42
)
```

---

## Model Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Baseline Median Model | 910.05 | 8517.95 | -0.011 |
| Decision Tree Regressor | 727.17 | 7454.50 | 0.226 |
| Random Forest Regressor | 662.81 | 5878.68 | 0.518 |

---

## Interpretation of Metrics

### MAE

MAE means **Mean Absolute Error**.

In this project, the Random Forest model has an MAE of about **663**, which means that, on average, the model makes an error of about 663 concurrent users.

### RMSE

RMSE penalizes large errors more strongly than MAE.

The RMSE is high because the dataset has strong outliers: some games have extremely large player counts.

### R2 Score

The Random Forest model achieved an R2 score of about **0.518**.

This means that the model explains around **51.8%** of the variation in the target variable.

---

## Main Findings

The best model was **Random Forest Regressor**.

The most important features were usually related to:

- positive reviews
- total reviews
- average playtime
- recent playtime
- recommendations
- median playtime

This means that user activity and review data are strongly connected with the popularity of Steam games.

---

## Conclusion

The project shows that it is possible to predict Steam game popularity using machine learning.

The Random Forest model performed better than the baseline model and the Decision Tree model. However, the task is still difficult because player activity is affected by many external factors, such as marketing, updates, streamers, discounts, and trends.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd steam-ai-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## Project Structure

```text
steam-ai-project/
│
├── data/
│   └── steam_top_games_2026.csv
│
├── notebooks/
│   └── steam_games_research.ipynb
│
├── outputs/
│   ├── model_results.csv
│   └── feature_importance.csv
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Questions I Can Answer About This Project

### What is the goal of the project?

The goal is to predict the peak number of concurrent users of Steam games using machine learning.

### What type of machine learning task is this?

This is a regression task because the target variable is numerical.

### What is the target variable?

The target variable is `peak_ccu`, which means peak concurrent users.

### Why did you use Random Forest?

Random Forest works well with mixed data, handles non-linear relationships, and usually generalizes better than a single Decision Tree.

### Why did you use OneHotEncoder?

Because categorical columns like genre cannot be used directly by machine learning models. OneHotEncoder converts categories into numerical columns.

### Why did you use SimpleImputer?

Because some columns contain missing values. SimpleImputer fills missing values so the model can train correctly.

### Why did you transform the target with log1p?

Because `peak_ccu` is highly skewed. Log transformation reduces the effect of extreme values and makes training more stable.

### What conclusion did you make?

The model can partly predict game popularity, but exact prediction is difficult because popularity depends on many external factors that are not fully represented in the dataset.
