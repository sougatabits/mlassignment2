# Machine Learning Assignment -2 Submitted by Sougata Das BITS ID- 2024DC04257
# Topic: Rainfall Prediction across regions in Australlia
 
# Problem Statement
 
The objective of this project is to predict whether it will rain tomorrow
(RainTomorrow) based on various meteorological features such as temperature,
humidity, pressure, wind speed, and rainfall for Australlia.
 
This is a Binary Classification problem.
 
Target Variable:
- Yes → Rain Tomorrow
- No → No Rain Tomorrow
 
 
# Dataset Description
 
Dataset Used: Rain in Australia Dataset
 
- Source: Kaggle
- Source Type : csv
- Total Instances: ~145,000
- Total Features: 23
- Type: Weather observation data
- Target Variable: RainTomorrow
 
The dataset contains both numerical and categorical features such as:
- Date
- Location
- MinTemp
- MaxTemp
- Rainfall
- Evaporation
- Sunshine
- WindGustDir
- WindGustSpeed
- WindDir9am
- WindDir3pm
- WindSpeed9am
- WindSpeed3pm
- Humidity9am
- Humidity3pm
- Pressure9am
- Pressure3pm
- Cloud9am
- Cloud3pm
- Temp9am
- Temp3pm
- RainToday
- RainTomorrow
 
Missing values were handled using:
- Median (for numerical features)
- Mode (for categorical features)
 
Categorical variables were label encoded.
 
 
# Model Comparison Table
 
| ML Model Name                                | Accuracy | AUC     | Precision   | Recall  | F1       | MCC |
|----------------------------------------------|----------|---------|-------------|---------|----------|-----|
|   rainfall_logistic_regression_classification| 0.844580 | 0.865472 |  0.727273 | 0.490667|  0.585987 |  0.509246
|        rainfall_decision_tree_classification | 0.782095 | 0.693237  | 0.513473 | 0.532078 | 0.522610 |  0.381598
|   rainfall_k-nearest_neighbor_classification | 0.835965  | 0.823849  | 0.687995 | 0.490824 | 0.572920  | 0.485271
|          rainfall_naive_bayes_classification | 0.804740 |  0.827552 |  0.559376 | 0.607373 | 0.582387 |  0.455907
|        rainfall_random_forest_classification | 0.856816 | 0.886617 |  0.772544 | 0.512000 | 0.615849  | 0.549280
|                rainfall_xgboost_classfication  | 0.860368 |  0.892677  | 0.758495 | 0.553255 | 0.639819  | 0.566475
 
 
## Observations
 
| ML Model Name         | Observation                                           |
|-----------------------|-------------------------------------------------------|
| Logistic Regression   | Performs well on linearly separable weather patterns.
| Decision Tree         | Tends to overfit but captures non-linear relationships.
| KNN                   | Sensitive to feature scaling and noise.
| Naive Bayes           | Fast and simple but assumes independence of features.
| Random Forest         | Provides better stability and reduced overfitting.
| XGBoost               | Generally provides highest performance due to boosting technique.

