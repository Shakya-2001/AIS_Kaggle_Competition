# AIS Competition — Beats Per Minute Prediction

## 1. Project Overview

The objective is to predict the `BeatsPerMinute` of audio tracks using numerical audio features.

**Problem Type:** Regression  
**Evaluation Metric:** Root Mean Squared Error (RMSE)  
**Development Environment:** Python, VS Code, Jupyter Notebook

## 2. Dataset

The competition dataset contains synthetic audio-feature data generated using a deep learning model.

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 524,164 | 11 |
| Testing | 174,722 | 10 |
| Submission | 174,722 | 2 |

### Input Features

- RhythmScore
- AudioLoudness
- VocalContent
- AcousticQuality
- InstrumentalScore
- LivePerformanceLikelihood
- MoodScore
- TrackDurationMs
- Energy

**Target Variable:** `BeatsPerMinute`

The training dataset contains no missing values.

The `id` column was excluded from model training.

## 3. Methodology

### Data Preparation

1. Loaded training and testing datasets using Pandas.
2. Separated input features and target variable.
3. Removed the identifier column.
4. Split the training data into 80% training and 20% validation using `random_state=42`.
5. Used the same validation split for all three models.

### Machine Learning Models

Three gradient boosting regression algorithms were evaluated:

- LightGBM
- CatBoost
- XGBoost

Early stopping was used to identify the best training iteration.

## 4. Model Evaluation

The models were evaluated using Root Mean Squared Error (RMSE).

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Lower RMSE indicates better predictive performance.

### Validation Results

| Model | Validation RMSE | Best Iteration |
|---|---:|---:|
| **LightGBM** | **26.436657** | **91** |
| XGBoost | 26.438474 | 93 |
| CatBoost | 26.439033 | 86 |

**Best Model: LightGBM**

LightGBM achieved the lowest validation RMSE among the three models.

## 5. Ensemble Experiment

Weighted averaging was evaluated using predictions from LightGBM, CatBoost, and XGBoost.

Different weight combinations were tested on the validation dataset.

| Ensemble | Validation RMSE |
|---|---:|
| LightGBM 100% | **26.436657** |
| LightGBM 90% + XGBoost 10% | 26.436734 |
| LightGBM 90% + CatBoost 10% | 26.436737 |
| LightGBM 80% + CatBoost 10% + XGBoost 10% | 26.436831 |

The ensemble experiments did not improve validation performance compared with LightGBM alone.

Therefore, LightGBM was selected for the final model.

## 6. Final Model Configuration

| Hyperparameter | Value |
|---|---|
| Algorithm | LightGBM Regressor |
| Estimators | 91 |
| Learning Rate | 0.03 |
| Number of Leaves | 31 |
| Maximum Depth | -1 |
| Feature Fraction | 0.9 |
| Row Subsample | 0.8 |
| Subsample Frequency | 1 |
| L2 Regularization | 5 |
| Random State | 42 |

The final model was retrained using all 524,164 training samples.

## 7. Final Predictions

The final trained model generated predictions for 174,722 test samples.

**Output file:** `submission_final.csv`

**Submission columns:**

- `id`
- `BeatsPerMinute`

The submission was checked for row count, column names, missing values, and non-finite predictions.

## 8. Results Summary

| Metric | Result |
|---|---|
| Best Algorithm | LightGBM |
| Validation RMSE | 26.436657 |
| Training Samples | 524,164 |
| Test Predictions | 174,722 |
| Kaggle Public Score | 26.38 |
| Kaggle Rank | Pending |

## 9. Conclusion

Three gradient boosting algorithms were trained and evaluated for beats-per-minute prediction.

LightGBM produced the lowest validation RMSE, outperforming CatBoost and XGBoost by a small margin.

Weighted ensembling did not provide additional improvements on the validation split.

The final LightGBM model was retrained on the complete training dataset and used to generate the competition submission.

## 10. Technologies

Python, Pandas, NumPy, Scikit-learn, LightGBM, CatBoost, XGBoost, Jupyter Notebook, VS Code, Git and GitHub.
