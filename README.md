# Writer Performance and Amnesia Prediction
This repository contains Python scripts for predicting writing times and amnesia error rates across multiple writers using advanced regression models, including Random Forest, XGBoost, LightGBM, and ensemble regressors. The project explores the relationship between cognitive load and handwriting characteristics, trained on multi-attempt handwriting data.

Project Structure

| File              | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `writing_time.py` | Predicts average writing time  for 106 writers using multiple regression models. |
| `amnesia.py`      | Predicts amnesia error rate  from feature data                  |


# Requirements
Install dependencies:
`pip install pandas numpy scikit-learn matplotlib xgboost lightgbm scipy`

1. Core Python Libraries:

   
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
matplotlib==3.9.2
seaborn==0.13.2
openpyxl==3.1.5

2. Machine Learning Models:


scikit-learn==1.5.1
xgboost==2.1.1
lightgbm==4.5.0
Optional for GPU-accelerated training (if supported)
cupy-cuda12x==13.2.0

3. Environment Information:

   
Python >= 3.10
OS: Ubuntu 22.04 / Windows 10+
GPU (optional): NVIDIA CUDA 12.x for accelerated XGBoost/LightGBM





# How to Run:

## 1. Writing Time Prediction

`python writing_time.py`

Outputs:



1. Regression plots for all models saved under Regression_Plots_All_Models/
2. CSV files with actual vs. predicted writing times
3. Console summary of RMSE and R² scores for all models 

## 2. Amnesia Prediction (Amnesia_all and Amnesia_1)

`python amnesia_all.py`
`python amnesia_1.py`

Output for both:



1. Console metrics: MAE, MSE, RMSE, R²
2. Sorted performance summary by R² score
3. Regression plots in regression4_plots/ (auto-created)
4. Best model printed in the terminal

# Model & Hyperparameters
All regression models are executed automatically in a single run — no manual selection is required. The scripts train, evaluate, and compare all models at once, and automatically report the best-performing model based on R².

Models Used 

| Model                                          | Key Hyperparameters                                        |
| ---------------------------------------------- | ---------------------------------------------------------- |
|**[Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)** | `fit_intercept=True`, `copy_X=True`, `tol=1e-06`, `n_jobs=None`, `positive=False` |
| **[Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) | `alpha=[0.1, 1, 5, 10, 20, 50, 100]`, `fit_intercept=True`, `copy_X=True`, `max_iter=None`, `tol=0.0001`, `solver='auto'`, `positive=False`, `random_state=None` |
| **[Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)**| `alpha=[0.0001, 0.001, 0.01, 0.1, 1]`, `fit_intercept=True`, `precompute=False`, `copy_X=True`, `max_iter=5000`, `tol=0.0001`, `warm_start=False`, `positive=False`, `random_state=None`, `selection='cyclic'` |
| **[ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)** | `alpha=[0.0001, 0.001, 0.01, 0.1, 1]`, `l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]`, `fit_intercept=True`, `precompute=False`, `max_iter=5000`, `copy_X=True`, `tol=0.0001`, `warm_start=False`, `positive=False`, `random_state=None`, `selection='cyclic'` |
| **[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)** | `kernel=['rbf', 'linear']`, `degree=3`, `gamma='scale'`, `coef0=0.0`, `tol=0.001`, `C=[0.1, 1, 5, 10]`, `epsilon=[0.01, 0.05, 0.1]`, `shrinking=True`, `cache_size=200`, `verbose=False`, `max_iter=-1` |
| **[KNNR](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)** | `n_neighbors=[3, 5, 7, 9]`, `weights=['uniform', 'distance']`, `algorithm='auto'`, `leaf_size=30`, `p=2`, `metric='minkowski'`, `metric_params=None`, `n_jobs=None` |
| **[RFR](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)** | `n_estimators=[100, 200]`, `criterion='squared_error'`, `max_depth=[None, 3, 5, 10]`, `min_samples_split=2`, `min_samples_leaf=1`, `min_weight_fraction_leaf=0.0`, `max_features=1.0`, `max_leaf_nodes=None`, `min_impurity_decrease=0.0`, `bootstrap=True`, `oob_score=False`, `n_jobs=None`, `random_state=42`, `verbose=0`, `warm_start=False`, `ccp_alpha=0.0`, `max_samples=None`, `monotonic_cst=None` |
| **[GBR](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)** | `loss='squared_error'`, `learning_rate=[0.03, 0.05, 0.1]`, `n_estimators=[100, 150, 200]`, `subsample=1.0`, `criterion='friedman_mse'`, `min_samples_split=2`, `min_samples_leaf=1`, `min_weight_fraction_leaf=0.0`, `max_depth=[2, 3]`, `min_impurity_decrease=0.0`, `init=None`, `random_state=42`, `max_features=None`, `alpha=0.9`, `verbose=0`, `max_leaf_nodes=None`, `warm_start=False`, `validation_fraction=0.1`, `n_iter_no_change=None`, `tol=0.0001`, `ccp_alpha=0.0` |
| **[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)** | `estimator=None`, `n_estimators=[50, 100, 200]`, `learning_rate=[0.01, 0.05, 0.1, 0.5]`, `loss='linear'`, `random_state=42` |
| **[XGBoost](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)** | `n_estimators=[150, 200, 300, 400]`, `learning_rate=[0.01, 0.03, 0.05]`, `max_depth=[2, 3, 4]`, `min_child_weight=[1, 3, 5]`, `subsample=[0.8, 0.85, 1.0]`, `colsample_bytree=[0.8, 0.85, 1.0]`, `gamma=[0.0, 0.1, 0.3]`, `reg_alpha=[0.0, 0.01, 0.1]`, `reg_lambda=[1.0, 3.0, 5.0]`, `random_state=42`, `objective='reg:squarederror'`, `verbosity=0`, `tree_method='hist'`, `n_jobs=-1` |
| **[LightGBM](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)** | `boosting_type='gbdt'`, `num_leaves=[15, 31, 63]`, `max_depth=-1`, `learning_rate=[0.03, 0.05, 0.1]`, `n_estimators=[100, 200]`, `subsample_for_bin=200000`, `objective=None`, `class_weight=None`, `min_split_gain=0.0`, `min_child_weight=0.001`, `min_child_samples=20`, `subsample=1.0`, `subsample_freq=0`, `colsample_bytree=1.0`, `reg_alpha=0.0`, `reg_lambda=0.0`, `random_state=42`, `n_jobs=None`, `importance_type='split'` |
| **[Voting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)** | `estimators`, `weights=None`, `n_jobs=None`, `verbose=False` |

Note: The table reports only those hyperparameters that were modified from their default settings in Scikit-learn; all other hyperparameters remain identical to those described in Pedregosa et al., “_Scikit-learn: Machine Learning in Python_,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
# Model Training Details
1. Standardization: All features except WriterId are standardized using StandardScaler.

2. Splits:
Writing time: Train/Val/Test words = 86:20
Amnesia: 80 % train, 10 % val, 10 % test

3. Evaluation Metrics:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)

# Visualization Outputs
Regression Scatter Plots:
X-axis → Actual values
Y-axis → Predicted values
Blue line → Regression fit
Red dashed line → Theoretical diagonal (ideal prediction)

# Results Summary
Each model’s RMSE and R² are printed in sorted order.
The model achieving the highest R² is automatically selected as the best-performing model for further predictions.
# Dataset 
The dataset used for the experiments is uploaded here.


