# Writer Performance and Amnesia Prediction
This repository contains Python scripts for predicting writing times and amnesia error rates across multiple writers using advanced regression models such as Random Forest, XGBoost, LightGBM, and ensemble regressors. The project explores the relationship between cognitive load and handwriting characteristics, trained on multi-attempt handwriting data.

Project Structure

| File              | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `writing_time.py` | Predicts average writing time per word for 106 writers using multiple regression models. |
| `amnesia_all.py`  | Predicts overall amnesia error rate (`Amnesia_all`) from feature data.                   |
| `amnesia_1.py`    | Predicts single-attempt amnesia error rate (`Amnesia1`).                                 |

#Requirements
Install dependencies:
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm scipy

#How to Run:

1. Writing Time Prediction

python writing_time.py

Outputs:
Regression plots for all models saved under Regression_Plots_All_Models/
CSV files with actual vs. predicted writing times
Console summary of RMSE and R² scores for all models 

2. Amnesia Prediction (Amnesia_all and Amnesia_1)

python amnesia_all.py
python amnesia_1.py

Output for both:

Console metrics: MAE, MSE, RMSE, R²
Sorted performance summary by R² score
Regression plots in regression4_plots/ (auto-created)
Best model printed in terminal

#Model & Hyperparameters
All regression models are executed automatically in a single run — no manual selection is required. The scripts train, evaluate, and compare all models at once, and automatically report the best-performing model based on R².

Models Used 

| Model                                          | Key Hyperparameters                                        |
| ---------------------------------------------- | ---------------------------------------------------------- |
| **RandomForestRegressor**                      | `n_estimators=100`, `random_state=42`                      |
| **GradientBoostingRegressor**                  | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3`     |
| **AdaBoostRegressor**                          | `random_state=42`                                          |
| **XGBoostRegressor**                           | `random_state=42`                                          |
| **LightGBMRegressor**                          | `n_estimators=200`, `learning_rate=0.1`, `random_state=42` |
| **KNeighborsRegressor**                        | Default parameters                                         |
| **SVR**                                        | `kernel='rbf'`, `C=1.0`, `epsilon=0.1`                     |
| **MLPRegressor**                               | `max_iter=2000`, `random_state=42`                         |
| **ElasticNet, Ridge, Lasso, LinearRegression** | Default regularization parameters                          |
| **VotingRegressor**                            | Combination of (Gradient Boosting, Random Forest, XGBoost) |

#Model Training Details
1. Standardization: All features except WriterId are standardized using StandardScaler.

2. Splits:
Writing time: Train/Val/Test words = 18 : 6 : 8
Amnesia: 70 % train, 10 % val, 20 % test

3. Evaluation Metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)

#Visualization Outputs
Regression Scatter Plots:
X-axis → Actual values
Y-axis → Predicted values
Blue line → Regression fit
Red dashed line → Theoretical diagonal (ideal prediction)

#Results Summary
Each model’s RMSE and R² are printed in sorted order.
The model achieving the highest R² is automatically selected as the best performing model for further predictions.

#Author Notes
1. Core Python Libraries
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
matplotlib==3.9.2
seaborn==0.13.2
openpyxl==3.1.5

2. Machine Learning Models
scikit-learn==1.5.1
xgboost==2.1.1
lightgbm==4.5.0
Optional for GPU-accelerated training (if supported)
cupy-cuda12x==13.2.0

3. Environment Information
Python >= 3.10
OS: Ubuntu 22.04 / Windows 10+
GPU (optional): NVIDIA CUDA 12.x for accelerated XGBoost/LightGBM
