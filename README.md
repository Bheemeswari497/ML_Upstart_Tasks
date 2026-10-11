# ML_Upstart_Tasks

Repository for UpStartInterns Machine Learning Engineering tasks.

## Tasks Directory

### [Task 01: Simple Linear Regression by Hand](Task-1/simple_regression/README.md)
- **Problem Statement:** Fit a Simple Linear Regression model to real-world advertising data, evaluate underlying model assumptions through residual diagnostics, and interpret model parameters in plain English.
- **Dataset:** Real-world Advertising Dataset (`dataset.csv`, $n = 200$).
- **Methodology:** Closed-form Ordinary Least Squares (OLS) mathematical derivation and calculation purely by hand without `scikit-learn` to fit the model.
- **Verification:** Independent verification against `scikit-learn.linear_model.LinearRegression`.
- **Fitted Model:** $\widehat{\text{Sales}} = 7.0326 + 0.0475 \times \text{TV}$ with $R^2 = 0.6119$.
- **Evaluation Metrics:** $\text{MAE} = 2.5498$, $\text{MSE} = 10.5127$, $\text{RMSE} = 3.2423$, $R^2 = 0.6119$.
- **Deliverables:** Standalone script [`Task-1/simple_regression/simple_regression.py`](Task-1/simple_regression/simple_regression.py), executed notebook [`Task-1/simple_regression/simple_regression.ipynb`](Task-1/simple_regression/simple_regression.ipynb), detailed task documentation [`Task-1/simple_regression/README.md`](Task-1/simple_regression/README.md), and generated figures inside [`Task-1/simple_regression/figures/`](Task-1/simple_regression/figures/).