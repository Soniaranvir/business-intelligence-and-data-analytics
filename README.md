# # Customer Analytics & Predictive Modeling

This repository contains a collection of Jupyter notebooks implementing machine learning and statistical techniques for customer analytics. The projects focus on customer segmentation, retention analysis, churn prediction, sales forecasting, uplift modeling, and A/B testing, using real-world datasets.

# #📌 Table of Contents

-📊 Project Descriptions
-📂 Datasets
-🛠️ Technologies Used
-🚀 Getting Started
-📜 License
-📊 Project Descriptions

1️⃣ Revenue & User Retention Calculation
-Computes Monthly Recurring Revenue (MRR), Customer Retention Rate (CRR), and Churn Rate using historical transaction data.
-Uses cohort analysis to identify retention trends over time.
-Techniques: SQL, Pandas, Data Visualization (Matplotlib, Seaborn).

2️⃣ Customer Segmentation
-Applies K-Means and Hierarchical Clustering to segment customers based on purchase frequency, recency, and monetary value (RFM Analysis).
-Implements Dimensionality Reduction (PCA, t-SNE) for visualization.
-Techniques: Scikit-learn, K-Means++, Silhouette Score, DBSCAN.

3️⃣ Customer Lifetime Value (CLV) Prediction
-Predicts Customer Lifetime Value using Gamma-Gamma & Beta-Geometric models and Machine Learning (XGBoost, Random Forest).
-Feature engineering includes transaction frequency, order value trends, and churn probability estimates.
-Techniques: Lifetimes Python Library, Regression Models, Bayesian Statistics.

4️⃣ Churn Prediction
-Develops a Supervised Learning model to identify customers likely to churn.
-Compares Logistic Regression, Decision Trees, and Gradient Boosting (XGBoost, CatBoost, LightGBM) for performance.
-Evaluates model performance using ROC-AUC, Precision-Recall, and F1-score metrics.

5️⃣ Predicting Next Purchase Day
-Uses Time Series Forecasting (ARIMA, LSTM, Prophet) to predict when a customer will make their next purchase.
-Incorporates seasonality decomposition, lag features, and rolling statistics for better forecasting.
-Techniques: Statsmodels, Facebook Prophet, TensorFlow (LSTM).

6️⃣ Sales Prediction
-Implements Linear Regression, Random Forest, and XGBoost for predicting future sales trends.
-Handles data imbalances, missing values, and outliers to improve model accuracy.
-Techniques: Time Series Analysis, Feature Engineering, Hyperparameter Tuning.

7️⃣ Market Response Models
-Builds econometric models to evaluate the impact of marketing efforts on revenue.
-Implements Multivariate Regression, Marketing Mix Models (MMM), and Price Elasticity Models.
-Techniques: Pandas, NumPy, Scikit-learn, Statsmodels.

8️⃣ Uplift Modeling
-Implements Causal Inference and Uplift Modeling to measure the impact of marketing campaigns.
-Compares Two-Model Approach, KL Divergence, and Tree-Based Uplift Models.
-Techniques: EconML, Scikit-learn, XGBoost, Treatment Effect Estimation.

9️⃣ A/B Testing Design & Execution
-Designs and analyzes controlled experiments (A/B, Multivariate, Sequential Testing).
-Uses Bayesian vs Frequentist methods to compare test groups.
-Evaluates statistical significance (p-values, confidence intervals, effect sizes).
-Techniques: SciPy, Statsmodels, Bayesian Inference (PyMC3).

# # 📂 Datasets

The repository includes real-world datasets for each project:

-Dataset	Description
-Kevin_Hillstrom_MineThatData.csv	Email marketing campaign dataset for Uplift Modeling.
-OnlineRetail.csv	E-commerce transaction data for Customer Segmentation & CLV.
-churn_data.csv	Customer churn dataset for Churn Prediction.
-market_response_model_data.csv	Data for Market Response Modeling.
-sales_prediction_train.csv / sales_prediction_test.csv	Sales forecasting dataset.

# # 🛠️ Technologies Used

This project leverages the following libraries & frameworks:

-Category	Technologies
-Programming	Python (Jupyter Notebooks)
-Data Processing	Pandas, NumPy, SQL
-Visualization	Matplotlib, Seaborn, Plotly
-Machine Learning	Scikit-learn, XGBoost, LightGBM, CatBoost
-Time Series Analysis	Statsmodels, Facebook Prophet, LSTM (TensorFlow/Keras)
-Statistical Modeling	SciPy, Statsmodels, Lifetimes, Bayesian Inference (PyMC3)
-A/B Testing & Causal Inference	EconML, DoWhy, PyMC3

# # 🚀 Getting Started

1️⃣ Clone Repository
git clone https://github.com/your-username/customer-analytics.git
cd customer-analytics
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Jupyter Notebook
jupyter notebook
Open any .ipynb file to explore the analysis.
