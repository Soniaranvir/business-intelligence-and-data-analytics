# 📊 Customer Analytics & Predictive Modeling

This repository contains a collection of Jupyter notebooks applying machine learning and statistical techniques to customer analytics, churn prediction, sales forecasting, and A/B testing using real-world datasets.

---

## 📌 Table of Contents
- [📝 Project Descriptions](#-project-descriptions)
- [📂 Datasets](#-datasets)
- [🛠️ Technologies Used](#️-technologies-used)
- [🚀 Getting Started](#-getting-started)
- [📜 License](#-license)

---

## 📝 Project Descriptions

### 1️⃣ Revenue & User Retention Calculation
**📌 Objective:** Analyze revenue trends & user retention.  
**✅ Key Tasks:**
- Calculate Monthly Recurring Revenue (MRR), Churn Rate, and Customer Retention Rate (CRR).
- Implement Cohort Analysis to visualize retention patterns.
- Use SQL & Pandas for data aggregation.

### 2️⃣ Customer Segmentation
**📌 Objective:** Identify customer groups using clustering techniques.  
**✅ Key Tasks:**
- Perform RFM Analysis (Recency, Frequency, Monetary Value).
- Apply K-Means, Hierarchical Clustering, and DBSCAN.
- Use PCA & t-SNE for dimensionality reduction.

### 3️⃣ Customer Lifetime Value (CLV) Prediction
**📌 Objective:** Predict long-term customer value.  
**✅ Key Tasks:**
- Use Gamma-Gamma & Beta-Geometric Models for CLV estimation.
- Build XGBoost Regression Models for CLV prediction.
- Engineer features from transaction history.

### 4️⃣ Churn Prediction
**📌 Objective:** Identify customers likely to churn.  
**✅ Key Tasks:**
- Train Logistic Regression, Decision Trees, and XGBoost models.
- Evaluate models using ROC-AUC, Precision-Recall, and F1-score.
- Handle class imbalance with SMOTE & resampling techniques.

### 5️⃣ Predicting Next Purchase Day
**📌 Objective:** Forecast customer purchase behavior.  
**✅ Key Tasks:**
- Use Time Series Models (ARIMA, Prophet, LSTM).
- Extract rolling statistics & lag features.
- Identify seasonality and trends.

### 6️⃣ Sales Prediction
**📌 Objective:** Forecast future sales trends.  
**✅ Key Tasks:**
- Train Random Forest, XGBoost, and Regression Models.
- Handle missing values & outliers in sales data.
- Perform hyperparameter tuning for better accuracy.

### 7️⃣ Market Response Models
**📌 Objective:** Measure marketing impact on revenue.  
**✅ Key Tasks:**
- Implement Marketing Mix Models (MMM) & Price Elasticity Analysis.
- Train Multivariate Regression models.
- Evaluate advertising spend efficiency.

### 8️⃣ Uplift Modeling
**📌 Objective:** Identify customers most likely to respond to marketing efforts.  
**✅ Key Tasks:**
- Train Two-Model Uplift, KL Divergence, and Tree-Based Uplift Models.
- Apply Causal Inference Techniques (EconML, DoWhy).
- Measure incremental impact of marketing campaigns.

### 9️⃣ A/B Testing Design & Execution
**📌 Objective:** Test business strategies using controlled experiments.  
**✅ Key Tasks:**
- Implement Bayesian vs Frequentist A/B Testing Approaches.
- Compute p-values, effect sizes, and confidence intervals.
- Apply Sequential Testing & Multi-Armed Bandits.

---

## 📂 Datasets

| Dataset                          | Description                                                   |
|----------------------------------|---------------------------------------------------------------|
| `Kevin_Hillstrom_MineThatData.csv` | E-commerce marketing campaign data for uplift modeling.       |
| `OnlineRetail.csv`               | Retail transaction data for customer segmentation & CLV.      |
| `churn_data.csv`                 | Customer churn dataset for classification models.             |
| `market_response_model_data.csv` | Sales and marketing data for market response models.          |
| `sales_prediction_train.csv` / `sales_prediction_test.csv` | Time-series sales data for forecasting. |

---

## 🛠️ Technologies Used

| Category                | Tools & Libraries                                     |
|-------------------------|-------------------------------------------------------|
| **Programming**         | Python (Jupyter Notebooks)                           |
| **Data Processing**     | Pandas, NumPy, SQL                                   |
| **Visualization**       | Matplotlib, Seaborn, Plotly                          |
| **Machine Learning**    | Scikit-learn, XGBoost, LightGBM, CatBoost            |
| **Time Series Analysis**| Statsmodels, Facebook Prophet, TensorFlow (LSTM)    |
| **Statistical Modeling**| SciPy, Statsmodels, Lifetimes                        |
| **A/B Testing & Causal Inference** | EconML, DoWhy, PyMC3                     |

---

