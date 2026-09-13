# 💳 SME Credit Scoring and Loan Eligibility System

An AI-powered prototype for assessing the credit risk and loan eligibility of Small and Medium Enterprises (SMEs) using machine learning, financial indicators, banking behavior, GST-related indicators, business characteristics, and local market factors.

## 🚀 Live Application

👉 [Open the Live SME Credit Assessment System](https://sme-credit-scoring.streamlit.app/)

👉 [Open the Google Colab Notebook](https://colab.research.google.com/drive/1fO-dj--8ujRpMGSfrOJxn8u9uXynfIFW?usp=sharing)

The application is deployed using Streamlit Community Cloud and can be accessed through a public `streamlit.app` URL.

## 📌 Project Overview

Small and Medium Enterprises often have limited access to traditional credit due to insufficient or fragmented credit history. This project develops a machine-learning-based prototype that evaluates SME financial and business characteristics to estimate credit risk and provide an indicative loan recommendation.

The system combines:

* Business characteristics
* Financial performance
* Cash-flow stability
* Existing liabilities
* Banking and transaction behavior
* GST compliance indicators
* Digital presence
* Online reputation
* Industry and location characteristics

The final system produces a project-specific credit score, risk category, eligibility decision, indicative loan recommendation, and SHAP-based model explanation.

## 🎯 Objectives

* Develop an SME credit-risk prediction model.
* Convert model-estimated risk into a project-specific credit score.
* Categorize SMEs according to their estimated credit risk.
* Apply rule-based eligibility checks.
* Calculate an indicative loan recommendation.
* Explain model predictions using SHAP.
* Deploy the complete application through Streamlit.

## 🧠 Machine Learning Model

The project evaluates multiple classification models and uses **XGBoost** as the final model.

The final model uses **46 engineered and encoded features** covering financial, operational, banking, GST, digital, industry, and location-related information.

### Model outputs

The system produces:

1. Model-estimated default risk
2. Credit score from 300–900
3. Risk category
4. Loan eligibility decision
5. Business-rule warnings
6. Indicative recommended loan amount
7. SHAP-based explanation of important risk factors

## 📊 Credit Risk Categories

| Credit Score | Risk Category  |
| -----------: | -------------- |
|      750–900 | Excellent      |
|      650–749 | Good           |
|      550–649 | Moderate       |
|      450–549 | High Risk      |
|      300–449 | Very High Risk |

The 300–900 score is a **project-specific scoring system** and is not an official CIBIL, bureau, or banking credit score.

## 🏦 Eligibility Rules

In addition to the machine-learning prediction, the prototype applies business rules including:

* Maximum debt service ratio
* Minimum business age
* Minimum GST filing regularity
* Non-negative net monthly cash flow

The system combines the model-based risk category with these business rules to determine the final prototype eligibility status.

## 💰 Loan Recommendation

The system calculates an indicative loan recommendation using:

* Monthly revenue
* Existing monthly debt obligation
* Maximum allowable debt-service level
* Assumed interest rate
* Assumed loan tenure
* Risk-category adjustment

The recommendation is **indicative only** and must not be interpreted as an actual loan approval or underwriting decision.

## 🔎 Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to identify the features that contributed most strongly to an individual prediction.

The Streamlit interface presents these as:

* 🔴 Factors increasing estimated default risk
* 🟢 Factors reducing estimated default risk

This improves transparency and makes the model output easier to interpret.

## 🗂️ Project Structure

```text
SME-Credit-Scoring/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│   ├── sme_credit_xgb.json
│   ├── scoring_configuration.json
│   ├── model_metadata.json
│   ├── model_feature_columns.json
│   ├── industry_table.json
│   └── location_table.json
│
└── src/
    ├── __init__.py
    └── backend.py
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Git & GitHub

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/veetesh27/SME-Credit-Scoring.git
cd SME-Credit-Scoring
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** from the GitHub repository.

After deployment, the application is accessible through a public `streamlit.app` URL.

## ⚠️ Limitations and Disclaimer

This project is an educational and demonstration prototype.

The dataset used for model development is **synthetic** and does not represent real bank customers, GST records, businesses, or actual loan outcomes.

Therefore:

* Model predictions should not be treated as real-world default probabilities.
* The credit score is not an official credit-bureau score.
* Loan recommendations are indicative and not approved loan amounts.
* The system is not intended for real banking or lending decisions.
* Production deployment would require validated real-world data, regulatory review, model validation, calibration, fairness testing, monitoring, security controls, and appropriate underwriting policies.

## 👨‍💻 Author

Developed as an AI/ML project demonstrating SME credit-risk assessment, explainable machine learning, rule-based eligibility, and Streamlit deployment.
