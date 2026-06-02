# 🛡️ Institutional Credit Risk Control Center & Underwriting Pipeline

[![Streamlit App](https://streamlit.io)](https://credit-risk-dashboard-hj7br3sxb8nmwu8g3saehk.streamlit.app/) 
![Python](https://shields.io)
![Framework](https://shields.io)
![Compliance](https://shields.io)

An end-to-end, production-grade credit risk modeling and portfolio telemetry ecosystem engineered to bridge the gap between machine learning and banking operations. This system automates the acquisition, underwriting, and portfolio tracking lifecycles while enforcing strict credit policy constraints and regulatory compliance frameworks.

**✨ Created & Engineered by:** Srinivasta

---

## 🎯 Core Project Capabilities (Fulfilling the JD)

### 📈 1. Scorecard Development & Underwriting
* Developed traditional underwriting scorecards using **Logistic Regression** and **Weight of Evidence (WoE)** binning logic.
* Applied **Information Value (IV)** filtering to optimize feature selection, mathematically establishing `income_bin` as a primary driver (IV: `0.3775`) over `amount_bin` (IV: `0.0521`).
* Scaled model log-odds into standard integer points using a Point-to-Double (PDO) configuration of `20` and a base anchor score of `524 Points` for clear corporate sizing parameters.

### 🚨 2. Early Warning Systems (EWS)
* Built a predictive Early Warning Model utilizing an optimized **XGBoost Classifier** architecture to flag early signs of consumer portfolio stress.
* Formulated data diagnostics that isolated debt-to-income leverage (`loan_percent_income`) as the dominant trigger—accounting for **58.52%** of the system's decision weight—followed by risk-premium pricing (`loan_int_rate` at **31.94%**).

### ⚔️ 3. Challenger Modeling & Stress Testing
* Maintained portfolio stability by constantly stress testing the baseline underwriting models against an experimental, high-capacity **Random Forest Challenger** architecture.
* Validating across an Out-of-Time (OOT) test segment, the Challenger model achieved an **AUC of 0.8757** vs the Champion's **0.8454 AUC**, delivering an **absolute lift of +0.0303** (+0.06 Gini lift) to automatically trigger a production shadow queue migration.

### 📊 4. Automated Data Telemetry Pipelines
* Architected a data pipeline that continuously tracks population data drift in real-time application queues.
* Computed a live **Population Stability Index (PSI) of 0.0413**, triggering a **GREEN / STABLE** health status on the telemetry dashboard. This stability validation automatically bypasses unnecessary model-retraining loops, saving on cloud computing server costs.

### 🤝 5. Cross-Functional API Gateway & Policy Integration
* Integrated machine learning probabilities with hard cutoff limits mandated by the credit policy committee to form a real-time **Automated Underwriting API Gateway**.
* Evaluated a 3,259 application testing batch under standard constraints (1-year employment floor, 40% maximum debt load, 25% AI risk cap) to approve **59.13% (1,927 apps)** and decline **40.87% (1,332 apps)**, separating rejections transparently by root cause.

### ⚖️ 6. Regulatory Audit Traceability (Explainable AI)
* Integrated the **SHAP (SHapley Additive exPlanations)** framework to eliminate "black-box" model scoring liabilities.
* Programmed an adverse action engine that prints an itemized, legally defensible risk receipt for individual applicants (e.g., *Profile ID: 6*), highlighting exactly how much each asset increased or decreased their Probability of Default (PD).

---

## 📊 Dataset Attribution & Acknowledgements
The core portfolio simulation records utilized in this project are sourced from the **[Kaggle Credit Risk Dataset](https://kaggle.com)**. Special credit and appreciation are extended to the dataset creators and the **Kaggle Data Science Community** for making these historical banking parameters publicly available, enabling the development of robust open-source risk frameworks.

---

## 💻 Tech Stack & Engineering Tools
* **Core Language:** Python 3.9+
* **Data Pipelines:** Pandas, NumPy, Scikit-Learn
* **Advanced ML Architectures:** XGBoost, Random Forest Ensemble
* **Explainable AI:** SHAP (TreeExplainer)
* **Production UI & Telemetry:** Streamlit Engine
* **Compliance Document Engine:** ReportLab (Automated PDF Generation)

---

## 🗂️ Repository Architecture
```text
├── app.py                         # Main Streamlit Desktop Application Script
├── requirements.txt               # Manifest of system cloud dependencies
├── credit_risk_dataset.csv        # Baseline portfolio simulation history dataset
└── README.md                      # Project architecture and technical specification
```

---

## 🚀 Local Installation & Deployment

To launch this interactive control panel dashboard locally on your desktop machine, execute the following steps in your terminal environment:

1. Clone this repository to your local directory:
   ```bash
   git clone https://github.com
   cd credit-risk-dashboard
   ```

2. Install the necessary machine learning and framework dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize and run the local desktop application instance:
   ```bash
   streamlit run app.py
   ```
4. Access the live interface by navigating your standard web browser to `http://localhost:8501`.

---

## 📋 Production Operational Summary (Live Dashboard View)
* **Total Portfolio Records Processed:** 3,259 Live Profiles
* **Portfolio Approval Volume:** 2,067 (63.4%) 🟢
* **Portfolio Rejection Volume:** 1,192 (36.6%) 🔴
* **Data Population Inflow Index:** 0.0413 PSI (System Normal / Distribution Stable)
* **Primary System Rejection Trigger:** Algorithmic Risk Limit Threshold Breeches (AI Model PD >= 25%)

---
*Developed as a data-driven risk framework by **Srinivasta** to optimize consumer acquisition underwriting while keeping institutional default rates securely managed.*
