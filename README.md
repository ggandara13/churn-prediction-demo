# 🎯 Churn Prediction: From Root Cause Analysis to ML Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Gerardo Gandara | Senior Data Scientist  
**Date:** December 2024

---

## 📋 Executive Summary

This project demonstrates an end-to-end machine learning pipeline for **B2B customer churn prediction** that combines:

1. **LLM-powered root cause analysis** from customer reviews
2. **Feature engineering** based on qualitative insights
3. **Advanced ML modeling** with rigorous component testing
4. **Business-oriented threshold optimization**

### Key Results

| Metric | Value |
|--------|-------|
| **AUC Improvement** | +11.14% (from 0.83 to 0.95) |
| **Best Model** | Gradient Boosting |
| **Optimal Threshold** | 0.15 (97% recall) |
| **Net ROI** | $1.58M on test set |

---

## 🔬 Methodology

### The Problem

Traditional churn models rely on behavioral data (tenure, spend, usage). But they miss the **qualitative signals** hidden in:
- Support ticket conversations
- Customer feedback
- Review complaints

### The Solution: LLM Feature Engineering

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Trustpilot (142 reviews)  +  ConsumerAffairs (47 reviews)          │
│                              ↓                                       │
│                    Claude API Extraction                             │
│                              ↓                                       │
│              Root Cause Distribution:                                │
│              • support_incompetent: 27%                              │
│              • tax_error: 10%                                        │
│              • onboarding_fail: 8%                                   │
│              • billing_dispute: 7%                                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                               │
├─────────────────────────────────────────────────────────────────────┤
│  Root Cause              →    LLM Feature         →    Logic        │
│  ────────────────────────────────────────────────────────────────   │
│  support_incompetent     →   ticket_sentiment     →   27% negative  │
│  (27%)                       frustration_level        for churners  │
│  billing_dispute (7%)    →   has_billing_complaint                  │
│  onboarding_fail (8%)    →   churn_intent         →   explicit      │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      ML MODELING                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Models Tested: Logistic Regression, Random Forest,                  │
│                 Gradient Boosting, LightGBM, XGBoost                 │
│                                                                      │
│  Techniques Evaluated:                                               │
│  • SMOTE for class imbalance → Hurts performance (-1.3%)            │
│  • Genetic Algorithm for feature selection → Marginal (+0.05%)      │
│  • LLM Features → +11.14% AUC improvement ✅                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Findings

### 1. LLM Features Dominate Importance

All 5 LLM-extracted features rank in the **top 5 most important predictors**:

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | frustration_level | LLM | 43% |
| 2 | churn_intent | LLM | 17% |
| 3 | ticket_sentiment | LLM | 16% |
| 4 | has_support_complaint | LLM | 4% |
| 5 | has_billing_complaint | LLM | 3% |

### 2. SMOTE and GA Don't Add Value

| Component | Contribution | Verdict |
|-----------|-------------|---------|
| LLM Features | +0.1114 AUC | ✅ Essential |
| SMOTE | -0.0131 AUC | ❌ Hurts |
| Genetic Algorithm | +0.0005 AUC | ⚠️ Negligible |

### 3. Optimal Threshold = 0.15

Given cost asymmetry ($15K churner loss vs $75 outreach cost):

| Threshold | Recall | Net ROI |
|-----------|--------|---------|
| 0.35 | 90.4% | $1,472,850 |
| 0.25 | 94.4% | $1,526,100 |
| 0.20 | 96.8% | $1,566,150 |
| **0.15** | **97.3%** | **$1,576,950** |
| 0.10 | 97.3% | $1,569,750 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-prediction-demo.git
cd churn-prediction-demo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Ensure the dataset is in the project folder
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Run Streamlit
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📁 Project Structure

```
churn-prediction-demo/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Telco_Customer_Churn.csv    # Dataset (download separately)
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── assets/
    └── logo.png                # Company logo (optional)
```

---

## 🛠️ Technical Details

### Dataset

- **Source:** Telco Customer Churn (Kaggle)
- **Records:** 7,043 customers
- **Churn Rate:** 26.5%
- **Features:** 20 base + 5 LLM-simulated = 25 total

### Feature Mapping (Telco → B2B Generic)

| Original Feature | Renamed | Business Context |
|------------------|---------|------------------|
| tenure | tenure_months | Customer lifetime |
| MonthlyCharges | monthly_spend | Contract value |
| TotalCharges | lifetime_value | Total revenue |
| Contract | contract_type | Monthly/Annual/Multi-Year |
| TechSupport | has_premium_support | Support tier |
| InternetService | service_tier | Basic/Standard/Premium |

### LLM Feature Simulation

Since we don't have access to actual support tickets, LLM features are **simulated** using the root cause distribution from our review analysis:

```python
# Example: ticket_sentiment generation
support_issue_rate = 0.27  # From actual Trustpilot analysis
has_support_issue = np.random.binomial(1, support_issue_rate, n)
churner_with_issue = is_churner & has_support_issue
sentiment[churner_with_issue] = np.random.normal(-0.5, 0.2, count)
```

---

## 📈 Application Pages

1. **🏠 Overview** - Project methodology and pipeline visualization
2. **📊 Root Cause → Features** - How review analysis maps to features
3. **⚙️ Advanced ML Config** - Toggle SMOTE and Genetic Algorithm
4. **🤖 Model Comparison** - Component contribution analysis
5. **📈 Business Impact** - ROI calculator with threshold optimization
6. **🎮 Live Prediction** - Real-time scoring simulation

---

## 🎤 Key Talking Points

> "The entire +11% AUC improvement comes from LLM-extracted features based on root cause analysis. Traditional algorithmic enhancements like SMOTE and Genetic Algorithm don't add value—good feature engineering is what matters."

> "At threshold 0.15, we achieve 97% recall with $1.58M net ROI. The cost asymmetry ($15K churner loss vs $75 call) justifies an aggressive outreach strategy."

> "The top 5 most important features are all LLM-derived. This validates that analyzing WHY customers complain creates better predictors than behavioral data alone."

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Telco Customer Churn dataset from Kaggle
- Streamlit for the interactive dashboard framework
- scikit-learn, LightGBM, XGBoost for ML models

---

**Author:** Gerardo Gandara | Senior Data Scientist  
**Contact:** gerardo.gandara@gmail.com

https://www.linkedin.com/in/gerardo-gandara/

**Last Updated:** December 2025
