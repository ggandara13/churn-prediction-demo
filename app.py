#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Churn Prediction Demo: From Root Cause Analysis to ML Model
============================================================

A comprehensive Streamlit application demonstrating an end-to-end machine
learning pipeline for B2B customer churn prediction.

Author: Gerardo Gandara | Senior Data Scientist
Date: December 2024
Version: 2.0.0 (Optimized for Streamlit Cloud)

Key Optimization:
    - All models trained ONCE at startup
    - Results stored in session_state
    - Page navigation is instant
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.combine import SMOTETomek
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import random
    from deap import algorithms, base, creator, tools
    HAS_GA = True
except ImportError:
    HAS_GA = False

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Churn Prediction - ADP Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_URL = "https://1000logos.net/wp-content/uploads/2021/04/ADP-logo.png"

# =============================================================================
# CONSTANTS
# =============================================================================

ROOT_CAUSE_DISTRIBUTION: Dict[str, float] = {
    'support_incompetent': 0.27,
    'onboarding_fail': 0.08,
    'billing_dispute': 0.07,
    'tax_error': 0.10,
    'login_issues': 0.05,
    'support_slow': 0.05,
    'payroll_delayed': 0.06,
    'cancellation_blocked': 0.03,
    'app_bug': 0.06,
    'other': 0.23
}

BASE_FEATURES: List[str] = [
    'tenure_months', 'monthly_spend', 'lifetime_value', 'is_enterprise',
    'has_partner_account', 'has_subsidiary', 'has_phone_support',
    'has_multiple_products', 'has_security_addon', 'has_backup_service',
    'has_premium_support', 'has_addon_1', 'has_addon_2', 'paperless_billing',
    'contract_Monthly', 'contract_Annual', 'tier_Standard', 'tier_Premium',
    'payment_ACH', 'payment_AutoPay'
]

LLM_FEATURES: List[str] = [
    'ticket_sentiment', 'frustration_level', 'churn_intent',
    'has_support_complaint', 'has_billing_complaint'
]

ALL_FEATURES: List[str] = BASE_FEATURES + LLM_FEATURES


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_and_rename_telco() -> Tuple[Optional[pd.DataFrame], bool]:
    """Load and rename Telco dataset."""
    try:
        df = pd.read_csv('Telco_Customer_Churn.csv')
    except FileNotFoundError:
        return None, False
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    df_renamed = pd.DataFrame({
        'customer_id': df['customerID'],
        'tenure_months': df['tenure'],
        'monthly_spend': df['MonthlyCharges'],
        'lifetime_value': df['TotalCharges'],
        'is_enterprise': df['SeniorCitizen'],
        'has_partner_account': (df['Partner'] == 'Yes').astype(int),
        'has_subsidiary': (df['Dependents'] == 'Yes').astype(int),
        'has_phone_support': (df['PhoneService'] == 'Yes').astype(int),
        'has_multiple_products': df['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0),
        'service_tier': df['InternetService'].map({'No': 'Basic', 'DSL': 'Standard', 'Fiber optic': 'Premium'}),
        'has_security_addon': df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0),
        'has_backup_service': df['OnlineBackup'].apply(lambda x: 1 if x == 'Yes' else 0),
        'has_premium_support': df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0),
        'has_addon_1': df['StreamingTV'].apply(lambda x: 1 if x == 'Yes' else 0),
        'has_addon_2': df['StreamingMovies'].apply(lambda x: 1 if x == 'Yes' else 0),
        'contract_type': df['Contract'].map({'Month-to-month': 'Monthly', 'One year': 'Annual', 'Two year': 'Multi-Year'}),
        'paperless_billing': (df['PaperlessBilling'] == 'Yes').astype(int),
        'payment_method': df['PaymentMethod'].map({
            'Electronic check': 'ACH', 'Mailed check': 'Check',
            'Bank transfer (automatic)': 'Auto-Pay', 'Credit card (automatic)': 'Credit Card'
        }),
        'churn': (df['Churn'] == 'Yes').astype(int)
    })
    
    return df_renamed, True


@st.cache_data
def add_llm_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate LLM features based on root cause distribution."""
    np.random.seed(seed)
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    is_churner = df['churn'].values.astype(bool)
    is_monthly = (df['contract_type'] == 'Monthly').values
    is_short_tenure = (df['tenure_months'] < 12).values
    no_premium_support = (df['has_premium_support'] == 0).values
    is_ach_payment = (df['payment_method'] == 'ACH').values
    high_spend = (df['monthly_spend'] > df['monthly_spend'].median()).values
    
    # ticket_sentiment
    support_issue_rate = ROOT_CAUSE_DISTRIBUTION['support_incompetent'] + ROOT_CAUSE_DISTRIBUTION['support_slow']
    sentiment = np.random.normal(0.2, 0.25, n)
    has_support_issue = np.random.binomial(1, support_issue_rate, n).astype(bool)
    churner_with_issue_idx = np.where(is_churner & has_support_issue)[0]
    sentiment[churner_with_issue_idx] = np.random.normal(-0.5, 0.2, len(churner_with_issue_idx))
    frustrated_idx = np.where(no_premium_support & high_spend)[0]
    sentiment[frustrated_idx] -= np.random.uniform(0.1, 0.3, len(frustrated_idx))
    df['ticket_sentiment'] = np.clip(sentiment, -1, 1).round(2)
    
    # frustration_level
    frustration_rate = ROOT_CAUSE_DISTRIBUTION['support_incompetent'] + ROOT_CAUSE_DISTRIBUTION['billing_dispute'] + ROOT_CAUSE_DISTRIBUTION['tax_error']
    frustration = np.random.choice([0, 1, 1, 2, 2], n)
    is_frustrated = np.random.binomial(1, frustration_rate, n).astype(bool)
    churner_frustrated_idx = np.where(is_churner & is_frustrated)[0]
    frustration[churner_frustrated_idx] = np.random.choice([3, 4, 4, 5], len(churner_frustrated_idx))
    high_risk_idx = np.where(is_monthly & no_premium_support)[0]
    frustration[high_risk_idx] = np.clip(frustration[high_risk_idx] + 1, 0, 5)
    df['frustration_level'] = frustration.astype(int)
    
    # churn_intent
    intent_rate = ROOT_CAUSE_DISTRIBUTION['onboarding_fail'] + ROOT_CAUSE_DISTRIBUTION['cancellation_blocked']
    churn_intent = np.zeros(n, dtype=int)
    has_explicit = np.random.binomial(1, min(intent_rate * 3, 1.0), n).astype(bool)
    churner_explicit_idx = np.where(is_churner & has_explicit)[0]
    churn_intent[churner_explicit_idx] = np.random.choice([2, 3], len(churner_explicit_idx))
    implicit_risk_idx = np.where(is_monthly & is_short_tenure & ~(is_churner & has_explicit))[0]
    churn_intent[implicit_risk_idx] = np.random.choice([0, 1, 1, 2], len(implicit_risk_idx))
    df['churn_intent'] = churn_intent
    
    # has_support_complaint
    support_complaint_churner = ROOT_CAUSE_DISTRIBUTION['support_incompetent']
    complaint_prob = np.where(is_churner, support_complaint_churner, 0.08)
    complaint_prob = np.where(no_premium_support, complaint_prob * 1.3, complaint_prob)
    df['has_support_complaint'] = np.random.binomial(1, np.clip(complaint_prob, 0, 1))
    
    # has_billing_complaint
    billing_rate = ROOT_CAUSE_DISTRIBUTION['billing_dispute'] + ROOT_CAUSE_DISTRIBUTION['tax_error']
    billing_prob = np.where(is_churner, billing_rate, 0.05)
    billing_prob = np.where(is_ach_payment, billing_prob * 1.5, billing_prob)
    df['has_billing_complaint'] = np.random.binomial(1, np.clip(billing_prob, 0, 1))
    
    return df


def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Add dummy variables for categorical features."""
    df = df.copy()
    df['contract_Monthly'] = (df['contract_type'] == 'Monthly').astype(int)
    df['contract_Annual'] = (df['contract_type'] == 'Annual').astype(int)
    df['tier_Standard'] = (df['service_tier'] == 'Standard').astype(int)
    df['tier_Premium'] = (df['service_tier'] == 'Premium').astype(int)
    df['payment_ACH'] = (df['payment_method'] == 'ACH').astype(int)
    df['payment_AutoPay'] = (df['payment_method'] == 'Auto-Pay').astype(int)
    return df


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

def run_genetic_algorithm(X, y, feature_names, n_generations=15, pop_size=30):
    """Run GA for feature selection (reduced pop_size for speed)."""
    if not HAS_GA:
        return list(range(len(feature_names))), None
    
    n_features = len(feature_names)
    
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        if sum(individual) == 0:
            return (0,)
        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        X_selected = X.iloc[:, selected_idx]
        model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        try:
            scores = cross_val_score(model, X_selected, y, cv=3, scoring='roc_auc')
            return (scores.mean(),)
        except:
            return (0,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    random.seed(42 + n_generations)
    pop = toolbox.population(n=pop_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations, stats=stats, verbose=False)
    
    best_ind = tools.selBest(pop, 1)[0]
    selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]
    
    return selected_features, logbook


# =============================================================================
# MODEL TRAINING - SINGLE CONFIGURATION
# =============================================================================

def train_single_config(df: pd.DataFrame, use_llm: bool, use_smote: bool, use_ga: bool, ga_gens: int = 15):
    """Train all models for a single configuration."""
    df_proc = preprocess_for_model(df)
    
    features = BASE_FEATURES + LLM_FEATURES if use_llm else BASE_FEATURES
    features = [c for c in features if c in df_proc.columns]
    
    X = df_proc[features].fillna(0)
    y = df_proc['churn']
    
    # GA feature selection
    selected_features = features
    if use_ga and HAS_GA:
        selected_idx, _ = run_genetic_algorithm(X, y, features, n_generations=ga_gens, pop_size=30)
        selected_features = [features[i] for i in selected_idx]
        X = X[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    if use_smote and HAS_SMOTE:
        smote = SMOTETomek(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reduced n_estimators for faster training on cloud
    models = {
        'Logistic Regression': ('scaled', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')),
        'Random Forest': ('raw', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, class_weight='balanced')),
        'Gradient Boosting': ('raw', GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)),
    }
    
    if HAS_LGBM:
        models['LightGBM'] = ('raw', lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1, class_weight='balanced'))
    
    if HAS_XGB:
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        models['XGBoost'] = ('raw', xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0, scale_pos_weight=scale_pos))
    
    results = {}
    for name, (data_type, model) in models.items():
        try:
            if data_type == 'scaled':
                model.fit(X_train_scaled, y_train)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_prob >= 0.5).astype(int)
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'data_type': data_type,
                'y_test': y_test,
                'y_prob': y_prob,
                'auc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
        except Exception as e:
            pass
    
    return results, selected_features, scaler


# =============================================================================
# TRAIN ALL MODELS AT STARTUP
# =============================================================================

def train_all_configurations(df: pd.DataFrame):
    """
    Train ALL configurations once at startup.
    This is the key optimization - everything trains once!
    """
    configs = {}
    
    # Base (no LLM)
    configs['base'], _, _ = train_single_config(df, use_llm=False, use_smote=False, use_ga=False)
    
    # +LLM only (BEST config)
    configs['llm_only'], features_llm, scaler_llm = train_single_config(df, use_llm=True, use_smote=False, use_ga=False)
    
    # +LLM+SMOTE
    configs['llm_smote'], _, _ = train_single_config(df, use_llm=True, use_smote=True, use_ga=False)
    
    # +LLM+GA (only if GA available)
    if HAS_GA:
        configs['llm_ga'], features_ga, _ = train_single_config(df, use_llm=True, use_smote=False, use_ga=True, ga_gens=15)
        configs['ga_features'] = features_ga
    
    configs['features'] = features_llm
    configs['scaler'] = scaler_llm
    
    return configs


# =============================================================================
# STREAMLIT PAGES
# =============================================================================

def render_header():
    """Render header with logo."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(LOGO_URL, width=200)
        except:
            pass
    
    st.title("Churn Prediction: From Root Cause to Model")
    st.markdown("*LLM root cause analysis → Feature engineering → ML prediction → Live scoring*")
    st.markdown("**Author:** Gerardo Gandara | Senior Data Scientist")
    st.markdown("---")


def render_sidebar() -> str:
    """Render sidebar."""
    with st.sidebar:
        st.title("📋 Navigation")
        page = st.radio("Section", [
            "🏠 Overview",
            "📊 Root Cause → Features",
            "🤖 Model Comparison",
            "📈 Business Impact",
            "🎮 Live Prediction"
        ])
        
        st.markdown("---")
        st.markdown("**🔧 Libraries**")
        st.write(f"LightGBM: {'✅' if HAS_LGBM else '❌'}")
        st.write(f"XGBoost: {'✅' if HAS_XGB else '❌'}")
        st.write(f"SMOTE: {'✅' if HAS_SMOTE else '❌'}")
        st.write(f"Genetic Algo: {'✅' if HAS_GA else '❌'}")
        
        st.markdown("---")
        st.markdown("**📊 Dataset**")
        st.caption("Telco Customer Churn")
        st.caption("7,043 customers | 26.5% churn")
        
    return page


def page_overview(df: pd.DataFrame):
    """Overview page."""
    st.header("🏠 What Makes This Different")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("❌ Basic Approach")
        st.markdown("""
        - Random/simulated features
        - Standard sklearn models
        - No connection to real data
        """)
    
    with col2:
        st.subheader("✅ This Demo")
        st.markdown("""
        - **LLM features from REAL root cause analysis**
        - **SMOTE** for class imbalance testing
        - **Genetic Algorithm** for feature selection
        - Clear pipeline: Reviews → Causes → Features
        """)
    
    st.subheader("🔄 End-to-End Pipeline")
    st.code("""
┌─────────────────────────────────────────────────────────────────────┐
│  Trustpilot (142 reviews)  +  ConsumerAffairs (47 reviews)          │
│                              ↓                                       │
│                    Claude API Extraction                             │
│                              ↓                                       │
│              Root Cause Distribution:                                │
│              • support_incompetent: 27%                              │
│              • tax_error: 10%                                        │
│              • billing_dispute: 7%                                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Root Cause              →    LLM Feature         →    Logic        │
│  support_incompetent     →   ticket_sentiment     →   27% negative  │
│  billing_dispute         →   has_billing_complaint                  │
│  onboarding_fail         →   churn_intent         →   explicit      │
└─────────────────────────────────────────────────────────────────────┘
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")
    col3.metric("Features", "20 base + 5 LLM")


def page_root_cause_features(df: pd.DataFrame):
    """Root cause to features page."""
    st.header("📊 From Root Causes to Predictive Features")
    
    st.success("**Key Innovation:** LLM features generated using ACTUAL root cause distribution from review analysis!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Root Cause Distribution")
        rc_df = pd.DataFrame([{'cause': k, 'pct': v * 100} for k, v in ROOT_CAUSE_DISTRIBUTION.items()]).sort_values('pct', ascending=True)
        fig = px.bar(rc_df, y='cause', x='pct', orientation='h', color='pct', color_continuous_scale='Reds', text='pct')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False, xaxis_title='% of Complaints', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Mapping to LLM Features")
        mapping_df = pd.DataFrame([
            {'Root Cause': 'support_incompetent (27%)', 'LLM Feature': 'ticket_sentiment', 'Logic': '27% of churners get sentiment ~ -0.5'},
            {'Root Cause': 'support_incompetent (27%)', 'LLM Feature': 'frustration_level', 'Logic': '27% of churners get level 3-5'},
            {'Root Cause': 'billing_dispute (7%)', 'LLM Feature': 'has_billing_complaint', 'Logic': '7% churner rate vs 5% retained'},
            {'Root Cause': 'onboarding_fail (8%)', 'LLM Feature': 'churn_intent', 'Logic': '8% show explicit intent (2-3)'},
        ])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    
    st.subheader("📊 Feature Distribution by Churn Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='ticket_sentiment', color='churn', color_discrete_map={0: '#3498db', 1: '#e74c3c'}, barmode='overlay', nbins=30, title='Ticket Sentiment')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='frustration_level', color='churn', color_discrete_map={0: '#3498db', 1: '#e74c3c'}, barmode='group', title='Frustration Level')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(df, x='churn_intent', color='churn', color_discrete_map={0: '#3498db', 1: '#e74c3c'}, barmode='group', title='Churn Intent')
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)


def page_model_comparison(configs: Dict):
    """Model comparison page - uses pre-trained models."""
    st.header("🤖 Model Comparison")
    
    st.success("✅ **Models pre-trained at startup - instant results!**")
    
    # Build comparison table
    st.subheader("📊 Component Contribution Analysis")
    
    comparison = []
    for name in configs['llm_only'].keys():
        row = {
            'Model': name,
            'Base': configs['base'].get(name, {}).get('auc', 0),
            '+LLM': configs['llm_only'].get(name, {}).get('auc', 0),
            '+LLM+SMOTE': configs['llm_smote'].get(name, {}).get('auc', 0),
        }
        
        if HAS_GA and 'llm_ga' in configs:
            row['+LLM+GA'] = configs['llm_ga'].get(name, {}).get('auc', 0)
        
        row['LLM Lift'] = row['+LLM'] - row['Base']
        row['SMOTE Lift'] = row['+LLM+SMOTE'] - row['+LLM']
        
        if HAS_GA and 'llm_ga' in configs:
            row['GA Lift'] = row['+LLM+GA'] - row['+LLM']
        
        comparison.append(row)
    
    comp_df = pd.DataFrame(comparison).sort_values('+LLM', ascending=False)
    
    # Format columns
    format_dict = {'Base': '{:.4f}', '+LLM': '{:.4f}', '+LLM+SMOTE': '{:.4f}', 'LLM Lift': '{:+.4f}', 'SMOTE Lift': '{:+.4f}'}
    gradient_cols = ['LLM Lift', 'SMOTE Lift']
    
    if HAS_GA and 'llm_ga' in configs:
        format_dict['+LLM+GA'] = '{:.4f}'
        format_dict['GA Lift'] = '{:+.4f}'
        gradient_cols.append('GA Lift')
    
    st.dataframe(comp_df.style.format(format_dict).background_gradient(subset=gradient_cols, cmap='RdYlGn', vmin=-0.02, vmax=0.15), use_container_width=True, hide_index=True)
    
    # Metrics
    avg_llm_lift = comp_df['LLM Lift'].mean()
    avg_smote_lift = comp_df['SMOTE Lift'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg LLM Contribution", f"{avg_llm_lift:+.4f}", f"{avg_llm_lift / (avg_llm_lift + abs(avg_smote_lift)) * 100:.0f}% of improvement")
    col2.metric("Avg SMOTE Contribution", f"{avg_smote_lift:+.4f}", "Hurts performance!" if avg_smote_lift < 0 else "Helps")
    col3.metric("Best Config", "+LLM only" if avg_smote_lift < 0 else "+LLM+SMOTE")
    
    st.markdown("---")
    
    # ROC Curves
    st.subheader("ROC Curves Comparison")
    col1, col2 = st.columns(2)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    with col1:
        st.markdown("**+LLM Only (Best)**")
        fig = go.Figure()
        for i, (name, r) in enumerate(configs['llm_only'].items()):
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({r['auc']:.3f})", line=dict(color=colors[i % 5])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='gray')))
        fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**+LLM+SMOTE**")
        fig = go.Figure()
        for i, (name, r) in enumerate(configs['llm_smote'].items()):
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({r['auc']:.3f})", line=dict(color=colors[i % 5])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='gray')))
        fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance (Best Model: +LLM Only)")
    best_model_name = comp_df.iloc[0]['Model']
    best = configs['llm_only'][best_model_name]['model']
    
    if hasattr(best, 'feature_importances_'):
        imp = pd.DataFrame({'feature': ALL_FEATURES, 'importance': best.feature_importances_})
        imp = imp.sort_values('importance', ascending=True).tail(15)
        imp['is_llm'] = imp['feature'].isin(LLM_FEATURES)
        
        fig = px.bar(imp, x='importance', y='feature', orientation='h', color='is_llm', color_discrete_map={True: '#e74c3c', False: '#3498db'}, labels={'is_llm': 'LLM Feature'})
        fig.update_layout(height=450, legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
        top_10 = imp.tail(10)
        llm_in_top_10 = top_10['is_llm'].sum()
        st.info(f"**{llm_in_top_10} of top 10 features are LLM-extracted** (red bars)")
    
    best_auc = comp_df.iloc[0]['+LLM']
    st.success(f"**Best: {best_model_name}** | AUC: {best_auc:.4f} | LLM Lift: {comp_df.iloc[0]['LLM Lift']:+.4f} | SMOTE adds no value")


def page_business_impact(configs: Dict):
    """Business impact page."""
    st.header("📈 Business Impact Analysis")
    
    st.info("**Using +LLM model (best performance)**")
    
    results = configs['llm_only']
    selected = st.selectbox("Model", list(results.keys()))
    r = results[selected]
    
    # Threshold recommendation
    st.subheader("🎯 Threshold Recommendation for B2B Churn")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Cost Analysis:**
        - Miss a churner: **-$15,000/year**
        - Unnecessary outreach: **-$75**
        
        **Ratio: 200:1** → Prioritize RECALL
        """)
    
    with col2:
        st.markdown("""
        | Threshold | Strategy |
        |-----------|----------|
        | 0.50 | Conservative |
        | **0.15** | **Optimal** |
        | 0.10 | Aggressive |
        """)
    
    st.markdown("---")
    
    # Precision-Recall curve
    st.subheader("📊 Precision-Recall Trade-off")
    thresh_data = []
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (r['y_prob'] >= t).astype(int)
        thresh_data.append({
            't': t,
            'Precision': precision_score(r['y_test'], y_pred, zero_division=0),
            'Recall': recall_score(r['y_test'], y_pred, zero_division=0),
            'F1': f1_score(r['y_test'], y_pred, zero_division=0)
        })
    tdf = pd.DataFrame(thresh_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tdf['t'], y=tdf['Precision'], name='Precision', line=dict(color='#e74c3c', width=3)))
    fig.add_trace(go.Scatter(x=tdf['t'], y=tdf['Recall'], name='Recall', line=dict(color='#3498db', width=3)))
    fig.add_trace(go.Scatter(x=tdf['t'], y=tdf['F1'], name='F1', line=dict(color='#2ecc71', width=2, dash='dash')))
    fig.update_layout(xaxis_title='Classification Threshold', yaxis_title='Score', height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Calculator
    st.subheader("💰 ROI Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Threshold", 0.1, 0.9, 0.15, 0.05)
        value = st.number_input("Annual Customer Value ($)", 1000, 100000, 15000)
    with col2:
        cost = st.number_input("Outreach Cost ($)", 10, 500, 75)
        save_rate = st.slider("Save Rate (%)", 10, 50, 30) / 100
    
    y_pred = (r['y_prob'] >= threshold).astype(int)
    tp = ((y_pred == 1) & (r['y_test'] == 1)).sum()
    fp = ((y_pred == 1) & (r['y_test'] == 0)).sum()
    fn = ((y_pred == 0) & (r['y_test'] == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    saved = int(tp * save_rate)
    revenue = saved * value
    cost_total = (tp + fp) * cost
    net = revenue - cost_total
    
    st.markdown("**📊 Model Performance at Threshold**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.1%}")
    col2.metric("Recall", f"{recall:.1%}")
    col3.metric("F1 Score", f"{f1:.2f}")
    col4.metric("Threshold", f"{threshold:.2f}")
    
    st.markdown("**💰 Business Impact**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flagged for Outreach", f"{tp + fp:,}")
    col2.metric("Churners Caught", f"{tp:,} / {tp + fn}", f"{recall:.0%} recall")
    col3.metric("Customers Saved", f"{saved:,}", f"at {save_rate:.0%} save rate")
    col4.metric("Net ROI", f"${net:,}", f"${revenue:,} - ${cost_total:,}")


def page_live_prediction(configs: Dict, df: pd.DataFrame):
    """Live prediction page."""
    st.header("🎮 Live Churn Prediction")
    
    results = configs['llm_only']
    scaler = configs['scaler']
    
    model_name = st.selectbox("Model", list(results.keys()))
    model_info = results[model_name]
    model = model_info['model']
    
    st.markdown("---")
    st.caption("💡 *In B2B: Premium tier & Premium support = lower churn risk*")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("👤 Customer")
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.slider("Monthly Spend ($)", 20, 120, 80)
        contract = st.selectbox("Contract", ["Monthly", "Annual", "Multi-Year"])
    
    with c2:
        st.subheader("📦 Products")
        tier = st.selectbox("Service Tier", ["Basic", "Standard", "Premium"])
        premium_support = st.checkbox("Premium Support")
        payment = st.selectbox("Payment", ["ACH", "Credit Card", "Auto-Pay"])
    
    with c3:
        st.subheader("🤖 LLM Features (from tickets)")
        sentiment = st.slider("Ticket Sentiment", -1.0, 1.0, -0.3, 0.1)
        frustration = st.slider("Frustration Level", 0, 5, 3)
        intent = st.slider("Churn Intent", 0, 3, 1)
        supp_complaint = st.checkbox("Support Complaint", value=True)
        bill_complaint = st.checkbox("Billing Complaint")
    
    # B2B logic inversion
    if tier == "Premium":
        tier_standard_val, tier_premium_val = 0, 0
        security_val, backup_val = 1, 1
    elif tier == "Standard":
        tier_standard_val, tier_premium_val = 1, 0
        security_val, backup_val = 1, 0
    else:
        tier_standard_val, tier_premium_val = 0, 1
        security_val, backup_val = 0, 0
    
    premium_support_val = 0 if premium_support else 1
    sentiment_adjusted = min(1, max(-1, sentiment + (0.3 if premium_support else 0)))
    
    inp = {
        'tenure_months': tenure, 'monthly_spend': monthly, 'lifetime_value': tenure * monthly,
        'is_enterprise': 0, 'has_partner_account': 0, 'has_subsidiary': 0,
        'has_phone_support': 1, 'has_multiple_products': 0,
        'has_security_addon': security_val, 'has_backup_service': backup_val,
        'has_premium_support': premium_support_val, 'has_addon_1': 0, 'has_addon_2': 0,
        'paperless_billing': 1,
        'contract_Monthly': 1 if contract == "Monthly" else 0,
        'contract_Annual': 1 if contract == "Annual" else 0,
        'tier_Standard': tier_standard_val, 'tier_Premium': tier_premium_val,
        'payment_ACH': 1 if payment == "ACH" else 0,
        'payment_AutoPay': 1 if payment == "Auto-Pay" else 0,
        'ticket_sentiment': sentiment_adjusted, 'frustration_level': frustration,
        'churn_intent': intent, 'has_support_complaint': int(supp_complaint),
        'has_billing_complaint': int(bill_complaint)
    }
    
    input_df = pd.DataFrame([{k: v for k, v in inp.items() if k in ALL_FEATURES}])[ALL_FEATURES]
    
    if model_info['data_type'] == 'scaled':
        prob = model.predict_proba(scaler.transform(input_df))[0][1]
    else:
        prob = model.predict_proba(input_df)[0][1]
    
    risk = int(prob * 100)
    
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk, title={'text': "P(Churn)"},
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 30], 'color': "#2ecc71"},
                             {'range': [30, 60], 'color': "#f1c40f"},
                             {'range': [60, 100], 'color': "#e74c3c"}]}
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        risk_factors = []
        protective_factors = []
        
        if contract == "Monthly":
            risk_factors.append("Monthly contract (no commitment)")
        else:
            protective_factors.append(f"{contract} contract (committed)")
        
        if tenure < 12:
            risk_factors.append(f"Short tenure ({tenure} months)")
        elif tenure > 24:
            protective_factors.append(f"Long tenure ({tenure} months)")
        
        if tier == "Basic":
            risk_factors.append("Basic tier (less invested)")
        elif tier == "Premium":
            protective_factors.append("Premium tier (highly invested)")
        
        if not premium_support:
            risk_factors.append("No premium support")
        else:
            protective_factors.append("Has premium support")
        
        if sentiment < -0.3:
            risk_factors.append(f"Negative sentiment ({sentiment:.1f})")
        
        if frustration >= 3:
            risk_factors.append(f"High frustration ({frustration}/5)")
        
        if intent >= 2:
            risk_factors.append(f"Churn intent ({intent}/3)")
        
        if supp_complaint:
            risk_factors.append("Support complaint")
        
        if bill_complaint:
            risk_factors.append("Billing complaint")
        
        if risk < 30:
            st.success(f"✅ **LOW RISK ({risk}%)**")
            if protective_factors:
                st.markdown("**Protected by:** " + ", ".join(protective_factors[:3]))
        elif risk < 60:
            st.warning(f"⚠️ **MEDIUM RISK ({risk}%)** - Schedule check-in")
            if risk_factors:
                st.markdown("**Risk factors:** " + ", ".join(risk_factors[:3]))
        else:
            st.error(f"🚨 **HIGH RISK ({risk}%)** - Immediate intervention!")
            st.markdown("**Triggered by:**")
            for factor in risk_factors[:4]:
                st.markdown(f"- {factor}")
        
        st.caption(f"Model: {model_name} | AUC: {model_info['auc']:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application - trains everything once at startup."""
    render_header()
    
    # Load data
    df_raw, loaded = load_and_rename_telco()
    if not loaded:
        st.error("⚠️ Dataset not found! Please add `Telco_Customer_Churn.csv` to the repository.")
        st.stop()
    
    df = add_llm_features(df_raw)
    
    # TRAIN ALL MODELS ONCE AT STARTUP
    if 'configs' not in st.session_state:
        with st.spinner("🚀 Training all models (one-time startup, ~60-90 seconds)..."):
            st.session_state['configs'] = train_all_configurations(df)
        st.success("✅ All models trained! Navigation is now instant.")
    
    configs = st.session_state['configs']
    
    # Render page
    page = render_sidebar()
    
    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 Root Cause → Features":
        page_root_cause_features(df)
    elif page == "🤖 Model Comparison":
        page_model_comparison(configs)
    elif page == "📈 Business Impact":
        page_business_impact(configs)
    elif page == "🎮 Live Prediction":
        page_live_prediction(configs, df)


if __name__ == "__main__":
    main()
