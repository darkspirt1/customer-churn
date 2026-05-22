import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
# ─── Auto-generate model if pkl doesn't exist ──────────
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def generate_model():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    df_majority = df[df['Churn'] == 0]
    df_minority = df[df['Churn'] == 1]
    df_minority_up = resample(df_minority, replace=True,
                              n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_up])

    X = df_balanced.drop('Churn', axis=1)
    y = df_balanced['Churn']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


# Auto-generate if missing
if not os.path.exists('churn_model.pkl'):
    generate_model()
# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Churn Deep Dive",
    page_icon="🔴",
    layout="wide"
)

# ─── Load Data & Model ─────────────────────────────────────


@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


df = load_data()
model = load_model()
playbook = pd.read_csv('churn_playbook.csv')

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/emoji/96/red-circle-emoji.png", width=60)
st.sidebar.title("Churn Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Churn Analysis",
    "🤖 Predict Churn",
    "📋 Churn Playbook"
])

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Churn Deep Dive — Overview")
    st.markdown("---")

    # KPI Cards
    total = len(df)
    churned = df['Churn'].sum()
    churn_rate = df['Churn'].mean() * 100
    avg_tenure = df['tenure'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",  f"{total:,}")
    col2.metric("Churned",          f"{churned:,}")
    col3.metric("Churn Rate",       f"{churn_rate:.1f}%")
    col4.metric("Avg Tenure",       f"{avg_tenure:.0f} months")

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df['Churn'].value_counts()
        ax.bar(['No Churn', 'Churned'], counts.values,
               color=['steelblue', 'tomato'])
        for i, v in enumerate(counts.values):
            ax.text(i, v+30, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 3))
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        bars = ax.bar(contract_churn.index, contract_churn.values,
                      color=['tomato', 'steelblue', 'green'])
        for bar, val in zip(bars, contract_churn.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Churn Rate %')
        st.pyplot(fig)

    # Tenure trend
    st.subheader("Churn Rate by Tenure")
    churn_trend = df.groupby('tenure')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(churn_trend.index, churn_trend.values, color='tomato', linewidth=2)
    ax.fill_between(churn_trend.index, churn_trend.values,
                    alpha=0.2, color='tomato')
    ax.axhline(y=churn_rate, color='gray',
               linestyle='--', label='Avg Churn Rate')
    ax.set_xlabel('Tenure (Months)')
    ax.set_ylabel('Churn Rate %')
    ax.legend()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 2 — CHURN ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🔍 Churn Analysis":
    st.title("🔍 Churn Analysis")
    st.markdown("---")

    analysis = st.selectbox("Select Analysis", [
        "By Payment Method",
        "By Internet Service",
        "By Senior Citizen",
        "Feature Usage Heatmap"
    ])

    if analysis == "By Payment Method":
        fig, ax = plt.subplots(figsize=(9, 4))
        payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
        bars = ax.bar(payment_churn.index,
                      payment_churn.values, color='tomato')
        for bar, val in zip(bars, payment_churn.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Churn Rate %')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        st.info("💡 Electronic check users have the highest churn rate.")

    elif analysis == "By Internet Service":
        fig, ax = plt.subplots(figsize=(7, 4))
        internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
        bars = ax.bar(internet_churn.index, internet_churn.values,
                      color=['steelblue', 'tomato', 'green'])
        for bar, val in zip(bars, internet_churn.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Churn Rate %')
        st.pyplot(fig)
        st.info(
            "💡 Fiber optic users churn despite paying more — service quality issue.")

    elif analysis == "By Senior Citizen":
        fig, ax = plt.subplots(figsize=(6, 4))
        senior_churn = df.groupby('SeniorCitizen')['Churn'].mean() * 100
        bars = ax.bar(['Non-Senior', 'Senior'], senior_churn.values,
                      color=['steelblue', 'tomato'])
        for bar, val in zip(bars, senior_churn.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Churn Rate %')
        st.pyplot(fig)

    elif analysis == "Feature Usage Heatmap":
        feature_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        churn_by_feature = {}
        for col in feature_cols:
            churn_by_feature[col] = df.groupby(col)['Churn'].mean() * 100
        feature_df = pd.DataFrame(churn_by_feature).T
        if 'No internet service' in feature_df.columns:
            feature_df = feature_df[['No', 'Yes']]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(feature_df, annot=True, fmt='.1f',
                    cmap='RdYlGn_r', linewidths=0.5, ax=ax)
        ax.set_title('Churn Rate by Feature Usage')
        st.pyplot(fig)
        st.info("💡 No OnlineSecurity or TechSupport = nearly 2x churn rate.")

# ══════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CHURN
# ══════════════════════════════════════════════════════════
elif page == "🤖 Predict Churn":
    st.title("🤖 Predict Customer Churn")
    st.markdown("Fill in customer details to get churn prediction")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)",
                                        value=float(tenure * monthly_charges))

    with col2:
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)",
                                       "Credit card (automatic)"])
        internet_service = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])

    with col3:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox(
            "Tech Support",    ["Yes", "No", "No internet service"])
        senior_citizen = st.selectbox("Senior Citizen",  ["No", "Yes"])

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):

        # Build input dict matching training columns
        input_dict = {
            'tenure':           tenure,
            'MonthlyCharges':   monthly_charges,
            'TotalCharges':     total_charges,
            'SeniorCitizen':    1 if senior_citizen == "Yes" else 0,
            'Partner':          0,
            'Dependents':       0,
            'PhoneService':     1,
            'MultipleLines':    0,
            'InternetService':  internet_service,
            'OnlineSecurity':   online_security,
            'OnlineBackup':     0,
            'DeviceProtection': 0,
            'TechSupport':      tech_support,
            'StreamingTV':      0,
            'StreamingMovies':  0,
            'Contract':         contract,
            'PaperlessBilling': 1,
            'PaymentMethod':    payment_method,
            'gender':           'Male',
        }

        input_df = pd.DataFrame([input_dict])

        # Encode
        le = LabelEncoder()
        df_temp = load_data().drop(['Churn', 'customerID'],
                                   axis=1, errors='ignore')
        for col in input_df.select_dtypes(include='object').columns:
            if col in df_temp.columns:
                le.fit(df_temp[col].astype(str))
                input_df[col] = le.transform(input_df[col].astype(str))

        # Align columns
        trained_cols = model.feature_names_in_ \
            if hasattr(model, 'feature_names_in_') \
            else df_temp.columns
        for col in trained_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_cols]

        prob = model.predict_proba(input_df)[0][1]

        # Display result
        if prob >= 0.75:
            st.error(f"🔴 HIGH RISK — Churn Probability: {prob*100:.1f}%")
            st.warning(
                "⚡ Immediate Action Required — assign customer success team")
        elif prob >= 0.45:
            st.warning(f"🟡 MEDIUM RISK — Churn Probability: {prob*100:.1f}%")
            st.info("📋 Send retention offer within 7 days")
        else:
            st.success(f"🟢 LOW RISK — Churn Probability: {prob*100:.1f}%")
            st.info("✅ Customer is stable — monitor monthly")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(['Risk'], [prob], color='tomato' if prob > 0.75
                else 'orange' if prob > 0.45 else 'green', height=0.4)
        ax.barh(['Risk'], [1-prob], left=[prob], color='#eee', height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.45, 0.75, 1])
        ax.set_xticklabels(['0%', '45%', '75%', '100%'])
        ax.set_title(f'Churn Probability: {prob*100:.1f}%')
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 4 — CHURN PLAYBOOK
# ══════════════════════════════════════════════════════════
elif page == "📋 Churn Playbook":
    st.title("📋 Churn Playbook")
    st.markdown("---")

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk",
                (playbook['Risk_Segment'] == '🔴 High Risk').sum())
    col2.metric("🟡 Medium Risk",
                (playbook['Risk_Segment'] == '🟡 Medium Risk').sum())
    col3.metric("🟢 Low Risk",
                (playbook['Risk_Segment'] == '🟢 Low Risk').sum())

    st.markdown("---")

    # Filter
    risk_filter = st.multiselect("Filter by Risk Segment",
                                 ['🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk'],
                                 default=['🔴 High Risk'])

    filtered = playbook[playbook['Risk_Segment'].isin(risk_filter)]

    st.dataframe(filtered[[
        'tenure', 'Contract', 'MonthlyCharges',
        'Churn_Probability', 'Risk_Segment', 'Action_Plan'
    ]].sort_values('Churn_Probability', ascending=False),
        use_container_width=True)

    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Playbook CSV",
        data=csv,
        file_name='churn_playbook_filtered.csv',
        mime='text/csv'
    )
