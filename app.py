import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Water Potability App", page_icon="💧", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("💧 Water Potability App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 EDA", "🔬 Prediction", "ℹ️ About"])

# -------------------------------
# Upload Dataset
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("water_potability.csv")

df = df.fillna(df.median())

# -------------------------------
# Train Model (shared across pages)
# -------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

st.sidebar.markdown("---")
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.selectbox("Select a model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier()

model.fit(X_train, y_train)

# ================================================
# PAGE 1: HOME
# ================================================
if page == "🏠 Home":
    st.title("💧 Water Potability Prediction App")
    st.markdown("### Welcome!")
    st.write(
        "This app uses machine learning to predict whether water is safe to drink "
        "based on physicochemical properties. Use the sidebar to navigate between pages."
    )

    st.markdown("---")
    st.subheader("Dataset Overview")
    st.write(f"**Total samples:** {len(df)}  |  **Features:** {len(df.columns)-1}  |  "
             f"**Potable:** {int(df['Potability'].sum())}  |  "
             f"**Not Potable:** {int((df['Potability'] == 0).sum())}")
    st.dataframe(df.head())

# ================================================
# PAGE 2: EDA
# ================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("Potability Distribution")
    fig, ax = plt.subplots()
    df["Potability"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_xticklabels(["Not Potable", "Potable"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Potability Count")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Missing Values (before fill)")
    raw_df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("water_potability.csv")
    st.bar_chart(raw_df.isnull().sum())

# ================================================
# PAGE 3: PREDICTION
# ================================================
elif page == "🔬 Prediction":
    st.title("🔬 Predict Water Potability")

    ph = st.slider("pH", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness", 0.0, 500.0, 200.0)
    solids = st.slider("Solids (TDS)", 0.0, 50000.0, 10000.0)
    chloramines = st.slider("Chloramines", 0.0, 10.0, 5.0)
    sulfate = st.slider("Sulfate", 0.0, 500.0, 250.0)
    conductivity = st.slider("Conductivity", 0.0, 1000.0, 500.0)
    organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 15.0)
    trihalomethanes = st.slider("Trihalomethanes", 0.0, 150.0, 75.0)
    turbidity = st.slider("Turbidity", 0.0, 10.0, 5.0)

    user_input = np.array([
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]).reshape(1, -1)

    user_input_scaled = scaler.transform(user_input)

    if st.button("Predict Potability"):
        prediction = model.predict(user_input_scaled)[0]
        if prediction == 1:
            st.success("✅ The water is predicted to be **Potable (Safe to Drink)**")
        else:
            st.error("❌ The water is predicted to be **Not Potable (Unsafe)**")
            st.warning("💡 Unsafe water leads to wastage when supplied. This highlights the importance of resource efficiency.")

    st.markdown("---")
    st.subheader("Model Performance on Test Data")

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    st.table(pd.DataFrame(metrics, index=["Score"]).T)

    fig4, ax4 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax4)
    st.pyplot(fig4)

# ================================================
# PAGE 4: ABOUT
# ================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 💧 Water Potability Prediction

    **Name:** Nikhil Vasista N R  
    **SRN:** PES1PG25CA305  
    **Domain:** Environment & Climate  
    **Goal:** SDG 12 – Responsible Consumption and Production  

    ---

    ### 📁 Dataset
    - **Source:** [Kaggle — Water Quality Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
    - **Samples:** ~3,276 rows
    - **Target:** `Potability` (1 = Safe, 0 = Unsafe)

    ---

    ### 🧪 Features
    | Feature | Description |
    |---|---|
    | pH | Acidity/alkalinity of water (WHO: 6.5–8.5) |
    | Hardness | Calcium & magnesium content (mg/L) |
    | Solids (TDS) | Total dissolved solids (mg/L) |
    | Chloramines | Disinfectant level (ppm) |
    | Sulfate | Sulfate concentration (mg/L) |
    | Conductivity | Electrical conductivity (μS/cm) |
    | Organic Carbon | Total organic carbon (ppm) |
    | Trihalomethanes | Byproducts of disinfection (μg/L) |
    | Turbidity | Clarity of water (NTU) |

    ---

    ### 🤖 Models Used
    - Logistic Regression — Fast, interpretable baseline classifier.
    - Random Forest — Ensemble method, typically higher accuracy.

    ---

    ### 🛠️ Tech Stack
    Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn
    """)

    st.caption("Built as a machine learning demo project for PES University.")
