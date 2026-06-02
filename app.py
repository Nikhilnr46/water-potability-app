import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

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
    st.subheader("What does this app do?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **EDA**\n\nExplore the dataset with charts and statistics.")
    with col2:
        st.success("🔬 **Prediction**\n\nEnter water parameters and get a potability prediction.")
    with col3:
        st.warning("ℹ️ **About**\n\nLearn about the dataset and the project.")

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
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select a feature", X.columns.tolist())
    fig2, ax2 = plt.subplots()
    df[df["Potability"] == 0][feature].hist(ax=ax2, alpha=0.6, label="Not Potable", color="#e74c3c", bins=30)
    df[df["Potability"] == 1][feature].hist(ax=ax2, alpha=0.6, label="Potable", color="#2ecc71", bins=30)
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Distribution of {feature}")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    im = ax3.imshow(corr, cmap="coolwarm", aspect="auto")
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_yticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax3.set_yticklabels(corr.columns, fontsize=9)
    plt.colorbar(im, ax=ax3)
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
    st.write("Adjust the sliders below and click **Predict Potability**.")

    st.subheader("Enter Water Parameters")

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

    st.markdown("---")
    st.subheader("Model Performance on Test Data")

    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

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

    **Goal:** Predict whether water is safe for human consumption using machine learning.

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
    - **Logistic Regression** — Fast, interpretable baseline classifier.
    - **Random Forest** — Ensemble method, typically higher accuracy.

    ---

    ### 🛠️ Tech Stack
    `Python` · `Streamlit` · `Scikit-learn` · `Pandas` · `Matplotlib`
    """)

    st.markdown("---")
    st.caption("Built as a machine learning demo project.")
