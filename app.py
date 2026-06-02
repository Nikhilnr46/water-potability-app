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
# Title
# -------------------------------
st.title("💧 Water Potability Prediction App")

# -------------------------------
# Upload Dataset Section
# -------------------------------
st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())
else:
    # Default dataset if no file uploaded
    df = pd.read_csv("water_potability.csv")
    st.write("Using default dataset (water_potability.csv)")
    st.dataframe(df.head())

# Handle missing values
df = df.fillna(df.median())

# -------------------------------
# Train/Test Split
# -------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Selection
# -------------------------------
st.subheader("Choose Model")
model_choice = st.selectbox("Select a model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier()

model.fit(X_train, y_train)

# -------------------------------
# User Input via Sliders
# -------------------------------
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

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Potability"):
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        st.success("The water is predicted to be **Potable (Safe to Drink)**")
    else:
        st.error("The water is predicted to be **Not Potable (Unsafe)**")

# -------------------------------
# Model Performance
# -------------------------------
st.subheader("Model Performance on Test Data")

y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
st.pyplot(fig)
