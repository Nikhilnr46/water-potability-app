import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AquaGuard — Water Potability AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #0d2137 60%, #071520 100%);
    border-right: 1px solid rgba(0,200,255,0.12);
}
[data-testid="stSidebar"] * { color: #c8dff0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
}

/* Main background */
.main { background: #f0f6fd; }

/* Cards */
.card {
    background: white;
    border-radius: 18px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(10,60,120,0.08);
    border: 1px solid rgba(10,80,160,0.07);
    margin-bottom: 20px;
}
.card-dark {
    background: linear-gradient(135deg, #0a1e3d, #0d2c52);
    border-radius: 18px;
    padding: 32px;
    color: white;
    border: 1px solid rgba(0,180,255,0.15);
}

/* Metric tiles */
.metric-tile {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(10,60,120,0.07);
    border-top: 4px solid #0ea5e9;
}
.metric-tile .val {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #0369a1;
    line-height: 1;
}
.metric-tile .lbl {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Hero headline */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    color: #0a1e3d;
}
.hero-accent { color: #0ea5e9; }

/* Section heading */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0a2540;
    border-left: 4px solid #0ea5e9;
    padding-left: 12px;
    margin-bottom: 16px;
}

/* Result banners */
.result-safe {
    background: linear-gradient(120deg,#dcfce7,#bbf7d0);
    border: 1.5px solid #22c55e;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-unsafe {
    background: linear-gradient(120deg,#fee2e2,#fecaca);
    border: 1.5px solid #ef4444;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
}
.result-sub { font-size: 0.95rem; color: #374151; margin-top: 6px; }

/* Parameter badge */
.param-badge {
    display:inline-block;
    background:#e0f2fe;
    color:#0369a1;
    border-radius:20px;
    padding:3px 12px;
    font-size:0.75rem;
    font-weight:600;
    margin:2px;
}

/* Sidebar logo area */
.sidebar-logo {
    font-family:'Syne',sans-serif;
    font-size:1.6rem;
    font-weight:800;
    color:#38bdf8 !important;
    letter-spacing:-0.02em;
}
.sidebar-tagline {
    font-size:0.72rem;
    color:#64a0c0 !important;
    letter-spacing:0.1em;
    text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💧 AquaGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Water Safety Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔬  Predict Potability"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("**Upload Dataset**")
    uploaded_file = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Model**")
    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Logistic Regression"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("AquaGuard v2.0 · ML-powered water safety analysis")


# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        try:
            df = pd.read_csv("water_potability.csv")
        except FileNotFoundError:
            # Synthetic fallback so the app always runs
            np.random.seed(42)
            n = 3276
            df = pd.DataFrame({
                "ph":               np.random.uniform(0, 14, n),
                "Hardness":         np.random.uniform(47, 323, n),
                "Solids":           np.random.uniform(320, 61227, n),
                "Chloramines":      np.random.uniform(0.35, 13.1, n),
                "Sulfate":          np.random.uniform(129, 481, n),
                "Conductivity":     np.random.uniform(181, 753, n),
                "Organic_carbon":   np.random.uniform(2.2, 28.3, n),
                "Trihalomethanes":  np.random.uniform(0.74, 124, n),
                "Turbidity":        np.random.uniform(1.45, 6.99, n),
                "Potability":       np.random.randint(0, 2, n),
            })
    df = df.fillna(df.median(numeric_only=True))
    return df


@st.cache_resource
def train_model(model_name, _df):
    X = _df.drop("Potability", axis=1)
    y = _df["Potability"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, acc, auc, list(X.columns)


df = load_data(uploaded_file)
model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, acc, auc, feature_cols = train_model(model_choice, df)


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":

    # Hero
    st.markdown(f"""
    <div class="card-dark" style="margin-bottom:28px;">
        <div class="hero-title" style="color:white;">
            AI-Powered<br><span class="hero-accent">Water Safety</span><br>Analysis
        </div>
        <p style="color:#94b8d4;font-size:1.05rem;margin-top:16px;max-width:560px;">
            AquaGuard uses machine learning to predict whether water is safe to drink
            based on 9 physicochemical parameters — helping communities make informed
            decisions about water quality.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    total = len(df)
    potable = int(df["Potability"].sum())
    not_potable = total - potable
    potable_pct = round(potable / total * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{total:,}", "Total Samples"),
        (f"{potable:,}", "Potable Samples"),
        (f"{not_potable:,}", "Non-Potable"),
        (f"{potable_pct}%", "Potable Rate"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
        <div class="metric-tile">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset preview + distribution
    col_left, col_right = st.columns([1.6, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-head">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(
            df.head(8).style.background_gradient(cmap="Blues", axis=0),
            use_container_width=True, height=260
        )

    with col_right:
        st.markdown('<div class="section-head">Potability Split</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        sizes = [potable, not_potable]
        colors = ["#0ea5e9", "#f43f5e"]
        wedge_props = {"width": 0.52, "edgecolor": "white", "linewidth": 3}
        ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90)
        ax.text(0, 0, f"{potable_pct}%\nPotable", ha="center", va="center",
                fontsize=16, fontweight="bold", color="#0a2540")
        legend_elems = [
            mpatches.Patch(facecolor="#0ea5e9", label="Potable"),
            mpatches.Patch(facecolor="#f43f5e", label="Not Potable"),
        ]
        ax.legend(handles=legend_elems, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10, frameon=False)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature distributions
    st.markdown('<div class="section-head">Feature Distributions</div>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(3, 3, figsize=(13, 8))
    axes = axes.flatten()
    palette = ["#0ea5e9", "#f43f5e"]
    for i, col in enumerate(feature_cols):
        for label, color in zip([0, 1], palette):
            subset = df[df["Potability"] == label][col]
            axes[i].hist(subset, bins=30, alpha=0.65, color=color, edgecolor="none")
        axes[i].set_title(col, fontsize=10, fontweight="600", color="#0a2540")
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)
        axes[i].spines[["top", "right"]].set_visible(False)
        axes[i].set_facecolor("#f8fbff")
    legend_elems2 = [
        mpatches.Patch(facecolor="#0ea5e9", label="Not Potable"),
        mpatches.Patch(facecolor="#f43f5e", label="Potable"),
    ]
    fig2.legend(handles=legend_elems2, loc="lower right", fontsize=10, frameon=False)
    fig2.patch.set_facecolor("#f0f6fd")
    fig2.tight_layout(pad=2)
    st.pyplot(fig2, use_container_width=True)

    # Parameters reference
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Water Quality Parameters</div>', unsafe_allow_html=True)
    param_info = {
        "pH": "Measures acidity/alkalinity. WHO recommends 6.5–8.5.",
        "Hardness": "Caused by Ca²⁺ & Mg²⁺. Hard water >200 mg/L.",
        "Solids (TDS)": "Total dissolved solids. Desired <500 mg/L.",
        "Chloramines": "Disinfectants in water. Limit: 4 mg/L.",
        "Sulfate": "Naturally occurring. Limit: 250 mg/L.",
        "Conductivity": "Ionic content indicator. Standard <400 μS/cm.",
        "Organic Carbon": "Measure of organic compounds. <2 mg/L preferred.",
        "Trihalomethanes": "Disinfection byproducts. Limit: 80 μg/L.",
        "Turbidity": "Water clarity. <1 NTU is ideal.",
    }
    cols = st.columns(3)
    for i, (param, desc) in enumerate(param_info.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="min-height:100px;padding:18px 20px;">
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            color:#0369a1;font-size:0.95rem;">{param}</div>
                <div style="font-size:0.82rem;color:#475569;margin-top:6px;
                            line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT POTABILITY
# ─────────────────────────────────────────────
elif page == "🔬  Predict Potability":

    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:2.2rem;
                    font-weight:800;color:#0a2540;">
            Water Potability <span style="color:#0ea5e9;">Predictor</span>
        </div>
        <div style="color:#64748b;font-size:0.95rem;margin-top:4px;">
            Adjust the sliders below, then click <b>Analyze Sample</b> to get a prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model performance strip
    mc1, mc2, mc3 = st.columns(3)
    for col, val, lbl in zip(
        [mc1, mc2, mc3],
        [f"{acc*100:.1f}%", f"{auc:.3f}", model_choice.split()[0]],
        ["Test Accuracy", "ROC-AUC Score", "Active Model"],
    ):
        col.markdown(f"""
        <div class="metric-tile">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input sliders
    st.markdown('<div class="section-head">Sample Parameters</div>', unsafe_allow_html=True)

    defaults = df[feature_cols].median()
    slider_cfg = [
        ("pH",              "ph",              0.0,   14.0,   0.1),
        ("Hardness",        "Hardness",        0.0,   500.0,  1.0),
        ("Solids (TDS)",    "Solids",          0.0,   61000.0,100.0),
        ("Chloramines",     "Chloramines",     0.0,   14.0,   0.1),
        ("Sulfate",         "Sulfate",         50.0,  500.0,  1.0),
        ("Conductivity",    "Conductivity",    100.0, 800.0,  1.0),
        ("Organic Carbon",  "Organic_carbon",  1.0,   30.0,   0.1),
        ("Trihalomethanes", "Trihalomethanes", 0.0,   130.0,  0.5),
        ("Turbidity",       "Turbidity",       1.0,   7.0,    0.01),
    ]

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        row1, row2, row3 = st.columns(3), st.columns(3), st.columns(3)
        input_vals = {}
        for idx, (label, key, lo, hi, step) in enumerate(slider_cfg):
            default_val = float(round(defaults.get(key, (lo + hi) / 2), 2))
            col = [row1, row2, row3][idx // 3][idx % 3]
            with col:
                input_vals[key] = st.slider(
                    label, lo, hi, default_val, step,
                    help=f"Adjust {label} value"
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_clicked = st.button(
            "🔬  Analyze Sample",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        user_arr = np.array([input_vals[k] for k in feature_cols]).reshape(1, -1)
        user_scaled = scaler.transform(user_arr)
        prediction = model.predict(user_scaled)[0]
        confidence = model.predict_proba(user_scaled)[0][prediction] * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-title" style="color:#15803d;">✅ Potable — Safe to Drink</div>
                <div class="result-sub">
                    The model is <b>{confidence:.1f}%</b> confident this water meets safety standards.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unsafe">
                <div class="result-title" style="color:#b91c1c;">⚠️ Not Potable — Unsafe</div>
                <div class="result-sub">
                    The model is <b>{confidence:.1f}%</b> confident this water does <b>not</b> meet safety standards.
                    Treatment is recommended before consumption.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary of entered values
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">Submitted Values</div>', unsafe_allow_html=True)
        badges = "".join(
            f'<span class="param-badge">{label}: {input_vals[key]}</span>'
            for label, key, *_ in slider_cfg
        )
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)

    # ── Model performance section ──────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Model Performance</div>', unsafe_allow_html=True)

    perf_left, perf_right = st.columns(2, gap="large")

    with perf_left:
        st.markdown("**Confusion Matrix**")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax3,
            colorbar=False,
            cmap="Blues",
            display_labels=["Not Potable", "Potable"]
        )
        ax3.set_title("Confusion Matrix", fontsize=12, fontweight="bold", color="#0a2540")
        fig3.patch.set_facecolor("#f0f6fd")
        ax3.set_facecolor("#f0f6fd")
        st.pyplot(fig3, use_container_width=True)

    with perf_right:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.plot(fpr, tpr, color="#0ea5e9", lw=2.5, label=f"AUC = {auc:.3f}")
        ax4.plot([0, 1], [0, 1], color="#cbd5e1", lw=1.5, linestyle="--")
        ax4.fill_between(fpr, tpr, alpha=0.08, color="#0ea5e9")
        ax4.set_xlabel("False Positive Rate", fontsize=10)
        ax4.set_ylabel("True Positive Rate", fontsize=10)
        ax4.set_title("ROC Curve", fontsize=12, fontweight="bold", color="#0a2540")
        ax4.legend(fontsize=10, frameon=False)
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.set_facecolor("#f8fbff")
        fig4.patch.set_facecolor("#f0f6fd")
        st.pyplot(fig4, use_container_width=True)

    # Classification report
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Full Classification Report**")
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).T.round(3)
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
        use_container_width=True,
    )

    # Feature importance (RF only)
    if model_choice == "Random Forest":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">Feature Importance</div>', unsafe_allow_html=True)
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        bars = ax5.barh(
            [feature_cols[i] for i in sorted_idx],
            importances[sorted_idx],
            color=[
                f"rgba({int(10+220*(v/importances.max()))},165,{int(233*(1-v/importances.max()))},1)"
                for v in importances[sorted_idx]
            ],
            edgecolor="none", height=0.6
        )
        ax5.set_xlabel("Importance Score", fontsize=10)
        ax5.set_title("Random Forest Feature Importances", fontsize=12,
                       fontweight="bold", color="#0a2540")
        ax5.spines[["top", "right", "left"]].set_visible(False)
        ax5.set_facecolor("#f8fbff")
        fig5.patch.set_facecolor("#f0f6fd")
        st.pyplot(fig5, use_container_width=True)
