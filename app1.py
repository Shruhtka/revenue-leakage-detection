import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt

# ======================== SETUP ========================
st.set_page_config(page_title="Revenue Leakage Smart Dashboard", layout="wide")
st.title("💸 Revenue Leakage Detection")

# ======================== LOAD DATA ========================
required_columns = ["Transaction_ID", "Invoice_Amount", "Discount_Applied", "Refund_Issued", "Transaction_Type", "Anomaly_Tag"]

@st.cache_data
def load_default_data():
    return pd.read_csv("Synthetic_Financial_Transactions.csv")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload Your Own CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in required_columns):
            st.sidebar.error("❌ Uploaded file is missing required columns.")
            df = load_default_data()
        else:
            st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()

features = ["Invoice_Amount", "Discount_Applied", "Refund_Issued"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ======================== SIDEBAR ========================
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox("🧠 Select ML Model", ["Isolation Forest", "DBSCAN", "K-Means"])
view_choice = st.sidebar.radio("🎯 Show anomalies detected by:", ["Model", "Ground Truth"])
anomaly_filter = st.sidebar.multiselect("🔍 Filter Anomaly Type", options=df["Anomaly_Tag"].unique(), default=list(df["Anomaly_Tag"].unique()))

# ======================== MODEL PREDICTIONS ========================
df_model = df.copy()

if model_choice == "Isolation Forest":
    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    preds = model.fit_predict(X_scaled)
    df_model["Model_Anomaly"] = pd.Series(preds).map({1: "Normal", -1: "Anomaly"})

elif model_choice == "DBSCAN":
    model = DBSCAN(eps=1.5, min_samples=5)
    preds = model.fit_predict(X_scaled)
    df_model["Model_Anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]

elif model_choice == "K-Means":
    model = KMeans(n_clusters=2, random_state=42)
    preds = model.fit_predict(X_scaled)
    anomaly_cluster = pd.Series(preds).value_counts().idxmin()
    df_model["Model_Anomaly"] = ["Anomaly" if p == anomaly_cluster else "Normal" for p in preds]

# ======================== METRICS FOR SELECTED MODEL ========================
y_true = df_model["Anomaly_Tag"].apply(lambda x: 1 if x != "Normal" else 0)
y_pred = df_model["Model_Anomaly"].apply(lambda x: 1 if x != "Normal" else 0)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# ======================== FILTER BASED ON VIEW ========================
if view_choice == "Model":
    filtered_df = df_model[df_model["Model_Anomaly"] == "Anomaly"]
    view_label = "Model-Detected"
else:
    filtered_df = df_model[df_model["Anomaly_Tag"] != "Normal"]
    view_label = "Ground Truth"

if view_choice == "Ground Truth":
    filtered_df = filtered_df[filtered_df["Anomaly_Tag"].isin(anomaly_filter)]

# ======================== DISPLAY METRICS ========================
st.subheader(f"📊 {view_label} Anomalies Overview ({model_choice})")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df_model))
col2.metric("Anomalies Detected", len(filtered_df))
col3.metric("Model", model_choice)

st.subheader("📈 Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Precision", f"{precision:.2f}")
col2.metric("📞 Recall", f"{recall:.2f}")
col3.metric("📐 F1-Score", f"{f1:.2f}")

# ======================== DATA TABLE ========================
st.markdown("### 📋 Anomaly Table")
st.dataframe(filtered_df[["Transaction_ID", "Invoice_Amount", "Discount_Applied", "Refund_Issued", "Transaction_Type", "Anomaly_Tag", "Model_Anomaly"]], use_container_width=True)

# ======================== CHART ========================
st.markdown("### 📈 Invoice vs Refund Plot")
fig = px.scatter(filtered_df, x="Invoice_Amount", y="Refund_Issued", color="Transaction_Type",
                 symbol="Model_Anomaly" if view_choice == "Model" else "Anomaly_Tag",
                 hover_data=["Transaction_ID", "Discount_Applied"], title="Anomalies Scatter Plot")
st.plotly_chart(fig, use_container_width=True)

# ======================== DOWNLOAD ========================
st.markdown("### 💾 Download Anomalies")
download_csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=download_csv, file_name=f"{view_label.replace(' ', '_')}_{model_choice}.csv", mime='text/csv')

# ======================== SHAP EXPLAINABILITY ========================
st.markdown("---")
st.subheader("🧠 SHAP Explainability (Only for Isolation Forest)")

if model_choice == "Isolation Forest":
    iso_model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    iso_model.fit(X_scaled)
    explainer = shap.Explainer(iso_model, X_scaled)
    shap_values = explainer(X_scaled)

    st.markdown("### 🔝 Top Contributing Features (Global Summary)")
    fig_summary, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=5, show=False)
    st.pyplot(fig_summary)

    st.markdown("### 🔍 Explain a Specific Transaction")
    transaction_idx = st.slider("Select a transaction index", 0, len(df_model)-1, 0)
    selected_tx = df_model.iloc[transaction_idx]

    st.write(f"**Transaction ID:** {selected_tx['Transaction_ID']}")
    st.write(f"**Amount:** £{selected_tx['Invoice_Amount']} | Refund: £{selected_tx['Refund_Issued']} | Discount: £{selected_tx['Discount_Applied']}")
    st.write(f"**Model Label:** {selected_tx['Model_Anomaly']}")

    st.markdown("#### 📌 SHAP Force Plot Explanation")
    fig_force = shap.plots.force(shap_values[transaction_idx], matplotlib=True, show=False)
    st.pyplot(fig_force)

else:
    st.info("ℹ️ SHAP explanations are only supported for Isolation Forest.")
