import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# ======================== SETUP ========================
st.set_page_config(page_title="Revenue Leakage Smart Dashboard", layout="wide")
st.title("💸 Revenue Leakage Detection")

# ======================== LOAD DATA ========================
@st.cache_data
def load_data():
    return pd.read_csv("Synthetic_Financial_Transactions.csv")

df = load_data()
features = ["Invoice_Amount", "Discount_Applied", "Refund_Issued"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ======================== SIDEBAR ========================
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox("^-^ Select ML Model", ["Isolation Forest", "DBSCAN", "K-Means"])
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

# ======================== FILTER BASED ON VIEW ========================
if view_choice == "Model":
    filtered_df = df_model[df_model["Model_Anomaly"] == "Anomaly"]
    view_label = "Model-Detected"
else:
    filtered_df = df_model[df_model["Anomaly_Tag"] != "Normal"]
    view_label = "Ground Truth"

# Apply anomaly type filter (works only for Ground Truth anomalies)
if view_choice == "Ground Truth":
    filtered_df = filtered_df[filtered_df["Anomaly_Tag"].isin(anomaly_filter)]

# ======================== DISPLAY METRICS ========================
st.subheader(f"📊 {view_label} Anomalies Overview ({model_choice})")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df_model))
col2.metric("Anomalies Detected", len(filtered_df))
col3.metric("Model", model_choice)

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

st.caption("Shruthika — 23020023")


