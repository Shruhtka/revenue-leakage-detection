import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
import joblib

print("📂 Current save location:", os.getcwd())

# Load and preprocess data
df = pd.read_csv("Synthetic_Financial_Transactions.csv")
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')
df['PaymentDelay'] = (df['Payment_Date'] - df['Transaction_Date']).dt.days.fillna(0)
df['DiscountApplied'] = df['Discount_Applied']
df['RefundAmount'] = df['Refund_Issued']
df['InvoiceAmount'] = df['Invoice_Amount']
X = df[['DiscountApplied', 'RefundAmount', 'InvoiceAmount', 'PaymentDelay']]

# Train and save Isolation Forest
if_model = IsolationForest(contamination=0.1, random_state=42)
if_model.fit(X)
joblib.dump(if_model, "if_model.pkl")
print("✅ if_model.pkl saved and fitted.")

# Train and save KMeans
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X)
joblib.dump(kmeans_model, "kmeans_model.pkl")
print("✅ kmeans_model.pkl saved and fitted.")

# Train and save DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_model.fit(X)
joblib.dump(dbscan_model, "dbscan_model.pkl")
print("✅ dbscan_model.pkl saved and fitted.")
