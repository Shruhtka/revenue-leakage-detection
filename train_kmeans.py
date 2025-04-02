import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset (make sure this file is in the same folder)
df = pd.read_csv("Synthetic_Financial_Transactions.csv")

# Preprocess
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')
df['PaymentDelay'] = (df['Payment_Date'] - df['Transaction_Date']).dt.days.fillna(0)
df['DiscountApplied'] = df['Discount_Applied']
df['RefundAmount'] = df['Refund_Issued']
df['InvoiceAmount'] = df['Invoice_Amount']

# Select Features
X = df[['DiscountApplied', 'RefundAmount', 'InvoiceAmount', 'PaymentDelay']]

# Train KMeans model
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X)

# Save the trained model
joblib.dump(kmeans_model, "kmeans_model.pkl")

print("✅ KMeans model saved as 'kmeans_model.pkl'")
