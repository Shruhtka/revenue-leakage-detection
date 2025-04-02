import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("Synthetic_Financial_Transactions.csv")
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')
df['PaymentDelay'] = (df['Payment_Date'] - df['Transaction_Date']).dt.days.fillna(0)
df['DiscountApplied'] = df['Discount_Applied']
df['RefundAmount'] = df['Refund_Issued']
df['InvoiceAmount'] = df['Invoice_Amount']
X = df[['DiscountApplied', 'RefundAmount', 'InvoiceAmount', 'PaymentDelay']]

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "if_model.pkl")
print("✅ if_model.pkl saved and trained successfully.")
