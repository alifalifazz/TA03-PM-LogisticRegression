# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load dataset
df = pd.read_csv("data/payments.csv")

# 2. Pisahkan fitur & target
X = df.drop(columns=["ID", "default payment next month"])
y = df["default payment next month"]

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Tangani imbalance dengan SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# 8. Simpan model & scaler
joblib.dump(model, "models/logreg_credit_model.joblib")
joblib.dump(scaler, "models/credit_scaler.joblib")

print("\n✅ Model dan scaler berhasil disimpan di folder 'models/'")
