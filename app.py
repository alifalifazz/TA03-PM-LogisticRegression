from flask import Flask, render_template, request 
import numpy as np
import joblib

# Inisialisasi Flask
app = Flask(__name__)

# 1️⃣ Muat model & scaler
model = joblib.load("models/logreg_credit_model.joblib")
scaler = joblib.load("models/credit_scaler.joblib")

@app.route('/')
def home():
    return render_template('index.html')

# 2️⃣ Proses input & prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari form HTML
        features = [
            float(request.form['LIMIT_BAL']),
            int(request.form['SEX']),
            int(request.form['EDUCATION']),
            int(request.form['MARRIAGE']),
            int(request.form['AGE']),
            int(request.form['PAY_0']),
            int(request.form['PAY_2']),
            int(request.form['PAY_3']),
            int(request.form['PAY_4']),
            int(request.form['PAY_5']),
            int(request.form['PAY_6']),
            float(request.form['BILL_AMT1']),
            float(request.form['BILL_AMT2']),
            float(request.form['BILL_AMT3']),
            float(request.form['BILL_AMT4']),
            float(request.form['BILL_AMT5']),
            float(request.form['BILL_AMT6']),
            float(request.form['PAY_AMT1']),
            float(request.form['PAY_AMT2']),
            float(request.form['PAY_AMT3']),
            float(request.form['PAY_AMT4']),
            float(request.form['PAY_AMT5']),
            float(request.form['PAY_AMT6'])
        ]

        # Normalisasi input sesuai scaler
        scaled_features = scaler.transform([features])

        # Prediksi
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        # Interpretasi hasil
        hasil = "Berisiko" if prediction == 1 else "Tidak Berisiko"
        return render_template('result.html', prediction=hasil, probability=round(probability*100, 2))

    except Exception as e:
        return f"Terjadi kesalahan: {e}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(__import__('os').environ.get('PORT', 5000)), debug=True)
