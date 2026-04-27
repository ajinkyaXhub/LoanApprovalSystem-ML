from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import webbrowser
import threading

app = Flask(__name__)

df = pd.read_csv("loan.csv")

df.columns = df.columns.str.strip()
df.ffill(inplace=True)

df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

le_edu = LabelEncoder()
le_emp = LabelEncoder()
le_status = LabelEncoder()

df['education'] = le_edu.fit_transform(df['education'])
df['self_employed'] = le_emp.fit_transform(df['self_employed'])
df['loan_status'] = le_status.fit_transform(df['loan_status'])

X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    dependents = int(request.form['dependents'])
    education = le_edu.transform([request.form['education']])[0]
    self_emp = le_emp.transform([request.form['self_employed']])[0]
    income = int(request.form['income'])
    loan_amount = int(request.form['loan_amount'])
    loan_term = int(request.form['loan_term'])
    cibil = int(request.form['cibil'])
    res_assets = int(request.form['res_assets'] or 0)
    com_assets = int(request.form['com_assets'] or 0)
    lux_assets = int(request.form['lux_assets'] or 0)
    bank_assets = int(request.form['bank_assets'] or 0)

    total_assets = res_assets + com_assets + lux_assets + bank_assets

    rejection_reason = None

    if cibil < 550:
        rejection_reason = "Very Low CIBIL Score"

    elif income < 150000 and loan_amount > 1000000:
        rejection_reason = "Income too low compared to requested loan"

    elif total_assets < (0.2 * loan_amount):
        rejection_reason = "Insufficient assets as backup"

    elif bank_assets < 30000 and loan_amount > 1200000:
        rejection_reason = "Low bank balance for high loan"

    elif dependents >= 5 and income < 300000:
        rejection_reason = "Too many dependents with low income"

    if rejection_reason:
        return render_template('index.html',
                               prediction_text="Rejected",
                               confidence="Rule-Based",
                               reason=rejection_reason)

    data = [[dependents, education, self_emp, income, loan_amount, loan_term,
             cibil, res_assets, com_assets, lux_assets, bank_assets]]

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0]

    confidence = round(max(probability) * 100, 2)

    result = "Approved" if prediction == 0 else "Rejected"

    return render_template('index.html',
                           prediction_text=result,
                           confidence=str(confidence) + "%",
                           reason="Based on ML Model Analysis")

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)