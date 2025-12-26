from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Yahin par model train hoga (koi .pkl nahi)
data = pd.read_csv("dataset.csv")
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])

    price = model.predict([[area, bedrooms, bathrooms]])[0]
    return render_template("result.html", price=f"₹ {int(price):,}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

