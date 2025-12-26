import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("dataset.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("house_model.pkl", "wb"))
print("Model trained & saved successfully")
