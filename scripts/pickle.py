import pickle

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

linear_model = LinearRegression().fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)
print("Test score: ", r2_score(y_test, y_pred_test))

Path("models").mkdir(parents=True, exist_ok=True)
pickle.dump(linear_model, open('models/model.pkl', 'wb'))

pickle_model = pickle.load(open('models/model.pkl', 'rb'))

y_pred_test = pickle_model.predict(X_test)

print("Test score: ", r2_score(y_test, y_pred_test))
