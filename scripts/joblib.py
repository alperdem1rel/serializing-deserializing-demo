import joblib

import numpy as np
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

tree_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred_test = tree_model.predict(X_test)
print("Test score: ", r2_score(y_test, y_pred_test))


Path("models").mkdir(parents=True, exist_ok=True)

filename = 'models/model.joblib'
joblib.dump(tree_model, filename)

joblib_model = joblib.load(filename)

y_pred_test = joblib_model.predict(X_test)

print("Test score: ", r2_score(y_test, y_pred_test))
