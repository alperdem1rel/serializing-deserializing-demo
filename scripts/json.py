import json

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

linear_model = LinearRegression().fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)
print("Test score: ", r2_score(y_test, y_pred_test))


model_param = {}

model_param['coef'] = list(linear_model.coef_)
model_param['intercept'] = linear_model.intercept_.tolist()

json_txt = json.dumps(model_param, indent=4)
print(json_txt)

Path("models").mkdir(parents=True, exist_ok=True)
with open('models/regressor_param.txt', 'w') as file:
    
    file.write(json_txt)
    
with open('models/regressor_param.txt', 'r') as file:
    
    json_text=json.load(file)


json_model = LinearRegression()

json_model.coef_ = np.array(json_text['coef'])
json_model.intercept_ = np.array(json_text['intercept'])

y_pred_test = json_model.predict(X_test)

print("Test score: ", r2_score(y_test, y_pred_test))
