#!/usr/bin/env python
# coding: utf-8

import pandas as pd
df = pd.read_csv("cardekho.csv")
df
df.dtypes

from sklearn.model_selection import train_test_split

X = df.drop(['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'max_power'], axis=1)
y = df['selling_price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f"mean_absolute_error: {mean_absolute_error(y_pred, y_test)}")
print(f"mean_squared_error:  {mean_squared_error(y_pred, y_test)}")
print(f"r2_score (variance): {r2_score(y_pred, y_test)}")

import numpy as np

# % of predictions within +-10% of the true price
acc_within_10pct = (np.abs(y_test - y_pred) <= 0.20 * y_test).mean() * 100
print(f"Accuracy within ±10%: {acc_within_10pct:.2f}%")

