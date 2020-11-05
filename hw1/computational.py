import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv('data.csv')


linear_model = LinearRegression(fit_intercept=False, copy_X=True)


columns_in_regressions = [
    ['X_1', 'X_2', 'X_3'],
    ['X_2'],
    ['X_1', 'X_2'],
    ['X_3', 'X_2']
]

for columns_in_regression in columns_in_regressions:
    linear_model.fit(data[columns_in_regression], data['Y'])
    print('Included variables: ', columns_in_regression)
    print('Parameter Estimates: ', linear_model.coef_, '\n')
