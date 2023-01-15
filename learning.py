import pandas as pd
import numpy as np
import random
import nltk

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

data_reg = pd.read_csv('check\\reviews_analyzed.csv', header=0, usecols=['review', 'semantic', 'score'])

X = np.array(data_reg['semantic']).reshape(data_reg.shape[0], 1)
y = np.array(data_reg['score']).reshape(data_reg.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

regression = LinearRegression()
regression.fit(X_train, y_train)

predicted = regression.predict(X_test)
predicted = [np.sum(np.round(x)) for x in predicted]
dumb_predict = [random.randint(0, 10) for x in range(0, len(y_test))]

metrics_df = pd.DataFrame(data={'metric': ['MSE', 'MAE', 'Score'],
                                'regression_model': [mean_squared_error(y_test, predicted),
                                                     mean_absolute_error(y_test, predicted),
                                                     r2_score(y_test, predicted)],
                                'dumb_model': [mean_squared_error(y_test, dumb_predict),
                                               mean_absolute_error(y_test, dumb_predict),
                                               r2_score(y_test, dumb_predict)]})
comparison_df = pd.DataFrame(
    data={'actual': (y_test.reshape(1, len(y_test))[0]).tolist(), 'predicted': predicted, 'dumb': dumb_predict})

print(metrics_df)

comparison_df.to_csv('check\\comparison.csv')
