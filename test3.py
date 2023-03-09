from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

data = pd.read_csv('train.csv')

X = data[['mip', 'stdip', 'ekip', "sip",'mc', 'stdc', 'ekc', 'sc']]
# X = pd.DataFrame(preprocessing.normalize(X, axis=0))
X = scaler.fit_transform(X)
#X=X.astype('float')
y = data[['target']]
#y=y.astype('float')
X = pd.DataFrame(X)
print(X[0].mean())

model = LogisticRegression(random_state=15, solver='lbfgs')
model.fit(X, y)

new = pd.read_csv('test.csv')
print(model.predict_proba(new))