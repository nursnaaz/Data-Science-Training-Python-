# Cross Validation Regression MSE  

import pandas 
from sklearn import cross_validation 
from sklearn.linear_model import LinearRegression 
url = "https://goo.gl/sXleFv" 
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV'] 
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names) 
array = dataframe.values 
X = array[:,0:13] 
Y = array[:,13] 
num_folds = 10 
num_instances = len(X) 
seed = 7 
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed) 
model = LinearRegression() 
scoring = 'mean_squared_error' 
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())  