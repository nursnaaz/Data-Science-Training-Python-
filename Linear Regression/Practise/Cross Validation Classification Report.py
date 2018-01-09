# Cross Validation Classification Report  

import pandas 
from sklearn import cross_validation 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report 

url = "https://goo.gl/vhm1eU" 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = pandas.read_csv(url, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
test_size = 0.33 
seed = 7 
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed) 
model = LogisticRegression() 
model.fit(X_train, Y_train) 
predicted = model.predict(X_test) 
report = classification_report(Y_test, predicted) 
print(report)  