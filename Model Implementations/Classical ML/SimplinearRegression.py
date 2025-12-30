'''
Implementation of Simple Linear Regression from stratch 
Justing using Math 

Pandas - for reading data 
Sklearn - for splitting data into train and validation set 
'''


import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('placement.csv')

print(data.head())

# Assigening data into varaibale X, and y 

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Spliting data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(data.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


class SimpelLinearRegression:


    '''
    Implementation of Simple Linear Regression (OLS)

    To make predictions using this we need to implement y = mx + c equation
    y and x are coming from data but we need to caculate m and c
    '''
    
    def __init__(self):

        self.m = None
        self.c = None

    def fit(self, X_train, y_train):
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):

            # for calcuation of m (Feature weight or slope)
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
            
        self.m = num/den 
        self.c = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):

        return (self.m * X_test) + self.c
    


lr = SimpelLinearRegression()

lr.fit(X_train, y_train)

print(f'coffecient_ : {lr.m}')
print(f'intercept_ : {lr.c}')

# Let's make predictions on the test set 

pred  = lr.predict(X_test)


