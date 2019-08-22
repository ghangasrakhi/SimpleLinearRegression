import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#read the data from file

df=pd.read_csv("FuelConsumptionCo2.csv")

#take a look at the dataset

df.head() 

#summarize the data

df.describe()

#select some of the features to explore 

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
#print(cdf)

#plot each of these features

viz=cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#plot each feature vs emission to see the linear relation

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='red')
plt.xlabel('Size of Engine')
plt.ylabel('Emissions')
plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='green')
plt.xlabel('Cylinders')
plt.ylabel('Emissions')
plt.show()

#select random rows for train and test data 
#80% for train data and 20% for test data

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

#train data distribution

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

#using skilearn package to model data

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

#the coefficients

print('coefficients:',regr.coef_)
print('Itercept:',regr.intercept_)

#plot the fit line over the data

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,regr.coef_[0][0]*train_x +regr.intercept_[0],'-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')

#evaluation
#we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
#There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: 
#Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
#Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
#Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
#R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat=regr.predict(test_x)

MAE=np.mean(np.absolute(test_y_hat-test_y))
print("mean absolute error: %.2f" %MAE)
print("sum squared error: %.2f" %np.mean((test_y_hat-test_y)**2))
print("R2-score :%.2f" %r2_score(test_y_hat,test_y))

























