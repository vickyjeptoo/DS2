#regression-predicting numerics not yes,no/0,1 eg:population,blood-pressure
#Encoding & filling empties is a must
#Linear regression models have many real-world applications in an array of industries such as
#economics (e.g. predicting growth), business (e.g. predicting product sales, employee performance),
#social science (e.g. predicting political leanings from gender or race), healthcare (e.g. predicting
#blood pressure levels from weight, disease onset from biological factors), and more.

import pandas
import sklearn

df=pandas.read_csv("Advertising.csv")
print(df)
print(df.shape)
print(df.isnull().sum())

array=df.values
X=array[:,1:4] # :means all rows from columns 1.....3,0 is a primary key
y=array[:,4]
#now we have equal samples
from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,y,test_size=0.30,random_state=42)

#models:linear regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

model=LinearRegression()
model.fit(X_train,Y_train)
print('Learning completed!')

# ask model to predict (X_test)
predictions=model.predict(X_test)
print(predictions)
#check accuracy/performance of the model
from sklearn.metrics import r2_score #used to give percentage of accuracy
print('R squared:',r2_score(Y_test,predictions)) #86% accuracy

from sklearn.metrics import mean_squared_error
print('Mean squared error',mean_squared_error(Y_test,predictions))
#above its squared so we find square root#1.something

new=[[204.1,32.9,46]]
observation=model.predict(new)
print('You will sell',observation,'TV sets')
#we need to plot the regression-line
#to show model predicted well they should be close to the line

import matplotlib.pyplot as plt
plt.style.use('seaborn')
fig,ax=plt.subplots()
ax.scatter(Y_test,predictions)
ax.plot(Y_test,Y_test)
ax.set_title('Predictions vs Y_test')
ax.set_xlabel('Y test')
ax.set_ylabel('Predictions')
plt.show()

