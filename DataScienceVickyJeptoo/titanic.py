#Predict survival on the Titanic and get familiar with ML basics
#link:https://justpaste.it/3lkgf
#train_df['Embarked'].describe()
# linear algebra
import matplotlib
import numpy as np

# data processing
import pandas


# data visualization
import seaborn as sns
#%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
df=pandas.read_csv('train.csv')
df2=pandas.read_csv('test.csv')
print(df)
print(df.shape)
#print(df.isnull().sum())

#177 missing :age
#687 missing :cabin
#2 missing values :Embarked
median=df['Age'].median()
df['Age'].fillna(median,inplace=True)
#print(df.isnull().sum())

df['Embarked'].fillna(0,inplace=True)
df['Embarked'].replace({1:'S',2:'C',3:'Q'},inplace=True)
medianEmbarked=df['Embarked'].median
df['Embarked'].fillna(medianEmbarked,inplace=True)
print(df.isnull().sum())

subset=df[['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
print(subset)

#Converting Features:

df.info()

#Converting “Fare” from float to int64, using the “astype()” function pandas provides:
data = [df, df2]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

df.info()
df2.info()
#We will use the Name feature to extract the Titles from the Name, so that we can build a new feature out of that.
data = [df, df2]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
df = df.drop(['Name'], axis=1)
df2 = df2.drop(['Name'], axis=1)

df.info()
df2.info()
#convert Sex from object to numeric
genders = {"male": 0, "female": 1}
data = [df, df2]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

df.info()
df2.info()
#we drop Cabin feature
df = df.drop(['Cabin'], axis=1)
df2 = df2.drop(['Cabin'], axis=1)
df.info()
df2.info()
#Ticket
df['Ticket'].describe()

#Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories.
# So we will drop it from the dataset.

df = df.drop(['Ticket'], axis=1)
df2 = df2.drop(['Ticket'], axis=1)
#convert Embarked from object to numeric
ports = {"S": 0, "C": 1, "Q": 2}
data = [df, df2]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna(0)
        dataset['Embarked'] = dataset['Embarked'].astype(int)
df.info()
df2.info()
#Age
#we need to convert the ‘age’ feature;from float to integer
#Then we create various age groups
data = [df, df2]

for dataset in data:
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


df.info()#age has been converted to int
#df['Age'].value_counts()

#FARE
#For the ‘Fare’ feature, we need to do the same as with the ‘Age’ feature.
# But it isn’t that easy, because if we cut the range of the fare values into a few equally big categories,
# 80% of the values would fall into the first category.
# Fortunately, we can use sklearn “qcut()” function, that we can use to see, how we can form the categories.
data = [df, df2]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#Drop PassengerId:

df = df.drop(['PassengerId'], axis=1)
#df2 = df2.drop(['PassengerId'], axis=1)

print(df.head(10))

#WE START TRAINNING THE MODEL
from sklearn import model_selection

array=df.values
X=array[:,1:8] # :means all rows from columns 0.....9
y=array[:,0] #10 counted here
from imblearn.over_sampling import SMOTE
sm=SMOTE(ratio='auto',kind='regular',random_state=42)
X_sampled, y_sampled=sm.fit_sample(X,y)

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X_sampled, y_sampled,test_size=0.30,random_state=42)

#from sklearn.tree import DecisionTreeClassifier
#object=KNeighborsClassifier() #77.61%
#object=DecisionTreeClassifier()#80.97%
#object=linear_model.SGDClassifier()#79.85%
#object=RandomForestClassifier()#80.22
#object=LogisticRegression()#80.97
object=KNeighborsClassifier()#77.61%
#Best models are DecisionTreeClassifier & LogisticRegression

print('Model Training on data ......please wait')
object.fit(X_train,Y_train)#model is fitted with data for training
print('Learning Completed')

#ask model to predict y for X_test
predictions=object.predict(X_test)
print(predictions)
#compare predictions with Y_test
#Accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))  #88% accuracy
#if below 75% you can change model
#Stochastic Gradient Descent (SGD):



from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
#model did not learn well from the yes ,hence imbalanced

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))


newpwerson = [[3,2,0,45,45,23,233]]
observe = object.predict(newpwerson)
print(observe)




#reference from :https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8