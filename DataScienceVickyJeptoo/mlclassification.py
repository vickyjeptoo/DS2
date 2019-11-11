#data-modcom.co/datascience/bankupdated1
import pandas
import sklearn

df=pandas.read_csv("bankupdated1.csv")
print(df)
#In ml you cannot deal with empties,if there were empties in your dataset you have to fill the empties
#cleaning -the ml cannot deal with text
#eliminate non-determining factors i.e day contacted,duration of call
subset=df[['age','housing','loan','job','education','marital','balance','default','campaign','previous','y']]
#print(subset)
#Data encoding
subset['education'].replace({'primary':0,'secondary':1,'tertiary':2,'unknown':3},inplace=True)
subset['marital'].replace({'married':0,'single':1,'divorced':2,'unknown':3},inplace=True)
subset['default'].replace({'yes':0,'no':1,'unknown':2,},inplace=True)
subset['job'].replace({'admin.':0,
                      'blue-collar':1,
                      'housemaid':2,
                      'entrepreneur':3,
                      'management':4,
                      'retired':5,
                      'self-employed':6,
                      'student':7,
                      'services':8,
                      'technician':9,
                      'unemployed':10,
                      'unknown':11},
                     inplace=True)
subset['housing'].replace({'yes':0,'no':1,'unknown':2,},inplace=True)
subset['loan'].replace({'yes':0,'no':1,'unknown':2,},inplace=True)
#split the data into training and test data at a ratio of 70:30
#splitting is done randomly
from sklearn import model_selection
#code doesn't know x and y,
array=subset.values
X=array[:,0:10] # :means all rows from columns 0.....9
y=array[:,10] #10 counted here
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,y,test_size=0.30,random_state=42)
#above we have splitted our data.X_train and Y_train (70%)
#classification models KnearestNeighbours,Logistic Regression models
#Linear Regression,DecisionTrees,Random Forest,Support Vector Machines.
#Gausian Models

#try new models
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#object=KNeighborsClassifier()
object=DecisionTreeClassifier()
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
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
#model did not learn well from the yes ,hence imbalanced

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))
#[[1192 13] out of 1205 it got 1192 for no's
#[149 3] out of the 152 it got only 3 for the yes
#model failed terribly in the 'yes',hence it needs some improvement in yes outcome
newperson=[[25,1,0,5,2,0,2500,1,2,0]] #load to predict for a new customer
observation = object.predict(newperson)
print(observation)
#to improve perfomance on yes
#modcom.co.ke/datascience
#google;pima indians dataset