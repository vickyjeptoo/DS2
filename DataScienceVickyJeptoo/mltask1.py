import pandas
import sklearn
df=pandas.read_csv('pimadata.csv')
print(df)
print(df.shape)
print(df.isnull().sum())

#predicting if someone has diabetes
#y is diabetes
subset=df[['num_preg','glucose_conc','diastolic_bp','skin_thickness','insulin','bmi','diab_pred','age','diabetes']]
from sklearn import model_selection
array=subset.values
X=array[:,0:8]
y=array[:,8]
#we hsve 500 negatives and 268 positives

import imblearn  #used to build data until the yes and no are equal;to balance
from imblearn.over_sampling import SMOTE
sm=SMOTE(ratio='auto',kind='regular')
X_sampled, y_sampled=sm.fit_sample(X,y)
#now we have equal samples

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X_sampled,y_sampled,test_size=0.30,random_state=42)

#import 5 models and compare
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
object=LinearDiscriminantAnalysis()
print('Model Training on data ......please wait')
object.fit(X_train,Y_train)#model is fitted with data for training
print('Learning Completed')

predictions=object.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))

#We need to know in the 30% of data ,how many 1 or 0 were there in diabetes column:
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))
#the model did not work correctly,more no's than yes's
#we get synthetic data to balance the yes and nos
#read on synthetic data,artificially manufactured data rather than generated data by real-world events
newperson=[[6,148,72,35,0,33.6,0.627,50]]
observation = object.predict(newperson)
print(observation)