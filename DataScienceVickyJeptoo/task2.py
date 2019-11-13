import pandas
import sklearn
df=pandas.read_csv('HR.csv')
print(df)
print(df.shape)
#print(df.isnull().sum())
#previous_year_rating is missing,fill the missing
median=df['previous_year_rating'].median()
df['previous_year_rating'].fillna(median,inplace=True)
print(df.isnull().sum())
#Encode gender/recruitment_channel to 0,1s
print(df.groupby('is_promoted').size())#0's is 928 and 1's is 93 hence we need to balance

#Encoding
df['gender'].replace({'m':0,'f':1},inplace=True)
df['recruitment_channel'].replace({'sourcing':0,'other':1,'referred':3},inplace=True)


array=df.values
X=array[:,0:9]
y=array[:,9]
#it's imbalanced so we use SMOTE to balance
from sklearn import model_selection
import imblearn
from imblearn.over_sampling import SMOTE
sm=SMOTE(ratio='auto',kind='regular',random_state=42)
X_sampled, y_sampled=sm.fit_sample(X,y)

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X_sampled,y_sampled,test_size=0.30,random_state=42)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
print('Model Training on data ......please wait')
model.fit(X_train,Y_train)#model is fitted with data for training
print('Learning Completed')
predictions=model.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))

#We need to know in the 30% of data ,how many 1 or 0 were there in diabetes column:
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))

#newperson data
newpeople=[[0,0,1,31,3,4,0,0,85],[1,1,1,26,2,4,0,0,47],[1,2,1,35,4,2,0,0,52]]
observation = model.predict(newpeople)
print(observation)
print('Predicted',observation)
