import pandas
import matplotlib

df=pandas.read_csv("school.csv")
print(df)

#correlation-finding relationship between numerical variables
#Positive correlation: >0-+1 ,above 0.5 strong positive correlation
#Negative correlation:one going up the other  is going down:0->-1,above -0.5 strong negative correlation
#0,means no correlation
#scatter,heatmaps
#Objective;find the relationship between reading and writting
print(df.isnull().sum())
#filing missing data in a numeric column
#fill missing with median of the whole class(reading),fill empties with median
#find median of reading

medianReading=df['Reading'].median()
print(medianReading) #81.94

#Fill missing with the median
df['Reading'].fillna(medianReading,inplace=True)
#check empties
print(df.isnull().sum()) #no empties at median

meadianWriting=df['Writing'].median()
df['Writing'].fillna(meadianWriting,inplace=True)

import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax.scatter(df['Writing'],df['Reading'],color='blue',s=30)
ax.set_title('Relationship btn Writing vs Reading')
ax.set_xlabel('Writing Score')
ax.set_ylabel('Reading Score')
plt.show()

#height and weight
#ignored the empties but you should fill in empties.
fig,ax=plt.subplots()
ax.scatter(df['Height'],df['Weight'],color='red',s=30)
ax.set_title('Relationship between Height vs Weight')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
plt.show()
#correlation helps in training machine models
print(df.isnull().sum())

#Heat maps-shows correlation for multiple variables
import seaborn as sns
fig,ax=plt.subplots()
df_x=df[['Reading','Writing','Math','English','SleepTime','StudyTime']]  #narrow down th df to df_x
sns.heatmap(df_x.corr(),cmap='Blues',annot=True)  #Check other params that can be passed at cmap
ax.set_title('Relationship between Variables')
plt.show()

#Histogram,shows distribution of a single variable,either normal distribution
#Objective-check distribution of Height
fig,ax=plt.subplots()
ax.hist(df['Height'],color='blue')
ax.set_title('Distribution of Weight vs Height')
ax.set_xlabel('Height')
ax.set_ylabel('Frequency')
plt.show()

#multilevel histogram,overlay
#Google color pallette
fig,ax=plt.subplots()                               
ax.hist(df['Reading'],color='blue',alpha=0.3,label='Reading Score')
ax.hist(df['Math'],color='red',alpha=0.3,label='Math Score')
ax.hist(df['English'],color='orange',alpha=0.3,label='English Score')
ax.hist(df['Writing'],color='green',alpha=0.3,label='Writing Score')
ax.set_title('Distribution of Subjects')
ax.set_xlabel('Subjects')
ax.set_ylabel('Frequency')
ax.legend(loc='best')
plt.show()

#piechart :comparing proportion of values in percentage
#Gender distribution using pie chart
#Fill empties
#specify 0,1 and 2-unknown

#fill missing values
df['Gender'].fillna(2,inplace=True)
#label numeric data,0-Male,1-Female,2-Unknown
df['Gender'].replace({0:'Male',1:'Female',2:'Unknown'},inplace=True)
fig,ax=plt.subplots()
df.groupby('Gender').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Gender')
ax.set_ylabel('')
plt.show()

#Do a pie chart to show distribution of How Students Commute
#medianHowCommute=df['HowCommute'].median()
#print(medianHowCommute) #81.94
#print(df.groupby('HowCommute').size())shows distribution in numbers

#Fill missing with the median
#df['HowCommute'].fillna(medianHowCommute,inplace=True)
#check empties
#print(df.isnull().sum()) #no empties at median
df['HowCommute'].fillna(6,inplace=True)
df['HowCommute'].replace({1:'Walk',2:'Bike',3:'Car',4:'Public Transit',5:'Other',6:'Unknown'},inplace=True)
fig,ax=plt.subplots()
df.groupby('HowCommute').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of how students commute')
ax.set_ylabel('')
plt.show()
#todo:piecharts for rank,smoking.