import pandas
import matplotlib

df = pandas.read_csv('school.csv')
print(df)
print(df.isnull().sum())
import matplotlib.pyplot as plt
#piechart for rank
df['Rank'].fillna(0,inplace=True)
df['Rank'].replace({1:'Freshman',2:'Sophomore',3:'Junior',4:'Senior',0:'Unknown Rank'},inplace=True)
fig,ax=plt.subplots()
df.groupby('Rank').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Students Rank')
ax.set_ylabel('')
plt.show()
#Bar Charts
#Used to present categorical data vs numeric
#Rank by Math
x=df.groupby('Rank')['SleepTime'].mean().plot(kind='bar',color='green')
ax.set_title('Students SleepTime mean Score by Rank')
ax.set_xlabel('Students Rank')
ax.set_ylabel('SleepTime')
plt.show()
#print(x)
#justpaste.it/25daq
#https://towardsdatascience.com/bar-chart-race-in-python-with-matplotlib-8e687a5c8a41
fig,ax=plt.subplots()
ax.barh(df['Rank'],df['SleepTime'])
ax.set_title('Students SleepTime mean Score by Rank')
ax.set_xlabel('Students Rank')
ax.set_ylabel('SleepTime')
plt.show()
#Stacked bar charts
#used when we have 2 or more categorical variables and atleast 1 numeric
#Find the Math performance by Gender and Rank
df['Gender'].fillna(2,inplace=True)
df['Gender'].replace({0:'Male',1:'Female',2:'Unknown Gender'},inplace=True)
df['Smoking'].fillna(3,inplace=True)
df['Smoking'].replace({0:'Nonsmoker',1:'Pastsmoker',2:'Current',3:'Unknown Smoking'},inplace=True)
#y=df.groupby(['Gender','Rank','Smoking'])['Math'].mean()
y=df.groupby(['Gender','Rank'])['Math'].mean()
print(y)
#plot stacked
y=df.groupby(['Rank','Gender'])['Math'].mean().unstack().plot(kind='bar',stacked=True)
plt.title('Comparing Gender and Rank vs Math Performance')
plt.xlabel('Gender')
plt.ylabel('Math Marks')
plt.legend(loc='best')
plt.show()
#plot unstacked
y=df.groupby(['Rank','Gender'])['Math'].mean().unstack().plot(kind='bar',stacked=False)
plt.title('Comparing Gender and Rank vs Math Performance')
plt.xlabel('Gender')
plt.ylabel('Math Marks')
plt.legend(loc='best')
plt.show()
#if you pass True you get a stacked plot and if you use false you get an unstacked plot
#do a stacked/unstacked for HowCommute,LiveOnCampus and StudyTime
df['HowCommute'].fillna(6,inplace=True)
df['HowCommute'].replace({1:'Walk',2:'Bike',3:'Car',4:'Public Transit',5:'Other',6:'UnknownHowCommute'},inplace=True)
#LiveOnCampus
#1.Oncampus, 0,Offcampus
df['LiveOnCampus'].fillna(3,inplace=True)
df['LiveOnCampus'].replace({0:'Off-Campus',1:'On-Campus',3:'UnknownLOC'},inplace=True)
y=df.groupby(['HowCommute','LiveOnCampus'])['StudyTime'].mean().unstack().plot(kind='bar',stacked=False)
plt.title('How Commuting and Living on Campus affects StudyTime')
plt.xlabel('Means of Commuting')
plt.ylabel('Study Time')
plt.legend(loc='best')
plt.show()
#install xampp server