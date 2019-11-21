import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("train.csv")
test=pd.read_csv('test.csv')
#below are the parameters and their description
#PassengerId: Passenger Identity
#Survived: Whether passenger survived or not
#Pclass: Class of ticket
#Name: Name of passenger
#Sex: Sex of passenger (Male or Female)
#Age: Age of passenger
#SibSp: Number of sibling and/or spouse travelling with passenger
#Parch: Number of parent and/or children travelling with passenger
#Ticket: Ticket number
#Fare: Price of ticket
#Cabin: Cabin number

#1.Sex v/s survived
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
plt.show()

#You can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true
#for women but not fully. For women the survival chances are higher between 14 and 40.
#For men the probability of survival is very low between the age of 5 and 18, but that isn’t true for women.
# Another thing to note is that infants also have a little bit higher probability of survival.

#2.Embarked, Pclass and Sex:
FacetGrid = sns.FacetGrid(train, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
plt.show()
#Embarked seems to be correlated with survival, depending on the gender.
#Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C.
#Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.

#3.Pclass
sns.barplot(x='Pclass', y='Survived', data=train)
plt.show()
#Pclass is contributing to a persons chance of survival,especially if this person is in class 1.

#4.SibSp and Parch
#we review who is alone and who has relatives,spouse or children in the titanic
data = [train, test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train['not_alone'].value_counts()

axes = sns.factorplot('relatives','Survived',
                      data=train, aspect = 2.5, )
plt.show()
#from the above we can see if you have less than 3 relatives you have a higher chance of survival

#5.Relation between passengers' survival and booking class


def make_pivot(param1, param2):
    df_slice = train[[param1, param2, 'PassengerId']]
    slice_pivot = df_slice.pivot_table(index=[param1], columns=[param2], aggfunc=np.size, fill_value=0)

    p_chart = slice_pivot.plot.bar()
    for p in p_chart.patches:
        p_chart.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.01))

    return slice_pivot
    return p_chart


make_pivot ('Survived','Pclass')
plt.show()

#the passangers in Pclass 3 did not have a high survival rate
#6.Relation between passengers' survival and their sex


make_pivot ('Survived','Sex')
plt.show()
#Females had a higher survival chance

#7.Heatmaps to show where we are missing data!
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
#Roughly 20 percent of the Age data is missing. The proportion of Age missing
#Looking at the Cabin column, it looks like we are just missing too much of that data

#8.Heatmap of the entire data
sns.heatmap(train.corr(), annot=True)
plt.show()
#9.Piechart to show distribution of gender
fig,ax=plt.subplots()
train.groupby('Sex').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Gender in the titanic')
ax.set_ylabel('')
plt.show()
#10.boxplot of the fare

sns.boxplot(train["Fare"])
plt.show()

#https://justpaste.it/28jks