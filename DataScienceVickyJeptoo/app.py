#connect to data
# pandas-helps connect to data source,data cleaning,manipulation of data
#matplotlib-drawing plots/graphs,visualization

import pandas
import matplotlib

#modcom.co.ke/datascience,where we are getting our data from
data=pandas.read_csv("school.csv") #called a dataframe as well,df
#print(data)
#print(data['SleepTime']),particular column

#print(data.shape)#no of rows and columns

#dealing with empties,if missing rows are <(less than)5% of the total population they can be removed
#print(data.isnull().sum())#whole data set
#print(data['height'].isnull().sum()) ,in a particular column

#remove all empties
data.dropna(inplace=True)#true,parameter passed to update after removing
print(data.isnull().sum()) #results in empties
print(data.shape)#records reduce to 53 files
# working with data with no empties;it is easier
#quering with pandas
newdata=data[data['Gender']==1]
newdata2=data[data['Height']>=70]
newdata3=data[data['Smoking']==1]
print(newdata)
print(newdata2)
print(newdata3)

#basic statistics
print(data.describe())

#plotting
#plots,pie,bar,scatter,line,area charts,gantt charts,box,histogram
#density,heatmaps,stacked/unstacked,pareto,maps charts
#scatter plots....
#compare sleep and study time

import matplotlib.pyplot as plt
fig ,ax=plt.subplots()
ax.scatter(data['SleepTime'],data['StudyTime'],color='green',s=40)
ax.set_title('Distribution of SleepTime and StudyTime')
ax.set_xlabel('SleepTime')
ax.set_ylabel('StudyTime')
plt.show()
