import pandas
import matplotlib
df=pandas.read_csv("power.csv",parse_dates=['Date'])
#modcom.co.ke/datascience
print(df)
print(df.isnull().sum())
import matplotlib
#Time series tools
#https://modcom.co.ke/datascience/Visualization/timeSeries_pandas.pdf
#Date formats:2/2/2013,1st Oct 2019,03-06-2006,13-Jun-2014,Feb 24 2005,2008-05-09
#date=pandas.to_datetime(['2/2/2013','13-Jun-2014','2008-05-09','Feb 25 2009','25th January 2006','Feb/03/2019'])
#Convert all time functions
#print(date)#yy/mm/dd
#df=pandas.read_csv("school.csv",parse_dates=['bday','enrolldate'])
#parse dates converts any date format to yy/mm/dd format
#print(date)

#line plot
#Power consumption by years
#Set date as index,starting column
df=df.set_index('Date')
import matplotlib.pyplot as plt
#using loc,to locate
print(df.loc['2010-02-20':'2010-12-20'])#20th Feb - 20th Dec 2010
print('============================================================')
print(df.loc['2010-02'])#Feb only
print('==============================')
print(df.loc['2010'])#all months in 2010
print('==========================================================')
print(df.loc['2010-01-02'])#one day
#plotting
df.loc['2010','Consumption'].plot()#whole 2010
plt.title('Electricity Consumption in 2010')
plt.xlabel('Year 2010')
plt.ylabel('Consumption(GWh)')
plt.show()

df.loc['2010-02-20':'2012-02-20','Consumption'].plot()
plt.title('Electricity Consumption between Feb 2010 to Feb 2012')
plt.xlabel('Year 2010')
plt.ylabel('Consumption(GWh)')
plt.show()

#Remove consumption...
df.loc['2010-02-01':'2013-02-20',].plot()
plt.title('Consumption of Power between Feb 2010 to Feb 2013')
plt.xlabel('Year 2010')
plt.ylabel('Consumption of Power(GWh)')
plt.show()

#Wind and Solar
df.loc['2010-02-01':'2013-02-20',['Wind','Solar']].plot()
plt.title('Consumption of Wind and Solar between Feb 2010 to Feb 2013')
plt.xlabel('Year 2010')
plt.ylabel('Wind and Solar(GWh)')
plt.show()

#Research on;resample data by dates
#
