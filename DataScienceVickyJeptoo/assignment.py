#https://modcom.co.ke/datascience/sales.csv
#Get sales.csv
import pandas
df=pandas.read_csv('sales.csv')
print(df)
print(df.isnull().sum())
import matplotlib.pyplot as plt

#histograms of ext price
#pie chart for show category distribution
#Which category brought more cash?
#Which company brought the highest cash?
#Which month,year has highest sales
#What is the correlation between unit price and ext price
#jupyterlab online free editor;jupyter.org

