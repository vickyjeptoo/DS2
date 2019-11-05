import pandas
import matplotlib.pyplot as plt
import pymysql #to connect to mysql DB


conn = pymysql.connect("localhost","root","1234D!@#$","ds_vicky_db") #if no password put ""

df=pandas.read_sql("SELECT *FROM `TABLE 5`",conn)
print(df)



#plot histogram
fig,ax=plt.subplots()
ax.hist(df['ApplicantIncome'],color='blue',alpha=0.3,label='Appl.Income')
ax.hist(df['CoapplicantIncome'],color='red',alpha=0.3,label='C.Apl.Income')
ax.hist(df['LoanAmount'],color='green',alpha=0.3,label='LoanAmount')
ax.set_title('Distribution of Income')
ax.set_xlabel('Income(USD)')
ax.set_ylabel('Frequency')
ax.legend(loc='best')
plt.show()

#piechart
fig,ax=plt.subplots()
df.groupby('Education').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Education')
ax.set_ylabel('')
plt.show()

#piechart for loan status
fig,ax=plt.subplots()
df.groupby('Loan_Status').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Loan_Status')
ax.set_ylabel('')
plt.show()

