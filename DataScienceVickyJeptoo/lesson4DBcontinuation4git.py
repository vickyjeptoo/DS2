import pandas
import matplotlib.pyplot as plt
import pymysql #to connect to mysql DB


conn = pymysql.connect("localhost","root","1234D!@#$","ds_vicky_db") #if no password put ""

#df=pandas.read_sql("SELECT *FROM `TABLE 5` WHERE ApplicantIncome>9000",conn)
#df=pandas.read_sql("SELECT *FROM `TABLE 5` WHERE Credit_History=1",conn)
#df=pandas.read_sql("SELECT *FROM `TABLE 5` WHERE LoanAmount>200 and Credit_History=1",conn)
df=pandas.read_sql("SELECT *FROM `TABLE 5` WHERE Education = 'graduate'",conn)
print(df)


#plot to improve matplotlib
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('Solarize_Light2')
#save your plots
#right click project
#new directory;'static'

#plot histogram
fig,ax=plt.subplots()
ax.hist(df['ApplicantIncome'],color='blue',alpha=0.3,label='Appl.Income')
ax.hist(df['CoapplicantIncome'],color='red',alpha=0.3,label='C.Apl.Income')
ax.hist(df['LoanAmount'],color='green',alpha=0.3,label='LoanAmount')
ax.set_title('Distribution of Income')
ax.set_xlabel('Income(USD)')
ax.set_ylabel('Frequency')
ax.legend(loc='best')
plt.savefig('static/hist.png')

#piechart
fig,ax=plt.subplots()
df.groupby('Education').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Education')
ax.set_ylabel('')
plt.savefig('static/piechart.png')
#plt.show()


#piechart for loan status
fig,ax=plt.subplots()
df.groupby('Loan_Status').size().plot(kind='pie',autopct='%1.1f%%')
ax.set_title('Distribution of Loan_Status')
ax.set_ylabel('')
plt.savefig('static/pie_loan.pdf')
plt.show()

#stacked bar chart for education,loan_status by ApplicantIncome
from matplotlib import rcParams#to make the plot fit well,best to do at the top of the code
rcParams.update({'figure.autolayout':True})
fig,ax=plt.subplots()
df.groupby(['Education','Loan_Status'])['ApplicantIncome'].mean().unstack().plot(kind='bar',stacked=True)
plt.title('Applicant mean by education category')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount(USD)')
plt.show()

#scatter plot:ApplicantIncome  ...CoApplicantIncome
fig,ax=plt.subplots()
ax.scatter(df['ApplicantIncome'],df['CoapplicantIncome'],color='blue',s=30)
ax.set_title('Relationship btn ApplicantIncome vs CoApplicantIncome')
ax.set_xlabel('ApplicantIncome')
ax.set_ylabel('CoApplicantIncome')
plt.show()

#Heat maps-shows correlation for multiple variables
import seaborn as sns
fig,ax=plt.subplots()
df_x=df[['ApplicantIncome','CoapplicantIncome']] #narrow down the df to df_x
sns.heatmap(df_x.corr(),cmap='Blues',annot=True)  #Check other params that can be passed at cmap
ax.set_title('Relationship between CoapplicantIncome and ApplicationIncome')
plt.show()

#csv datasets ,kaggle datasets