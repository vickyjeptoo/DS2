import pandas
import sklearn
df = pandas.read_csv("pima-data-orig.csv")

# feature selection types:

# 1. Univariate Selection
# 2. Recursive Feature Elimination(RFE)
# 3. Principle Component Analysis(PCA)
# 4. Feature Importance

# Why do we need feature selection

# 1. Improve accuracy
# 2. Reduces training time
# 3. Reduces overfiting...reduce redundant columns

array = df.values
x = array[:,0:8]
y = array[:,8]
# univariate selection
from sklearn.feature_selection import SelectKBest, chi2
model = SelectKBest(score_func=chi2, k=5) # get top 5
fitting = model.fit(x,y) # gets top 5 features
top5 = fitting.scores_
print(top5)
# RFE
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
model2 = DecisionTreeClassifier()
rfe = RFE(model2,6) # if you want the top five. RFE needs a model to work
fitting2 = rfe.fit(x,y)
print('Feature',fitting2.support_)

# feature importance
from sklearn.ensemble import ExtraTreesClassifier
model3 = ExtraTreesClassifier()
model3.fit(x,y)
print(model3.feature_importances_)

# principle component analysis
# **********************************






































































