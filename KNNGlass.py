
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

glass = pd.read_csv("F:\\ExcelR\\excelRASS\\ass8 KNN\\glass.csv")
glass.dtypes

feature_cols=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
x=glass[feature_cols]
y=glass.Type
print(x.shape)
print(y.shape)
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
xtrain.shape
ytrain.shape
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain,ytrain)