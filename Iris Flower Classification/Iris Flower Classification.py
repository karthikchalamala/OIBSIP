import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('iris.csv')
print(df)

#EDA
df.describe()

df["Species"].unique()
df.groupby("Species").size()

corr = df.corr(numeric_only = True)
plt.subplots(figsize=(10,6))
sns.heatmap(corr,annot=True)

#target variable separaton
x = df.iloc[:,1:5].values
y = df.iloc[:,5].values

#spliting into test and training datas from iris dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.figure()
sns.pairplot(df.drop("Id",axis=1),hue="Species",height=3,markers=['o','s','D'])
plt.show()

#KNeaerwst Neighbors Classifier Implementation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

#accuracy result
accuracy = accuracy_score(y_test,y_pred)*100
print(f"Accuracy:{round(accuracy,2)}")
