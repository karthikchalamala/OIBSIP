import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('spam.csv',encoding='ISO-8859-1')
print(df)

print(df.head())

df['v2'].value_counts()

df['v1'].value_counts()

#separating x and y differently
x = df.v2.values
y = df.v1.values

#spilting the datasets into train test datas
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25)

#Data preprocessing step
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(xtrain)

#Data preprocessing step
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(xtrain)

#defining the model using M; Algorithm(navie bayes)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train,ytrain)

x_test = cv.transform(xtest)
x_test.toarray()

print(model.score(x_test,ytest))

email = ['get an iphone 15 for free','use this product to be fair within 7 days,otherwise money return','give your account number of bank,to get 1000 for the prediction']
cv_email = cv.transform(email)
print(model.predict(cv_email))
