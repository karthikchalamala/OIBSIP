import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")
print(df)

print(df.head())

print(df.describe())

print(df.shape)

print(df.isnull().sum())

print(df.isnull().sum())

sns.heatmap(df.corr(),annot=True)
sns.lmplot(data=df,x='Newspaper',y='Sales')
sns.lmplot(data=df,x='TV',y='Sales')
sns.lmplot(data=df,x='Radio',y='Sales')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = df[['TV','Radio','Newspaper']]
y = df['Sales']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.65,random_state=0)
model = LinearRegression()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(model.intercept_)
print(model.coef_)
act_predict = pd.DataFrame({
    'Actual':y_test.values.flatten(),
    'predict':y_predict.flatten()
})
print(act_predict.head(20))
sns.lmplot(data=act_predict,x='Actual',y='predict')
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("Mean_absolute_error",mean_absolute_error(y_test,y_predict))
print("Mean_squared_error:",mean_squared_error(y_test,y_predict))
print("Square_Mean_absolute_error",np.sqrt(mean_absolute_error(y_test,y_predict)))
print("r2_score",r2_score(y_test,y_predict))
