# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ml-lab-1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
print("coefficient",lr.coef_)
print("intercept",lr.intercept_)

```

## Output:
![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/cf4ca874-7403-4b84-9711-35a2db6747f5)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/c917d841-636f-41a3-8891-4fb4f10298bd)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/913838df-f806-45b0-8926-a2cb0375d33e)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/14142ed1-b307-410c-93a4-c388c0776632)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/ac011e49-1484-4a01-95d3-9cccb0c99942)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/162780573/4c6978ee-ebdc-41cc-93cf-32069f8f5e51)

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
