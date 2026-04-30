# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: HEMALISHA T
## RegisterNumber: 212225040123

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score as rs
from sklearn.model_selection import train_test_split

df=pd.read_csv('students_mark.csv')
df.head(11)

hours=np.array(df['Hours studied']).reshape(-1,1)
marks=np.array(df['Marks'])
x_train, x_test, y_train, y_test=train_test_split(hours, marks, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

rmse=np.sqrt(mse(y_test, y_pred))
r2=rs(y_test, y_pred)

print("Model Evalution:")
print("Slope (m): ", model.coef_[0])
print("Intercept (b): ", model.intercept_)
print(f"Root_mean_squared error (rmse): {rmse:.4f}")
print(f"R2_scored error (r2): {r2:.4f}\n")

plt.plot(hours, model.predict(hours), color='r', label='Predicted data')
plt.scatter(hours, marks, color='g', label='Actual data')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()
```


## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
