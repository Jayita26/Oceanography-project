#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score 
#%%
df = pd.read_csv('C:/Users/jayit/Downloads/archive (4)/bottle.csv')
df
df2 = df[['Depthm','Salnty', 'T_degC']]
df3 = df2[df2['Depthm'] ==0]
df4 = df3[['Salnty', 'T_degC']]
df4
x=df3['Salnty']
y=df3['T_degC']
x
y
X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.3, random_state=42)
X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape
lr=linear_model.LinearRegression()
lr
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_valid)
y_pred.shape
print(lr.score(X_train, Y_train))
print(mean_squared_error(Y_valid,y_pred),mean_absolute_error(Y_valid,y_pred),r2_score(Y_valid,y_pred))  
#%%  
plt.scatter(X_train, Y_train, color="darkgreen")
plt.scatter(X_valid, Y_valid, color="pink")
plt.plot(X_valid, y_pred, color='black')
plt.xlabel('Temperature')
plt.ylabel('Salinity')
plt.title('Linear Regression Model')
plt.show()
#%%
plt.figure(figsize=(15, 15))
sb.heatmap(df4.corr(), annot=True, cbar=False)
plt.show()

#%%
import numpy as np
import pandas as pd
import tensorflow as tf
dataset= pd.read_csv("C:/Users/jayit/OneDrive/Documents/Project/Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,text_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.tranform(x_test)
