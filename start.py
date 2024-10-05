import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder #for the putting the value in the classifical form
from sklearn.preprocessing import StandardScaler # for creating the homogeneous data for the columns
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from Knn import Knn

#import the dataset
df=pd.read_csv('Social_Network_Ads.csv')

#convert the gender column in the numerical form
df=df.iloc[:,1:] # select the required columns
encoder=LabelEncoder()
df['Gender']=encoder.fit_transform(df['Gender'])

#convert the whole columns in the same numerical form
scalar=StandardScaler()
X=df.iloc[:,0:3].values
X=scalar.fit_transform(X)
y=df.iloc[:,-1].values

#divide the data into the train and test form
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#creating the model for training
kd=KNeighborsClassifier(n_neighbors=5)
kd.fit(X_train,y_train)

#Do the prediction
y_pred=kd.predict(X_test)

#find the accuracy of the model
print(accuracy_score(y_test,y_pred))

# KNN from scratch
apnaKnn = Knn(k=5)

apnaKnn.fit(X_train,y_train)
y_pred1 = apnaKnn.predict(X_test)
print(accuracy_score(y_test,y_pred1))

      