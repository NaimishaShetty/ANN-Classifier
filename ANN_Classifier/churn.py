import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 

lEncoder1 = LabelEncoder() 
x[:,1] = lEncoder1.fit_transform(x[:,1])

lEncoder2 = LabelEncoder() 
x[:,2] = lEncoder1.fit_transform(x[:,2])

ohEncoder = OneHotEncoder(categorical_features=[1])
x = ohEncoder.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.preprocessing import StandardScaler

sScalar = StandardScaler()
x = sScalar.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

#******************************Data preprocessing ends**********************************************
import keras
from keras.models import Sequential
from keras.layers import Dense

annClassifier = Sequential()

#input layer
annClassifier.add(Dense(input_dim = 11, activation='relu', units=6))

#1st hidden layer
annClassifier.add(Dense(activation='relu',units=6))

#2nd hidden layer
annClassifier.add(Dense(activation='relu',units=6))

#output layer
annClassifier.add(Dense(units=1,activation='sigmoid'))

annClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = annClassifier.fit(x_train,y_train,batch_size=10,epochs=100)


