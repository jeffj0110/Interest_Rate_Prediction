# Prototype Treasury Yield prediction using Keras and Tensorflow.
# J. Jones 
# April 2021
#
# Script creates a model, runs through training data and then does a validation with a separate set of data.
#
# A final prediction with forecasted dependent economic variables is generated for 3/31/2021.
#
import keras
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# load the dataset
dfs = []
Training_Data_Set = 'D:\FITSolutions_Prototypes\Interest_Rate_Predictor\IR_Training_Data.csv'
dfs.append(pd.read_csv(Training_Data_Set))

# need to convert date strings to number of days since Jan 1, 1970 for Keras
# insert column at beginning of list
frame = pd.concat(dfs)
frame.insert(0,'DaysSince01011970',0)

start_date = datetime(1970,1,1)
rowcount = 0
for EachDate in frame['Date_MMDDYYYY']:
    date_time_str = frame.iloc[rowcount,1]
    date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y')
    dayssince1970 = date_time_obj - start_date
    frame.iloc[rowcount,0] = dayssince1970.days
    rowcount += 1

#Remove the text column from the dataframe
frame = frame.drop('Date_MMDDYYYY',1)

# Convert dataframe to numpy array

IntRate_Dataset = np.array(frame)
X = IntRate_Dataset[:,0:7]
y = IntRate_Dataset[:,7:14]

scaler_X = MinMaxScaler()
scaler_X.fit(X)
xscale = scaler_X.transform(X)
#scaler_y = MinMaxScaler()
#scaler_y.fit(y)
#yscale = scaler_y.transform(y)

# I copied this function to create an R2 (R Squared) value for this regression model
# https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#define Keras model
model = Sequential()
model.add(Dense(75, input_dim=7, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(200))
model.add(Dense(7))
model.summary()

# compile the keras model
optimizer = keras.optimizers.Adam(lr=0.007)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', coeff_determination])

# fit the keras model on the dataset
model.fit(xscale, y, epochs=6000, batch_size=200)

Val_Data_Set = 'D:\FITSolutions_Prototypes\Interest_Rate_Predictor\IR_Validation_Data.csv'
dfs = pd.read_csv(Val_Data_Set)
# need to convert date strings to number of days since Jan 1, 1970 for Keras
# insert column at beginning of list
frame = dfs
frame.insert(0,'DaysSince01011970',0)

start_date = datetime(1970,1,1)
rowcount = 0
for EachDate in frame['Date_MMDDYYYY']:
    date_time_str = frame.iloc[rowcount,1]
    date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y')
    dayssince1970 = date_time_obj - start_date
    frame.iloc[rowcount,0] = dayssince1970.days
    rowcount += 1

#Remove the text column from the dataframe
frame = frame.drop('Date_MMDDYYYY',1)

IntRate_Dataset = np.array(frame)
X = IntRate_Dataset[:,0:7]
y = IntRate_Dataset[:,7:14]

scaler_X = MinMaxScaler()
scaler_X.fit(X)
xscale = scaler_X.transform(X)

results = model.predict(xscale)

Xnew = scaler_X.inverse_transform(xscale)

resulting_array = np.concatenate((Xnew, results), axis=1)

df = pd.DataFrame(data=resulting_array)
df.columns=["Days","Real GDP growth","Real disposable income growth","Unemployment rate","CPI inflation rate","Fed Disc Rate", "China Disc Rate",
"3-month Treasury rate", "6-month Treasury rate","1-year Treasury yield","2-year Treasury yield","3-year Treasury yield","5-year Treasury yield","10-year Treasury yield"]

df.insert(0,'Quarter_End'," ")
start_date = datetime(1970,1,1)
rowcount = 0
for DayCounter in df['Days']:
    qend = start_date + timedelta(days=DayCounter)
    df.iloc[rowcount,0] = qend.strftime("%Y/%m/%d")
    rowcount += 1

df.sort_values("Quarter_End", ascending=True, inplace=True )
df.drop("Days",1)
df.to_csv('D:\FITSolutions_Prototypes\Interest_Rate_Predictor\IR_Validation_Results.csv', index=False)
