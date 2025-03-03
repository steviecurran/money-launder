#!/opt/miniconda3/envs/py3-TF2.0/bin/python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import collections
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)
import pickle
import os 
import sys

df = pd.read_csv("all_data.csv"); #print(df) 

df1 = df.copy()
df2 =df[['ID','Target_ml','e_23456','e_64645','e_43568','CountPaymentMethod','Is_Retail','e_11234','e_34454']] # set_kNN

############# DEEP LEARNING ###############
def DL(n_loops,data,label):
    big_array = []

    for i in range(0,n_loops):
        print("On loop %d of %d" %(i+1, n_loops))
        df = data.sample(frac=1)
        df['Target_ml'] = df['Target_ml'].astype(int)
        target = 'Target_ml'

        sus = df[df[target] == 1]; n1 = len(sus)
        not_sus = df[df[target] == 0]; n0 = len(not_sus)

        if n1 > n0:
            class0 = df.loc[df[target] == 0]
            class1 = df.loc[df[target] == 1][:n0]
        else:
            class0 = df.loc[df[target] == 0][:n1]
            class1 = df.loc[df[target] == 1]
        #print("Sample sizes now", len(class0),len(class1))
        df = pd.concat([class0, class1])

        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(drop=True, inplace=True)
        del df['ID']

        # DROP TARGET
        X = df.drop([target], axis = 1); y = df[target]
        ## SPLIT DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        ## SCALE FEATURES - TRAIN AND TEST SEPERATELY
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        X_train = np.array(X_train);X_test= np.array(X_test)
        y_train= np.array(y_train);y_test= np.array(y_test)# need np array for TensorFlow

        HLS = 50

        def build_model():
          model = keras.Sequential([
            layers.Dense(HLS, activation='relu', input_dim=len(df.columns)-1), # sigmoid - for binary
            layers.Dense(HLS, activation='relu'),
              # layers.Dense(HLS, activation='sigmoid'),
              # layers.Dense(HLS, activation='relu'),
              # layers.Dense(HLS, activation='tanh'),
              # layers.Dense(HLS, activation='softmax'),
              # layers.Dense(HLS, activation='tanh'),
              # layers.Dense(HLS, activation='tanh'),
            layers.Dense(1)
          ])

          model.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
          return model

        model = build_model(); print(model.summary())

        batch_size = 100  # Start Training Our Classifier 
        epochs = 1000 
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=20) 

        history = model.fit(X_train, y_train, batch_size = batch_size,epochs = epochs,verbose = 0, validation_data = (X_test, y_test), callbacks =[early_stopping])

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        #print(hist)

        score = model.evaluate(X_test, y_test, verbose=0)
        #print('Test loss:', score[0]); print('Test accuracy:', score[1])
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5)
        # Classification metrics can't handle a mix of continuous and binary targets
        
        TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
        acc = round((TP + TN)/(TP + FP + TN + FN),4)
        print('Validation accuracy is %1.2f percent' %(100*(TP + TN)/(TP + FP + TN + FN)))
        
        #big_array.append(score[1])
        big_array.append(acc)
    data = pd.DataFrame(big_array, columns=[label]);
    
    out = 'temp_%s.csv'  %(label)
    data.to_csv(out, index = False)
    return out
        
n_loops = int(input("Number of loops [e.g. 100]? "))
    
file1=DL(n_loops,df1,'All')
file2=DL(n_loops,df2,'set_kNN')

dfa = pd.read_csv(file1)
dfb = pd.read_csv(file2)

dfa['set_kNN'] = dfb['set_kNN']
print(dfa)
os.system("rm temp*.csv")

#date_text = 'date "+%F-%T"'
import subprocess
date = subprocess.check_output('date "+%F-%T"', shell=True).strip()
# UNIQUE NAME IN CASE WANT TO COMBINE WITH OTHER RUNS
print(date)

out = 'DL_loops=%d-%s.csv'  %(n_loops,date)
dfa.to_csv(out, index = False)
print("Written to %s" %(out))

