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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_in = pd.read_csv("all_data.csv"); #print(df) 

def sample(data):
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
    del df['ID']  # input_dim=len(df_in.columns)-2)
    # DROP TARGET
    X = df.drop([target], axis = 1); y = df[target]
    ## SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    ## SCALE FEATURES - TRAIN AND TEST SEPERATELY
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    X_train = np.array(X_train);X_test= np.array(X_test)
    y_train= np.array(y_train);y_test= np.array(y_test)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

from random import randrange

def rand_HLS():
    i = randrange(0,len(HLSs))
    return HLSs[i]

def rand_act():
    j = randrange(0,len(acts))
    return acts[j]


#### GRID SEARCH - RU =N EACH MODEL SEVERAL TIMES FOR STABILITY

HLSs = [10,20,50,100,200,500]
acts = ['relu','sigmoid','tanh','softmax']# CLASSIFICATION FUNCTIONS 
max_layers = 11 # FOR 10 HIDDEN
columns = []
for k in range(0,max_layers):
      column = "HL_%d" %(k)
      columns.append(column)

# CREATE A MODEL AND RUN FOR, SAY
df2 =  pd.DataFrame()
N = 10
no = randrange(1,max_layers); #print(no)
HLS = rand_HLS(); #print(no,HLS)

for i in range(0,no): 
    first_layer=rand_act()

#print(acts[0],first_layer)
    
model_layers = ""
layers_arr= []
for i in range(0,no): # WAS 1, no
    act=rand_act()
    model_layers = model_layers + "layers.Dense(%d, activation='%s')," %(HLS,act)
    layers_arr.append(act)

# print("---------\n")
# print(model_layers)
# print("\n---------\n")

diff = max_layers - len(layers_arr); 
for a in range (0,diff):   
    layers_arr.append('')

#### DL MODEL ######
for i in range(0,N):
    X_train,X_test,y_train,y_test = sample(df_in);#print(X_test)
    blah = "model = keras.Sequential([\n layers.Dense(%d, activation='%s', input_dim=len(df_in.columns)-2),%slayers.Dense(1)])\nmodel.compile(optimizer= 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])" %(HLS,first_layer,model_layers);
    #print(blah)
    exec(blah); #print(model.summary())

    batch_size = 100  # data points processed before the model's internal parameters (weights) are updated
    # smaller takes longer. Leave fixed just now
    epochs = 1000 
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20)
    history = model.fit(X_train, y_train, batch_size = batch_size,epochs = epochs,verbose = 0, validation_data = (X_test, y_test), callbacks =[early_stopping])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    score = model.evaluate(X_test, y_test, verbose=0)
    df1 = pd.DataFrame([layers_arr], columns=columns)
    df1['HLS'] = HLS;
    df1['Score']= score[1]
    df2 = df2.append(df1);
print(df2)

df3 = df2.iloc[[1]]
df3['Mean'] = np.mean(df2['Score']); df3['SD'] = np.std(df2['Score'],ddof=1)
df3['N'] = N
print(df3)

outfile = "N=%d.csv" %(N)
#outfile = 'DL-grid_%d-max_layers_N=%d.csv' %(max_layers,N)
#df3.to_csv(outfile, index=False, header=False); # header=False  FOR ./DL-grid.csh SCRIPT
#print('Written to %s' %(outfile))

