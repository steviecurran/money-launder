#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os 
import sys
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("all_data.csv");
#df = df[['Target_ml','e_23456','e_64645','e_43568','CountPaymentMethod','Is_Retail','e_11234','e_34454']]
df = df.sample(frac=1) 
df['Target_ml'] = df['Target_ml'].astype(int)
target = 'Target_ml'

steps = np.linspace(100,1700,17)
n_loops = 100

res3=[]
for i in range (0,17): 
    n = int(steps[i]);# print(n)
    print("On step %d of 17" %(i))
    for i in range(0,n_loops):
        print("On loop %d of %d" %(i+1, n_loops))
        class0 = df.loc[df[target] == 0][:n]
        class1 = df.loc[df[target] == 1][:n]

        #print("Sample sizes now", len(class0),len(class1))
        df1 = pd.concat([class0, class1])
        df1 = df1.reindex(np.random.permutation(df1.index))
        df1.reset_index(drop=True, inplace=True)
        #print(n,len(df1)); #print(df); print(df.describe())

        test_frac = 0.2; cv = 10

        X = df1.drop([target], axis = 1); y = df1[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        res=[n]; res4=[]
        classifiers = {
            "LR": LogisticRegression(C=0.08858667, solver='newton-cg'),
            "KNN": KNeighborsClassifier(n_neighbors=6),
            "SVC": SVC(C=1, gamma=0.1),
            "DTC": DecisionTreeClassifier(max_depth = 5),
        }

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
            #res.append(round(training_score.mean(),3))
            predictions = classifier.predict(X_test); 
            TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
            acc = (TP + TN)/(TP + FP + TN + FN)
            res.append(round(acc,4))
            
        res3.append(res)
#print(res3)
res2 = np.reshape(res3,(-1,5));print(res2)
data = pd.DataFrame(res2, columns=['n','LR','kNN','SVC','DTC']);print(data)

out = 'ML_sampled_loops=%d_all-validation.csv'  %(n_loops)
data.to_csv(out, index = False)
print("Writen to %s" %(out))
