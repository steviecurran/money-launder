#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import collections
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df1 = pd.read_csv("all_data.csv"); #
target = 'Target_ml'

############# MACHINE LEARNING ###############
def ML(n_loops):
    big_array = []

    for i in range(0,n_loops):
        print("on loop %d of %d" %(i+1, n_loops))
        df = df1.sample(frac=1) # randomise
        df[target] = df[target].astype(int)

        sus = df[df[target] == 1]; n1 = len(sus)
        not_sus = df[df[target] == 0]; n0 = len(not_sus)

        if n1 > n0:
            class0 = df.loc[df[target] == 0]
            class1 = df.loc[df[target] == 1][:n0]
        else:
            class0 = df.loc[df[target] == 0][:n1]
            class1 = df.loc[df[target] == 1]
        df = pd.concat([class0, class1])

        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(drop=True, inplace=True)

        del df['ID']

        test_frac = 0.2; cv = 10
        ## DROP TARGET
        X = df.drop([target], axis = 1); y = df[target]
        ## SPLIT DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)
        ## SCALE FEATURES - TRAIN AND TEST SEPERATELY
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        res=[]
        classifiers = {
            "LR": LogisticRegression(C=0.08858667, solver='newton-cg'),
            "KNN": KNeighborsClassifier(n_neighbors=6),
            "SVC": SVC(C=1, gamma=0.1),
            "DTC": DecisionTreeClassifier(max_depth = 5),
        }

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
            predictions = classifier.predict(X_test); # print('Testing\n', confusion_matrix(predictions, y_test))
            TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
            acc = 100*(TP + TN)/(TP + FP + TN + FN)
            
            res.append(round(acc,3))

        res2 = np.reshape(res,(-1,4)); 
        big_array.append(res2);

    res3 = np.reshape(big_array,(-1,4));

    data = pd.DataFrame(res3, columns=['LR','kNN','SVC','DTC']);
    out = 'ML_loops=%d.csv'  %(n_loops)
    data.to_csv(out, index = False)
    return out

## HISTOGRAMS
import matplotlib.pyplot as plt
import os 
import sys

def plots(data):
    
    big = 14; small = 12 # text sizes
    data = pd.read_csv(out) 
    
    plt.rcParams.update({'font.size': big})
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    plt.ylabel('Number', size=big); plt.xlabel('Validation accuracy [%]', size=big)

    min_val = 58.8; max_val = 87.9;
    desired_bin_size = 1
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    
    para = data['SVC'] 
    ax.hist(para, bins=bins, color="w", edgecolor='b', linewidth=3, alpha=0.8);
    #ax.set_ylim([0, 1.5*y2]) # EXTEND y-axis TO FIT TEXT
    y1, y2 = ax.get_ylim()
    x_pos = 58; 
    y_pos = 0.9*y2; step = (y2-y1)/16
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "SVC: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-3*step,text, fontsize = small, c = 'b')

    para = data['LR']
    ax.hist(para, bins=bins, color="w", edgecolor='r', linewidth=3);  
    mean = np.mean(para); std = np.std(para); 
    text = "LR:   \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos,text, fontsize = small, c = 'r')

    para = data['kNN']
    ax.hist(para, bins=bins, color="w", edgecolor='g', linewidth=3, alpha=0.75);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "kNN: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-step,text, fontsize = small, c = 'g')

    para = data['DTC'] 
    ax.hist(para, bins=bins, color="w", edgecolor='orange', linewidth=3, alpha=0.75);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "DTC: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-2*step,text, fontsize = small, c = 'orange')

    plt.tight_layout()
    plot = "%s" %(out); png = "%s.png" %(plot)
    eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps);  print("Plot written to", png);
    plt.show()

ans = str(input("Run machine learning [could take a while], or straight to plotting histograms? [m/other]: "))

if ans == "m":
    n_loops = int(input("Number of loops [e.g. 1000]? "))
    ML(n_loops)
    out = ML(n_loops)

else:
    os.system("ls *loops*csv")
    out = input("Input file?: ")    

plots(out)
