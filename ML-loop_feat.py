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
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import collections
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df1 = pd.read_csv("all_data.csv"); #

############# MACHINE LEARNING ###############
def ML(n_loops):
    big_array = []

    for i in range(0,n_loops):
        df = df1.sample(frac=1) # randomise
        df['Target_ml'] = df['Target_ml'].astype(int)

        sus = df[df['Target_ml'] == 1]; n1 = len(sus)
        not_sus = df[df['Target_ml'] == 0]; n0 = len(not_sus)

        if n1 > n0:
            class0 = df.loc[df['Target_ml'] == 0]
            class1 = df.loc[df['Target_ml'] == 1][:n0]
        else:
            class0 = df.loc[df['Target_ml'] == 0][:n1]
            class1 = df.loc[df['Target_ml'] == 1]
        df = pd.concat([class0, class1])

        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(drop=True, inplace=True)

        del df['ID']
        set_orig = ['Target_ml','e_23456','e_64645','e_43568','e_11234','e_34454','CountDeposit','DifferentMethodWithdrawals']
        set_LR = ['Target_ml','CountPaymentMethod','e_64645','e_43568','IP_Counts','e_23456','e_34454','Is_Retail']
        set_SVC = ['Target_ml','e_23456','e_64645','e_43568','IP_Counts','Is_Retail','e_34454','CountPaymentMethod']
        set_kNN =['Target_ml','e_23456','e_64645','e_43568','CountPaymentMethod','Is_Retail','e_11234','e_34454']
        set_DTC =['Target_ml','CountWithdrawal','TotalDeposits','DifferentMethodWithdrawals','CountDeposit','e_23456','e_64645','IP_Counts']

        #choice = set_orig; ext = "orig"
        #choice = set_LR; ext = "LR"
        #choice = set_SVC; ext = "SVC"
        #choice = set_kNN; ext = "kNN"
        choice = set_DTC; ext = "DTC"
        
        df = df[choice] 
        
        test_frac = 0.2; cv = 10

        X = df.drop(['Target_ml'], axis = 1); y = df['Target_ml']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        res=[]
        classifiers = {
            "LR": LogisticRegression(C=0.004832930238571752, solver='newton-cg'),
            "KNN": KNeighborsClassifier(n_neighbors=17),
            "SVC": SVC(C=1, gamma=0.1),
            "DTC": DecisionTreeClassifier(max_depth = 6),
        }

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
            res.append(round(training_score.mean(),3))

        res2 = np.reshape(res,(-1,4)); #print(res2)
        big_array.append(res2);

    res3 = np.reshape(big_array,(-1,4));#print(res3)

    data = pd.DataFrame(res3, columns=['LR','kNN','SVC','DTC']);
    out = 'ML_loops=%d-feat_%s.csv'  %(n_loops,ext)
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

    plt.ylabel('Number', size=big); plt.xlabel('Mean scores [%]', size=big)

    para = 100*data['LR']
    ax.hist(para, bins=4*int(para.max() - para.min()), color="w", edgecolor='r', linewidth=3);  
    x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
    ax.set_ylim([0, 1.5*y2]) # EXTEND y-axis TO FIT TEXT
    y1, y2 = ax.get_ylim()
    x_pos = x1; y_pos = 0.9*y2; step = (y2-y1)/16
   
    mean = np.mean(para); std = np.std(para); 
    text = "LR:   \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos,text, fontsize = small, c = 'r')

    para = 100*data['kNN']
    ax.hist(para, bins=12, color="w", edgecolor='g', linewidth=3);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "kNN: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-step,text, fontsize = small, c = 'g')

    para = 100*data['DTC'] 
    ax.hist(para, bins=14, color="w", edgecolor='orange', linewidth=3, alpha=0.9);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "DTC: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-2*step,text, fontsize = small, c = 'orange')

    para = 100*data['SVC'] 
    ax.hist(para, bins=10, color="w", edgecolor='b', linewidth=3, alpha=0.8);
    mean = np.mean(para); std = np.std(para); #print(mean,std)
    text = "SVC: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
    plt.text(x_pos,y_pos-3*step,text, fontsize = small, c = 'b')

    plt.tight_layout()
    plot = "%s" %(out); png = "%s.png" %(plot)
    eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps); plt.show(); print("Plot written to", png);
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
