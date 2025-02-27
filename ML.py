#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
import pickle

df = pd.read_csv("all_data.csv"); print(df) 

############# MACHINE LEARNING ###############
df = df.sample(frac=1)
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
print("Sample sizes now", len(class0),len(class1))
df = pd.concat([class0, class1])

df = df.reindex(np.random.permutation(df.index))
df.reset_index(drop=True, inplace=True)
del df['ID']

#df.to_csv('data.csv', index = False) # FOR OPTIMISATION, ETC

print(df)

test_frac = 0.2
max_iter = 1000
n_neighbors = 6
max_depth = 5
cv = 10

## DROP TARGET
X = df.drop([target], axis = 1); y = df[target]
## SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)
## SCALE FEATURES - TRAIN AND TEST SEPERATELY
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifiers = {
    "LR": LogisticRegression(C=0.08858667, solver='newton-cg'),
    #"Knn": KNeighborsClassifier(n_neighbors=n_neighbors),
    #"DTC": DecisionTreeClassifier(max_depth = max_depth),
    #"SVC": SVC(C=1, gamma=0.1,probability=True) # True for CalibrationDisplay
    }


for key, classifier in classifiers.items():
    clf = classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
    print("For a %1.1f test fraction (%d train & %d test) %s training score = %1.3f percent"
          %(test_frac, len(X_train), len(X_test), key, training_score.mean()*100))

    predictions = classifier.predict(X_test); print('Testing\n', confusion_matrix(predictions, y_test))
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    #print('True Positive(TP)  = ', TP); print('False Positive(FP) = ', FP); print('True Negative(TN)  = ', TN); print('False Negative(FN) = ', FN)
    
    print('Validation accuracy of %s is %1.2f percent' %(key, 100*(TP + TN)/(TP + FP + TN + FN)))

    #### CALIBRATION PLOTS ######
    from sklearn.calibration import CalibrationDisplay
    nbins= 20
    strat='quantile'
    #strat = 'uniform'
    font = 12

    plt.rcParams.update({'font.size': font})
    fig, ax1 = plt.subplots(figsize = (6, 4))
    #ax2 = fig.add_axes([0.55, 0.2, 0.35, 0.3]) # INSET left, bottom, width, height
    ax2 = fig.add_axes([0.15, 0.6, 0.35, 0.3])
    ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()
    plt.setp(ax1.spines.values(),linewidth=2)
    plt.setp(ax2.spines.values(),linewidth=2)
    ax1.tick_params(direction='in', pad = 5,length=6, width=1.5, which='major')
    ax1.tick_params(direction='in', pad = 5,length=3, width=1.5, which='minor')
    ax2.tick_params(axis='both', which='major', labelsize=0.8*font)
    
    disp = CalibrationDisplay.from_estimator(clf,X_test,y_test,n_bins=nbins,strategy=strat,
                                             pos_label=None, color='r', name=key, ref_line=True,
                                             ax=ax1,zorder = 2, label = r"%s, $n = %d$" %(key,len(X_test)))

    ax2.hist(disp.y_prob,range=(0,1),bins=nbins,label=key, color="silver", edgecolor='k',lw=2,zorder=1,alpha=0.7)
    #ax2.set_xlabel(r'Mean predicted probability', size=0.8*font);
    ax2.set_ylabel(r'Number',size=0.8*font)
    
    #print(disp.y_prob); print(disp.prob_pred); print(disp.prob_true)
    diff2  = (disp.prob_pred - disp.prob_true)**2; 
    diff2_sum = diff2.sum(); print(diff2_sum)
    ax1.plot([], [], ' ', label=r"$\chi^2/n_{\rm bins} = %1.3f$" %(diff2_sum/(nbins)))# Extra label on the legend
    ax1.legend(fontsize = font,loc='lower right',labelcolor='k')

    plt.tight_layout()
    outfile = 'cal_display_%s-nbins=%d.eps' %(key,nbins)
    #plt.savefig(outfile, format = 'eps'); print('Written to %s' %(outfile))
    plt.show()
    
    #FEATURE IMPORTANCE #########
    # from sklearn.inspection import permutation_importance
    # results = permutation_importance(classifier, X_train, y_train, scoring='accuracy')
    # importance = results.importances_mean
    # features = df.drop(target,axis = 1) 
    # features = features.columns.tolist()
    # df1 = pd.DataFrame(features, columns=['Feature']); 
    # df2 = pd.DataFrame(importance, columns=['Importance'])
    # df1['Importance'] = df2.Importance  # adding to df1
    # print(df1.sort_values(by=['Importance'], ascending=False)) 

    #pickle.dump(classifier, open('%s.pickle' %(key), 'wb'))

