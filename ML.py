#!/usr/local/bin/python3
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
    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
    "SVC": SVC(C=1, gamma=0.1),
    "DTR": DecisionTreeClassifier(max_depth = max_depth),
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
    print("For a %1.1f test fraction (%d train & %d test) %s training score = %1.3f percent" %(test_frac, len(X_train), len(X_test), key, training_score.mean()*100))

    predictions = classifier.predict(X_test); print('Testing\n', confusion_matrix(predictions, y_test))
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    #print('True Positive(TP)  = ', TP); print('False Positive(FP) = ', FP); print('True Negative(TN)  = ', TN); print('False Negative(FN) = ', FN)

    print('Validation accuracy of %s is %1.2f percent' %(key, 100*(TP + TN)/(TP + FP + TN + FN)))


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

