#!/usr/bin/python3
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import matplotlib

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv"); #print(df)

cv=10
test_frac = 0.2

X = df.drop(['Target_ml'], axis = 1); y = df['Target_ml']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier =SVC()
classifier.fit(X_train, y_train)
training_score = cross_val_score(classifier, X_train, y_train, cv=cv) 
print("For a %1.1f test fraction (%d train & %d test) training score = %1.3f" %(test_frac, len(X_train), len(X_test),training_score.mean()*100))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

print(grid.best_estimator_)
