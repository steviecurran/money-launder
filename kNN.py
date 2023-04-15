#!/usr/bin/python3
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import matplotlib
import os 
import sys

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("all_data.csv"); print(df) 

############# MACHINE LEARNING ###############
df = df.sample(frac=1) 
df['Target_ml'] = df['Target_ml'].astype(int)

sus = df[df['Target_ml'] == 1]; n1 = len(sus)
not_sus = df[df['Target_ml'] == 0]; n0 = len(not_sus)

if n1 > n0:
    class0 = df.loc[df['Target_ml'] == 0]
    class1 = df.loc[df['Target_ml'] == 1][:n0]
else:
    class0 = df.loc[df['Target_ml'] == 0][:n1]
    class1 = df.loc[df['Target_ml'] == 1]
print("Sample sizes now", len(class0),len(class1))
df = pd.concat([class0, class1])

df = df.reindex(np.random.permutation(df.index))
df.reset_index(drop=True, inplace=True)
del df['ID']

cv=10
test_frac = 0.2

X = df.drop(['Target_ml'], axis = 1); y = df['Target_ml']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(6,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)

k_range = list(range(1, 51))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    k_scores.append(scores.mean())
    print(k, scores.mean())
#print(k_scores)
plt.plot(k_range, k_scores,  c = 'r')
plt.xlabel('Number of nearest neighbours, $k$')
plt.ylabel('kNN cross-validation acccuracy')
plt.tight_layout()
plot = "kNN-results"; png = "%s.png" % (plot); 
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png); os.system(eps); plt.show(); print("Plot written to", png);
plt.show()

