#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 
import sys
import scipy

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv('ML_sampled_loops=100_all-validation.csv'); #print(df)
df['n'] = df['n'].astype(int)
#print(df.columns.values)
cols = ['LR','kNN','SVC','DTC']

big = 14; small = 12 # text sizes
arr = []

for i in range (1,18):
    data = df[df['n'] == 100*i]
    for (j, col) in enumerate(cols):
        #print(100*i,col,100*np.mean(data[col]), 100*np.std(data[col]))
        arr.append(100*i)
        arr.append(col)
        arr.append(round(100*np.mean(data[col]),5))
        arr.append(round(100*np.std(data[col]),5))
        
    #print(arr)
    averages = np.reshape(arr,(-1,4)); #print(averages)
df = pd.DataFrame(averages, columns=['n','ML','mean','sd']);print(df)    
df['n'] = df['n'].astype(int)
df['ML'] = df['ML'].astype(str)
df['mean'] = df['mean'].astype(float)
df['sd'] = df['sd'].astype(float)

LR = df[df['ML'] == 'LR']
kNN = df[df['ML'] == 'kNN']
SVC = df[df['ML'] == 'SVC']
DTC = df[df['ML'] == 'DTC']

#print(kNN); print(kNN.dtypes)

plt.rcParams.update({'font.size': big})
plt.figure(figsize=(10,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)

x = SVC['n']; y = SVC['mean']/100; e = SVC['sd']/100; 
ax.errorbar(x, y, yerr=e, xerr=None, fmt='s', c = 'b', capsize=6)
ax.plot(x,y,c='b')

x = kNN['n']; y = kNN['mean']/100; e = kNN['sd']/100; 
ax.errorbar(x, y, yerr=e, xerr=None, fmt='s', c = 'g', capsize=6)
ax.plot(x,y,c='g')

x = DTC['n']; y = DTC['mean']/100; e = DTC['sd']/100; 
ax.errorbar(x, y, yerr=e, xerr=None, fmt='s', c = 'orange', capsize=6)
ax.plot(x,y,c='orange')

x = LR['n']; y = LR['mean']/100; e = LR['sd']/100; 
ax.errorbar(x, y, yerr=e, xerr=None, fmt='s', c = 'r', capsize=6)
ax.plot(x,y,c='r')

plt.text(1650,73.5,"SVG", fontsize = small, c = 'b')
plt.text(1650,72,"kNN", fontsize = small, c = 'g')
plt.text(1650,70.5,"DTC", fontsize = small, c = 'orange')
plt.text(1650,69,"LR", fontsize = small, c = 'r')

ax.set_xlabel('Number of each suspect and non-suspect players'); ax.set_ylabel('Validation accuracy [%]')

plt.tight_layout() 
plot = "sampled_plot"; png = "%s.png" % (plot);
eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
plt.savefig(png);  os.system(eps);
plt.show(); print("Plot written to", png)
plt.clf(); plt.cla(); plt.close()
