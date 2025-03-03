#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import os 
import sys
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

#infile = "DL_loops=100-b'2023-04-04-12:13:10'.csv" # training scores
infile = "DL_loops=100-b'2023-05-11-12:33:57'.csv" # test scores 

#infile = "DL_loops=100-b'2025-03-03-13:38:08'.csv"
df1 = pd.read_csv("DL_loops=100-b'2023-05-11-12:33:57'.csv")
df2 = pd.read_csv("DL_loops=100-b'2025-03-03-07:55:40'.csv")

font = 14;

plt.rcParams.update({'font.size': font})
plt.figure(figsize=(6,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)
ax.tick_params(direction='in', pad = 7,length=6, width=1.5, which='major',right=True,top=True)
ax.tick_params(direction='in', pad = 7,length=3, width=1.5, which='minor',right=True,top=True)

min_val = 100*min(df1['All']); max_val = 100*max(df1['All'])
dbs = 1
min_boundary = -1.0 * (min_val % dbs - min_val)
max_boundary = max_val - max_val % dbs + dbs
n_bins = int((max_boundary - min_boundary) / dbs) + 1
bins = np.linspace(min_boundary, max_boundary, n_bins)

para = 100*df2['All']
mean = np.mean(para); std = np.std(para); 
ax.hist(para, bins=bins, color="w", edgecolor='b', linewidth=2, zorder = 2, alpha=1,
        label = "Best:    \u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std));  

para = 100*df1['All']
mean = np.mean(para); std = np.std(para); 
ax.hist(para, bins=bins, color="r", edgecolor='r', linewidth=2, zorder = 1,
        label = "Simple: \u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std));  

df3 = df1[df1['All'] < 0.5]
para = 100*df3['All']; print(para)
ax.hist(para, bins=bins, color="r", edgecolor='r', linewidth=2, zorder = 3) # TO SHOW OULIER ON TOP

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

ax.legend(fontsize = 0.8*font,loc="upper left")
plt.ylabel('Number', size=font); plt.xlabel('Validation accuracy [%]', size=font)


plt.tight_layout()
plot = "DL_histos"
plt.savefig("media/%s.eps" %(plot), format = 'eps');
print("Plot written to %s.eps" %(plot));
#eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 

plt.show()
