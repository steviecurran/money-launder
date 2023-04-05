#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

infile = "DL_loops=100-b'2023-04-04-12:13:10'.csv"
df = pd.read_csv(infile); print(df)

big = 14; small = 12 # text sizes

plt.rcParams.update({'font.size': big})
plt.figure(figsize=(6,5))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)

plt.ylabel('Number', size=big); plt.xlabel('Mean scores [%]', size=big)

para = 100*df['All']
ax.hist(para, bins=int(para.max() - para.min()), color="w", edgecolor='r', linewidth=3);  
x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
#ax.set_ylim([0, 1.5*y2]) # EXTEND y-axis TO FIT TEXT
#y1, y2 = ax.get_ylim()
x_pos = 1.03*x1; y_pos = 0.95*y2; step = (y2-y1)/16
mean = np.mean(para); std = np.std(para); 
text = "All:    \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
plt.text(x_pos,y_pos,text, fontsize = small, c = 'r')

para = 100*df['set_kNN']
ax.hist(para, bins=int(para.max() - para.min()), color="w", edgecolor='b', linewidth=3, alpha=0.8);  
mean = np.mean(para); std = np.std(para); 
text = "kNN: \u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std)
plt.text(x_pos,y_pos-step,text, fontsize = small, c = 'b')

plt.tight_layout()
plot = "DL_histos"; png = "%s.png" %(plot)
eps = "convert %s %s.eps; mv %s.eps  media/." % (png, plot,plot); 
#plt.savefig(png); os.system(eps);  print("Plot written to", png);
plt.show()
