#!/usr/bin/python3
import numpy as np
import pandas as pd
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0])

# ## LOOK AT DATA
df1 = pd.read_csv("details.csv"); #print(df1) 
df2 = pd.read_csv("details_fixed.csv"); #print(df2) 
#print(df2.describe())

df3 = pd.read_csv("payments.csv"); print(df3) 
df4 = pd.read_csv("profiling.csv"); print(df4) 

## MERGE BY ID
df23 = df2.merge(df3,on = 'ID')
df234 = df23.merge(df4,on = 'ID'); print(df234.describe())

## CHECK FOR MISSING VALUES
def missing(data):
    cols = data.columns 
    for (i,col) in enumerate(cols):
        print("Column %s  - missing values = " %(col), data[col].isnull().sum())
missing(df234)

## SUSPECT PLAYERS
df5 = pd.read_csv("suspect.csv"); print(df5); print(df5.describe())
print(df234.loc[df234['ID'] == 100095464]) # LOOKS LIKE THESE ALREADY IN OTHER DATAFRAME

df2345 = pd.merge(df234, df5, how='outer', on='ID');
df2345['Target_ml'] = df2345['Target_ml'].replace(np.nan, 0)
df2345['Target_ml'] = df2345['Target_ml'].astype(int)
print(df2345); print(df2345.describe())

## SAVE DATA 
#df2345.to_csv('all_data.csv', index = False)

## PLOTS HISTOGRAMS OF FEATURES
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import os 
import sys

df = df2345.copy()
#df = pd.read_csv("all_data.csv");
#print(df.columns)
#paras = ['e_23456'] # TO TEST
paras = ['Life_Time', 'Age', 'CountDeposit', 'CountWithdrawal', 'TotalDeposits', 'TotalWithdrawal', 'CountPaymentMethod', 'DifferentMethodWithdrawals', 'IP_Counts', 'e_11234', 'e_23456', 'e_34454', 'e_43568','e_64645']

for (i, para) in enumerate(paras):
    xmin = min(df[para]); 
    xmax = max(df[para]);
    
    if para == 'CountPaymentMethod':
        desired_bin_size = 1
    elif para == 'IP_Counts':
        desired_bin_size = 1
    elif para == 'e_64645':
        desired_bin_size = 1
    else:
        desired_bin_size =  int((xmax-xmin)/10)
    
    data = df[para]
    min_val = np.min(data); max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.setp(ax.spines.values(),linewidth=2)

    text = '%s' %(para)
    plt.ylabel('Number', size=14); plt.xlabel(text, size=14)
    sus = df.loc[df['Target_ml'] == 1]
    non = df.loc[df['Target_ml'] == 0]; #print(len(sus),len(non))
    
    ax.hist(non[para], bins=bins, color="w", edgecolor='grey', linewidth=3);
    mean = np.mean(non[para]); std = np.std(non[para])

    ymin, ymax = plt.ylim(); #print(ymax)
         
    if xmax < 10:
        text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std); xoff = 3.9
    elif xmax >10 and mean <= 100:
        text = "\u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std); xoff = 2.9
    elif xmax >100 and mean <= 1000:
        text = "\u03BC = %1.1f, \u03C3 = %1.0f" %(mean,std);xoff = 2.5  
    elif xmax > 1000 and mean <= 10000:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 3.5
    else:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std); xoff = 2.5
        
   
    x_pos = xmax-((xmax-xmin)/xoff); y_pos = ymax; y_skip =  ymax-((ymax-ymin)/3);  
    plt.text(x_pos,y_pos, text, fontsize = 12, c = 'k', horizontalalignment='left',verticalalignment='top') 
    
    ax.hist(sus[para], bins=bins, color="w", edgecolor='r', linewidth=3);
    mean = np.mean(sus[para]); std = np.std(sus[para])
    if xmax < 10:
        text = "\u03BC = %1.2f, \u03C3 = %1.2f" %(mean,std); xoff = 3.9
    elif xmax >10 and mean < 100:
        text = "\u03BC = %1.1f, \u03C3 = %1.1f" %(mean,std); xoff = 2.9
    elif xmax >100 and mean < 1000:
        text = "\u03BC = %1.1f, \u03C3 = %1.0f" %(mean,std);xoff = 2.5  
    else:
        text = "\u03BC = %1.0f, \u03C3 = %1.0f" %(mean,std);xoff = 2.2
    
    plt.text(x_pos,y_pos-y_skip, text, fontsize = 12, c = 'r', horizontalalignment='left',verticalalignment='top') 
    
    ax.set_yscale('log');
    def update_ticks(z, pos):
        if z ==1:
            return '1 '
        elif z >1 and z <1000:
            return '%d' %(z)
        elif z < 1 and z > 0.001:
            return z
        else:
            return  '10$^{%1.0f}$' %(np.log10(z)) 

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
      
    plt.tight_layout(pad=0.1)
    plot = "histo_%s" %(para); png = "%s.png" %(plot);
    eps = "convert %s %s.eps; mv %s.eps media/." % (png, plot,plot); 
    plt.savefig(png); os.system(eps);print("Plot written to", png);
    #plt.show()
    plt.clf(); plt.cla(); plt.close()
