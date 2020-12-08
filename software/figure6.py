# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot colorcet bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../datasets/"
# ------------------------------

import os

import pandas as pd

import numpy as np

import math

import numba

import scipy.stats

import iqplot

import bokeh.io
from bokeh.plotting import figure
from bokeh.layouts import row

bokeh.io.output_notebook()

# Load the data into a data frame
fname = os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv")

df6 = pd.read_csv(fname)
df6['labeled'] = df6['labeled'].apply(lambda x: 'labeled' if x == True else 'unlabeled')

# extract the labeled and unlabeled datasets as np arrays

labeled = df6.loc[df6["labeled"] == 'labeled', "time to catastrophe (s)"].values

unlabeled = df6.loc[df6["labeled"] == 'unlabeled', "time to catastrophe (s)"].values

def ecdf(x, data):
    '''Approximates F(x), the ECDF value for each point x in the vector x'''
    
    # generate set of x and y values for the ECDF
    x_vals = np.sort(data)
    n = len(x_vals)
    y_vals = [i/n for i in list(range(1, n + 1))]
    
    # make an array of differences for interpolation
    diffs_x = np.diff(x_vals)
    diffs_y = np.diff(y_vals)
    
    # handles input of ints rather than arrays
    if type(x) == int:
        x = [x]
    
    x = np.array(x)
    
    F_s = []
    
    for x_val in x:
        # if x is below the lowest datapoint, add 0 (so F still integrates to 1)
        if x_val < x_vals[0]:
            F_s.append(0)
            continue
        
        elif x_val in x_vals:
            F = (np.where(x_vals == x_val)[0][-1] + 1) / n
            F_s.append(F)
            continue
            
        elif x_val > x_vals[-1]:
            F_s.append(1)
            continue
            
        lower_index = len(data[np.where(data < x_val)]) - 1

        diff_x = diffs_x[lower_index]        
        diff_y = diffs_y[lower_index] 

        lower_x = x_vals[lower_index] 
        lower_y = y_vals[lower_index] 

        F = lower_y + diff_y * ((x_val - lower_x) / diff_x)

        F_s.append(F)
    
    return F_s

def confidence_bounds(data, alpha = 0.05):
    '''takes data and returns values to make ECDF plots of the data (F), 
    the lower bound (L), and the upper bound (U)'''
    
    x = np.sort(data)
    n = len(x)
    
    # regular ECDF
    F = np.array([i/n for i in list(range(1, n + 1))])
    
    # calculate epsilon using DKW, given in problem statement
    eps = math.sqrt(math.log(2 / alpha) / (2*len(data)))
    
    # y-coordinates for the lower (L) and upper (U) bounds of the ECDF
    L = np.array([max(0, f - eps) for f in F])
    U = np.array([min(1, f + eps) for f in F])
    
    return x, F, L, U

# calculate these bounds for the labeled and unlabeled data
[x_lab, F_lab, L_lab, U_lab] = confidence_bounds(labeled, alpha = 0.05)
[x_unlab, F_unlab, L_unlab, U_unlab] = confidence_bounds(unlabeled, alpha = 0.05)


# plot the ECDFs together with confidence intervals
p = figure(plot_width=500, 
           plot_height=500, 
           title = 'Catatstrophe time ECDFs with confidence intervals from theory')

p.xaxis.axis_label = "time to catastrophe (s)"
p.yaxis.axis_label = "ECDF"


# use p.varea to vertically color the area between the upper and lower bounds
# and then add on the raw ECDFs
p.varea(x = x_lab, 
        y1 = L_lab, 
        y2 = U_lab, 
        alpha = 0.4, 
        legend_label = 'labeled')

p.circle(x = x_lab, 
         y = F_lab, 
         legend_label = 'labeled')

p.varea(x = x_unlab,
        y1 = L_unlab, 
        y2 = U_unlab, 
        color = 'darkorange', 
        alpha = 0.4,
        legend_label = 'unlabeled')

p.circle(x = x_unlab, 
         y = F_unlab, 
         color = 'darkorange', 
         legend_label = 'unlabeled')

p.legend.location = "bottom_right"


# plot the original ECDF with built-in confidence intervals
p2 = iqplot.ecdf(
    data=df6,
    q='time to catastrophe (s)',
    cats='labeled',
    conf_int=True,
    height = 500,
    width = 500,
    title = 'Catatstrophe time ECDFs with bootstrapped confidence intervals'
)

bokeh.io.show(row(p, p2))
