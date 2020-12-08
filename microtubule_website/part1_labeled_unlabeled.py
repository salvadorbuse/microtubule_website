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
df6.head()

# Plot the data with an overlaid 95% confidence interval
p = iqplot.ecdf(
    data=df6,
    q='time to catastrophe (s)',
    cats='labeled',
    conf_int=True,
    height = 500,
    width = 500,
    title = 'ECDF of catastrophe times'
)

bokeh.io.show(p)

# extract the labeled and unlabeled datasets as np arrays

labeled = df6.loc[df6["labeled"] == 'labeled', "time to catastrophe (s)"].values

unlabeled = df6.loc[df6["labeled"] == 'unlabeled', "time to catastrophe (s)"].values

plug_in_mean_labeled = np.mean(labeled)
plug_in_mean_unlabeled = np.mean(unlabeled)

frac_diff = plug_in_mean_unlabeled / plug_in_mean_labeled

# define a function for drawing a bootstrap sample 
# (n values drawn randomly with replacement from a dataset of size n)
# this function was defined in the notes

@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

# define a function which repeatedly ('size' times) extracts a bootstrap sample, calculates the mean,
# and returns these means in an np array

@numba.njit
def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

# create these arrays of means for the labeled and unlabeled waiting times
bs_mean_labeled = draw_bs_reps_mean(labeled, size = 10**4)
bs_mean_unlabeled = draw_bs_reps_mean(unlabeled, size = 10**4)

# use np.percentile to extract the 95% confidence interval
mean_labeled_conf_int = np.percentile(bs_mean_labeled, [2.5, 97.5])
mean_unlabeled_conf_int = np.percentile(bs_mean_unlabeled, [2.5, 97.5])

mid_labeled_conf_int = 0.5 * sum(mean_labeled_conf_int)
delta_labeled_conf_int = 100 * ((plug_in_mean_labeled - mid_labeled_conf_int) / plug_in_mean_labeled) 

mid_unlabeled_conf_int = 0.5 * sum(mean_unlabeled_conf_int)
delta_unlabeled_conf_int = 100 * ((plug_in_mean_unlabeled - mid_unlabeled_conf_int) / plug_in_mean_unlabeled) 

# Generate plot for bootstrapped means
df_bs = pd.DataFrame(
    data={
        "labeled": ["labeled"] * len(bs_mean_labeled) + ["unlabeled"] * len(bs_mean_unlabeled),
        'mean time to catastrophe (s)': np.concatenate((bs_mean_labeled, bs_mean_unlabeled)),
    }
)


p = iqplot.ecdf(
    data = df_bs, 
    q = 'mean time to catastrophe (s)',
    cats = 'labeled',
    height = 500,
    width = 500
)

p.legend.location = "top_left"

bokeh.io.show(p)

# generates permutation samples by randomly shuffling the x and y datasets
@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

# find the difference in means for 'size' permutation samples of x and y
@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation mean differences."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out


# find the difference in variances for 'size' permutation samples of x and y
@numba.njit
def draw_perm_reps_diff_var(x, y, size=1):
    """Generate array of permuation variance differences."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.var(x_perm) - np.var(y_perm)

    return out


# find the difference in skewness for 'size' permutation samples of x and y
def draw_perm_reps_diff_skew(x, y, size=1):
    """Generate array of permuation skewness differences."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = scipy.stats.skew(x_perm) - scipy.stats.skew(y_perm)

    return out

# Compute statistical functional differences for original data set
diff_mean = np.mean(labeled) - np.mean(unlabeled)
diff_var = np.var(labeled) - np.var(unlabeled)
diff_skew = scipy.stats.skew(labeled) - scipy.stats.skew(unlabeled)

# Draw replicates
perm_reps_mean = draw_perm_reps_diff_mean(labeled, unlabeled, size=10**4)
perm_reps_var = draw_perm_reps_diff_var(labeled, unlabeled, size=10**4)
perm_reps_skew = draw_perm_reps_diff_skew(labeled, unlabeled, size=10**4)

# Compute p-value
p_val_mean = np.sum(perm_reps_mean >= diff_mean) / len(perm_reps_mean)
p_val_var = np.sum(perm_reps_var >= diff_var) / len(perm_reps_var)
p_val_skew = np.sum(perm_reps_skew >= diff_skew) / len(perm_reps_skew)

# use theory to estimate confidence intervals: we make use of built-in scipy.stats functions
# here, loc is the mean, so we use np.mean
# scale is the standard deviation, so we take the square root of the variance, whose formula 
# is given in the problem statement: the variance of the raw data, divided by n

mean_labeled_conf_int_theory = scipy.stats.norm.interval(
    0.95, scale = math.sqrt(np.var(labeled) / len(labeled)), loc = np.mean(labeled))

mean_unlabeled_conf_int_theory = scipy.stats.norm.interval(
    0.95, scale = math.sqrt(np.var(unlabeled) / len(unlabeled)), loc = np.mean(unlabeled))

# plot the ECDF of the bootstrap means: same as plot above
p = iqplot.ecdf(
    data = df_bs, 
    q = 'mean time to catastrophe (s)',
    cats = 'labeled',
    outline_line_alpha = 0.5,
    height = 500,
    width = 500
)

p.legend.location = "top_left"

# generate the theoretical distribution: sample points directly from the normal distribution
ecdf_ys = np.arange(1, 10**4 +1) / 10**4

labeled_theory = np.random.normal(scale = math.sqrt(np.var(labeled) / len(labeled)), 
                                  loc = np.mean(labeled), size = 10 ** 4)

unlabeled_theory = np.random.normal(scale = math.sqrt(np.var(unlabeled) / len(unlabeled)), 
                                  loc = np.mean(unlabeled), size = 10 ** 4)

# overlay the theoretical distribution as black lines on top of the ECDFs
p.line(np.sort(labeled_theory), ecdf_ys, color = 'black', line_width = 2)
p.line(np.sort(unlabeled_theory), ecdf_ys, color = 'black', 
       line_width = 2, legend_label = 'theoretical ECDF')


bokeh.io.show(p)

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
