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

