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
