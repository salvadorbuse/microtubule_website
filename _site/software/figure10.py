# -*- coding: utf-8 -*-
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

import pandas as pd

import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import iqplot
import math

import os, sys, subprocess
import bebi103

import bokeh.io
import bokeh.plotting
from bokeh.layouts import row, grid
import iqplot

bokeh.io.output_notebook()

# Load the data into a data frame
fname = os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv")

# Tidy the data
df = pd.read_csv(fname, skiprows = 9)
cols = list(df.columns)
cols = list(map(lambda x: int(x.split(' ')[0]), cols))
df.columns = cols
df = df[sorted(cols)]
df = df.melt()
df.columns = ['concentration (µM)', 'time (s)']
df = df.dropna()

# Generate ECDFs for catastrophe time at various tubulin concentrations
p = iqplot.ecdf(
    data = df,
    q = 'time (s)',
    cats = ['concentration (µM)'])

bokeh.io.show(p)
