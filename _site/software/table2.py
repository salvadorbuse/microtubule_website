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

# Define log likelihood function for Gamma distribution
def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma measurements, parametrized by alpha and beta"""
    alpha, beta = params
    
    if min(params) <= 0:
        return -np.inf
    
    if np.isclose(alpha, 0) or np.isclose(beta ** 2, 0):
        return -np.inf
    
    return np.sum(st.gamma.logpdf(n, alpha, loc=0, scale=1/beta))

# Find the MLEs, parameters of the Gamma distribution
def find_gamma_mles(df):
    '''Generate dataframe of MLE values from dataframe of experimental data'''
    concentrations = df['concentration (µM)'].unique()
    
    mles = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for concentration in concentrations:

            times = df.loc[df["concentration (µM)"] == concentration, "time (s)"].values
            
            res = scipy.optimize.minimize(
                fun = lambda params, n: -log_like_iid_gamma(params, n),
                x0=np.array([2.5, 0.1]),
                args=(times,),
                method='Powell'
            )

            if res.success:
                alpha_mle, beta_mle, = res.x
            else:
                raise RuntimeError('Convergence failed with message', res.message)

            
            mles.append(list(res.x))
    
    mles = np.array(mles).T
    
    mles = pd.DataFrame(data ={
        'concentration (µM)':concentrations,
        'alpha':mles[0],
        'beta':mles[1],
        'alpha/beta':mles[0] / mles[1]
    })
    
    return mles

# Generate dataframe with MLE values
df_mles = find_gamma_mles(df)
df_mles
