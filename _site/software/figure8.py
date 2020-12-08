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
df.columns = ['concentration (uM)', 'time (s)']
df = df.dropna()

# Extract data for 12 uM tubulin
df12 = df.loc[df['concentration (uM)'] == 12, 'time (s)']

# Set up and seed random number generator
rg = np.random.default_rng(seed = 123)

# Log likelihood function for Gamma distribution
def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma measurements, parametrized by alpha and beta"""
    
    alpha, beta = params
    
    if min(params) <= 0:
        return -np.inf
    
    if np.isclose(alpha, 0) or np.isclose(beta ** 2, 0):
        return -np.inf
    
    return np.sum(st.gamma.logpdf(n, alpha, loc=0, scale=1/beta))

# print out α and β from MLEs
n = df12.values

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    res = scipy.optimize.minimize(
        fun=lambda params, n: -log_like_iid_gamma(params, n),
        x0=np.array([3, 3]),
        args=(n,),
        method='Powell'
    )

if res.success:
    alpha_mle, beta_mle = res.x
else:
    raise RuntimeError('Convergence failed with message', res.message)


single_samples_gamma = np.array(
    [rg.gamma(alpha_mle, 1/beta_mle, size=len(n)) for _ in range(100000)]
)

# Define log likelihood function for Poisson distribution
def log_like_t_model(t, b1, delta_b):
    '''log likelihood '''
    f = ((b1 * (delta_b + b1)) / delta_b) * math.exp(-b1 * t) * (1 - math.exp(-delta_b * t))
    
    if f <= 0 or np.isclose(f, 0):
        return -np.inf
    
    return math.log(f)

def log_like_iid_model(params, n):
    """Log likelihood for i.i.d. Poisson measurements parametrized by β1 and Δβ."""
    
    b1, delta_b = params
    
    if min(params) <= 0:
        return -np.inf
    
    if np.isclose(b1, 0) or np.isclose(delta_b, 0):
        return -np.inf
    
    log_pdf = []
    
    for t in n:
        log_pdf.append(log_like_t_model(t, b1, delta_b))
    
    return np.sum(log_pdf)

# print values for β1, β2 and Δβ
n = df12.values

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    res = scipy.optimize.minimize(
        fun=lambda params, n: -log_like_iid_model(params, n),
        x0=np.array([0.0001, 0.0001]),
        args=(n,),
        method='Powell',
        options = {'maxiter': 1000} 
    )

if res.success:
    b1_mle, delta_b_mle = res.x
else:
    raise RuntimeError('Convergence failed with message', res.message)

# Generate x-values for plotting predictive ECDF, Poisson distribution
def gen_model(beta_1, delta_b, size, rg):
    '''generate samples for Poisson model'''
    beta_2 = beta_1 + delta_b
    
    t = rg.exponential(1/beta_1, size = size) + rg.exponential(1/beta_2, size = size)
    
    return t

single_samples_poisson = np.array([gen_model(b1_mle, 
                                             delta_b_mle, 
                                             size=len(df12), 
                                             rg=np.random.default_rng()) 
                                   for _ in range(10000)])

# Generate Q-Q plots
p_gamma = bebi103.viz.qqplot(
    data=df12,
    samples=single_samples_gamma,
    x_axis_label="time (s)",
    y_axis_label="time (s)",
    title="Gamma"
)

p_poisson = bebi103.viz.qqplot(
    data=df12,
    samples=single_samples_poisson,
    x_axis_label="time (s)",
    y_axis_label="time (s)",
    title="Poisson"
)

plots = p_gamma, p_poisson

bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))
