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

# Generate predictive ECDF for Gamma distribution
def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

single_samples_gamma = np.array(
    [rg.gamma(alpha_mle, 1/beta_mle, size=len(n)) for _ in range(100000)]
)

n_theor_gamma = np.arange(0, 1.2 * max(df12))

ecdfs_gamma = np.array([ecdf(n_theor_gamma, sample) for sample in single_samples_gamma])

ecdf_low_gamma, ecdf_high_gamma = np.percentile(ecdfs_gamma, [2.5, 97.5], axis=0)

# Generate ECDF for 12 µM tubulin
p = iqplot.ecdf(
    data=df.loc[df['concentration (uM)'] == 12], 
    q = "time (s)", 
    palette = ['orange']
)

p = bebi103.viz.fill_between(
    x1=n_theor_gamma,
    y1=ecdf_high_gamma,
    x2=n_theor_gamma,
    y2=ecdf_low_gamma,
    patch_kwargs={"fill_alpha": 0.5},
    x_axis_label="time (s)",
    y_axis_label="ECDF",
    title="Gamma"
)

# Overlay ECDF with predictive ECDF
p_gamma = iqplot.ecdf(data=df.loc[df['concentration (uM)'] == 12], 
                      q = "time (s)", palette=['orange'], 
                      p=p)

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

# Generate predictive ECDF, Poisson distribution
n_theor_poisson = np.arange(0, 1.2 * max(df12))

ecdfs_poisson = np.array([ecdf(n_theor_poisson, sample) 
                          for sample in single_samples_poisson])

ecdf_low_poisson, ecdf_high_poisson = np.percentile(ecdfs_poisson, [2.5, 97.5], axis=0)

# Generate ECDF for 12 uM tubulin
p = iqplot.ecdf(
    data=df.loc[df['concentration (uM)'] == 12], 
    q="time (s)", 
    palette = ['orange']
)

p = bebi103.viz.fill_between(
    x1=n_theor_poisson,
    y1=ecdf_high_poisson,
    x2=n_theor_poisson,
    y2=ecdf_low_poisson,
    patch_kwargs={"fill_alpha": 0.5},
    x_axis_label="time (s)",
    y_axis_label="ECDF",
    title="Poisson"
)

# Overlay ECDF with predictive ECDF
p_poisson = iqplot.ecdf(data=df.loc[df['concentration (uM)'] == 12], 
                        q = "time (s)", 
                        palette=['orange'], p=p)

# Show the plots
bokeh.io.show(row(p_gamma, p_poisson))
