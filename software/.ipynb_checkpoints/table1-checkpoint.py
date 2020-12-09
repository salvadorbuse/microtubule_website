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

# Generate dataframe containing estimated paramters and AIC values for the 
# Gamma and Poisson distributions for 12 µM tubulin
df_mle = pd.DataFrame(index=['alpha_Gamma', 'beta_Gamma', 'beta1_Poisson', 'beta2_Poisson'])
df_mle['12 µM'] = [alpha_mle, beta_mle, b1_mle, b1_mle + delta_b_mle]

ell_gamma = log_like_iid_gamma((alpha_mle, beta_mle), df12.values)
df_mle.loc["log_like_Gamma"] = ell_gamma

ell_poisson = log_like_iid_model((b1_mle, b1_mle + delta_b_mle), df12.values)
df_mle.loc["log_like_Poisson"] = ell_poisson

df_mle.loc['AIC_Gamma'] = -2 * (df_mle.loc['log_like_Gamma'] - 2)
df_mle.loc['AIC_Poisson'] = -2 * (df_mle.loc['log_like_Poisson'] - 2) 

# Calculate the Aikaike weights and add them to the dataframe
AIC_max = max(df_mle.loc[['AIC_Gamma', 'AIC_Poisson'], '12 µM'])

numerator_gamma = np.exp(-(df_mle.loc['AIC_Gamma', '12 µM'] - AIC_max) / 2)
numerator_poisson = np.exp(-(df_mle.loc['AIC_Poisson', '12 µM'] - AIC_max) / 2)
denominator = numerator_gamma + numerator_poisson

df_mle.loc['AW_Gamma', '12 µM'] = numerator_gamma / denominator 
df_mle.loc['AW_Poisson', '12 µM'] = numerator_poisson / denominator

df_mle
