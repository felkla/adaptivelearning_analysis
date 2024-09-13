""" Figure S12: Parameter recovery second experiment

    1. Load data
    2. Prepare figure
    3. Plot recovery results
    4. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, latex_plt


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

# Recovery files obtained from "al_recovery.py" script
param_recov = pd.read_pickle('al_data/param_recov_exp2.pkl')
true_params = pd.read_pickle('al_data/true_params_recov_exp2.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Create figure
fig_height = 15
fig_width = 15
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Y-label distance
ylabel_dist = -0.25

# ------------------------
# 3. Plot recovery results
# ------------------------

# omikron 0
plt.subplot(331)
plt.plot(true_params['omikron_0'].values, param_recov['omikron_0'].values, '.', color='k')
plt.title('Motor Noise')
plt.xlim(-0.5, 11)
plt.ylim(-0.5, 11)
plt.xticks(np.arange(0.0, 11, 2))
plt.yticks(np.arange(0.0, 11, 2))
plt.ylabel('Recovered Parameter')
plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

# omikron 1
plt.subplot(332)
plt.plot(true_params['omikron_1'].values, param_recov['omikron_1'].values, '.', color='k')
plt.title('Learning-Rate Noise')
plt.xlim(-0.05, 0.8)
plt.ylim(-0.05, 0.8)
plt.xticks(np.arange(0.0, 0.85, 0.2))
plt.yticks(np.arange(0.0, 0.85, 0.2))

# b_0
plt.subplot(333)
plt.plot(true_params['b_0'].values, param_recov['b_0'].values, '.', color='k')
plt.title('Intercept')
plt.xlim(-10, 35)
plt.ylim(-10, 35)
plt.xticks(np.arange(-10, 35, 10))
plt.yticks(np.arange(-10, 35, 10))

# b_1
plt.subplot(334)
plt.plot(true_params['b_1'].values, param_recov['b_1'].values, '.', color='k')
plt.xlim(-1.6, 1.1)
plt.ylim(-1.6, 1.1)
plt.xticks(np.arange(-1.5, 1.1, 0.5))
plt.yticks(np.arange(-1.5, 1.1, 0.5))
plt.title('Slope')
plt.ylabel('Recovered Parameter')
plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

# u
plt.subplot(335)
plt.plot(true_params['u'].values, param_recov['u'].values, '.', color='k')
plt.title('Uncertainty Underestimation')
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.xticks(np.arange(-2, 12, 2))
plt.yticks(np.arange(-2, 12, 2))

# s
plt.subplot(336)
plt.plot(true_params['s'].values, param_recov['s'].values, '.', color='k')
plt.title('Surprise Sensitivity')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))

# h
plt.subplot(337)
plt.plot(true_params['h'].values, param_recov['h'].values, '.', color='k')
plt.title('Hazard Rate')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xlabel('True Parameter')
plt.ylabel('Recovered Parameter')
plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

# d
plt.subplot(338)
plt.plot(true_params['d'].values, param_recov['d'].values, '.', color='k')
plt.title('Anchoring Bias')
plt.xlim(-0.01, 0.6)
plt.ylim(-0.01, 0.6)
plt.xticks(np.arange(0, 0.6, 0.1))
plt.yticks(np.arange(0, 0.6, 0.1))
plt.xlabel('True Parameter')

# sigma_H
plt.subplot(339)
plt.plot(true_params['sigma_H'].values, param_recov['sigma_H'].values, '.', color='k')
plt.title('Catch Trial')
plt.xlim(-1, 33)
plt.ylim(-1, 33)
plt.xlabel('True Parameter')
plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']  # label letters
label_subplots(f, texts, x_offset=0.07, y_offset=0.01)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_12.pdf"
plt.savefig(savename)

# Show figure
plt.show()
