## Script to try out the Adaptive Learning (al) scripts from Rasmus Bruckner
#
# Felix Klaassen

import numpy as np
#import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from al_plot_utils import latex_plt, cm2inch, label_subplots, plot_image

# STEP 0 
# set task parameters for simulation
nTrials = 200
outcomeRange = (0, 300) # change this to 360 for circular data
outStd = 10
hazard = 0.1
initialGuess = np.around(outcomeRange[1]/2, decimals=0)

## STEP 1
# generate task trials (outcomes)...
isCP = np.full([nTrials, 1],np.nan) # binary changepoint variable
Mu = np.full([nTrials, 1],np.nan) # mean of distribution
outcomes = np.full([nTrials, 1],np.nan) # outcome

# loop through the trials and get changepoints and means
for t in range(0,nTrials):
    # flip a coin to determine whether this will be a changepoint
    if np.random.uniform(0,1) < hazard or (t == 0):
        isCP[t] = True
        Mu[t] = np.random.uniform(0,1) * (outcomeRange[1]-outcomeRange[0]) + outcomeRange[0]
    else:
        isCP[t] = False
        Mu[t] = Mu[t-1]
    outcomes[t] = np.random.normal(Mu[t], outStd)
    # normalize values outside of outcomeRange
    if outcomes[t] < outcomeRange[0]:
        outcomes[t] = abs(outcomes[t])
    elif outcomes[t] > outcomeRange[1]:
        outcomes[t] = outcomeRange[1] - (outcomes[t]-outcomeRange[1])

# vector for model predictions
B = np.full([nTrials, 1],np.nan) # predictions
B[0] = initialGuess

mu = np.full([nTrials, 1],np.nan) # beliefs (same as B?)

# vector for prediction errors
PEs = np.full([nTrials, 1],np.nan) # prediction errors

# vector for helicopter visibility (0 = invisible, 1 = visible)
v = np.full([nTrials, 1],0) # fill with 0s for now, assume no catch trials

# vector for high-value/reward trials
hv = np.full([nTrials, 1],0) # fill with 0s for now, assume no high-value trials

# vector for trial-by-trial updates
Up = np.full([nTrials, 1],np.nan)

# vector for relative uncertainty
RU = np.full([nTrials, 1],np.nan)

# vector for changepoint probabilities
CPP = np.full([nTrials, 1],np.nan)

# vector for learning rates
LR = np.full([nTrials, 1],np.nan)

## STEP 2
# initialize reduced bayesian mod variables
from AlAgentVarsRbm import AgentVars
redBayesVars = AgentVars()

# Run reduced bayesian model to get predictions...
import AlAgentRbm
agent = AlAgentRbm.AlAgent(redBayesVars)

# model needs to loop through the outcomes and learn
for t in range(0,nTrials):

    # get input parameters for this trial
    RU[t] = agent.tau_t  # relative uncertainty of current trial
    b_t = B[t]          # current belief (for first trial is an initialGuess)
    Mu_t = outcomes[t]  # actual helicopter position (i.e., outcome of the trial)

    delta_t = Mu_t - b_t # compute prediction error
    PEs[t] = delta_t # store PE for later

    v_t = v[t]       # helicopter visibility trials (currently all set to 0)
    hv_t = hv[t]     # high value trials (currently all set to 0)

    # let the model learn based on these values
    agent.learn(delta_t, b_t, v_t, Mu_t, hv_t)

    # get model predictions
    Up[t] = agent.a_t    # predicted update (for next trial)
    CPP[t] = agent.omega_t # changepoint probability of current trial
    LR[t] = agent.alpha_t  # learning rate of current trial

    mu[t] = agent.mu_t # updated belief (this is pretty much identical to B[t+1])

    # next prediction = current prediction + update
    if t < nTrials-1:
        B[t+1] = b_t + agent.a_t
        if B[t+1] > outcomeRange[1]: # normalize prediction to fit within the outcomeRange
            B[t+1] = outcomeRange[1]
        elif B[t+1] < outcomeRange[0]:
            B[t+1] = outcomeRange[0]

# plot mean, outcomes, and model predictions (Mu)
fig_width = 17
fig_height = 8.5*2.25
plt.figure(figsize=cm2inch(fig_width, fig_height))

plt.subplot(4,1,1)
trialnrs = np.linspace(1, nTrials, nTrials)
plt.plot(trialnrs, outcomes,'o',           # outcomes
         markerfacecolor = 'r', markeredgecolor = 'k',
         linewidth = 1)
plt.plot(trialnrs, Mu,'--k',linewidth = 1) # mean of the distributions
plt.plot(trialnrs, B,'-b', linewidth = 2)  # model predictions
plt.ylim((outcomeRange[0], outcomeRange[1]))
plt.ylabel('Value')
plt.legend(('Outcome','Helicopter mean','Model'),loc='best')
#plt.legend(('Outcome','Mu','Prediction'),
#           loc = 'upper center', bbox_to_anchor = (0.5, 1.05),
#           ncol = 3)

# Plot RU and CPP
plt.subplot(4,1,2)
plt.plot(trialnrs, CPP,'-c')
plt.plot(trialnrs, RU, '-m')
plt.ylim((0,1.1))
plt.ylabel('Variable')
plt.yticks(np.arange(0,1.5,0.5))
plt.legend(('CPP','RU'))
#plt.show()

# plot learning rates
plt.subplot(4,1,3)
plt.plot(trialnrs, LR, '-k')
plt.ylim((0,1.1))
plt.ylabel('alpha')
plt.yticks(np.arange(0,1.5,0.5))

# plot PEs
plt.subplot(4,1,4)
plt.plot(trialnrs, PEs, '-k')
#plt.ylim((0,1.1))
plt.xlabel('Trials'); plt.ylabel('Prediction Error')
plt.show()