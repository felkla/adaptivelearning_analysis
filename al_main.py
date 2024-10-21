## Script to try out the Adaptive Learning (al) scripts from Rasmus Bruckner
#
# Felix Klaassen

import numpy as np

# STEP 0 
# set task parameters for simulation
nTrials = 200
outcomeRange = (0, 300) # change this to 360 for circular data
outStd = 10
hazard = 0.1

# get reduced bayesian mod variables
from AlAgentVarsRbm import AgentVars
redBayesVars = AgentVars()

## STEP 1
# generate task trials (outcomes)...
isCP = np.full([nTrials, 1],np.nan) # binary changepoint variable
Mu = np.full([nTrials, 1],np.nan) # mean of distribution
outcomes = np.full([nTrials, 1],np.nan) # outcome

# loop through the trials and get changepoints and means
for t in range(1,nTrials+1):
    # flip a coin to determine whether this will be a changepoint
    if np.random.uniform(0,1) < hazard or (t == 1):
        isCP[t-1] = True
        Mu[t-1] = np.random.uniform(0,1) * (outcomeRange[1]-outcomeRange[0]) + outcomeRange[0]
    else:
        isCP[t-1] = False
        Mu[t-1] = Mu[t-2]
    outcomes[t-1] = np.random.normal(Mu[t-1], outStd)

## STEP 2
# Run reduced bayesian model to get predictions...
import AlAgentRbm
agent = AlAgentRbm.AlAgent(redBayesVars)
agent.learn() # model needs to loop through the outcomes and learn