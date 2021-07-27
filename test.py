import CAPT
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime

degree = 3
def createTrainedLocalGNN(state_dic_location, dimNodeSignals = [2*(2*degree + 1), 64], nFilterTaps = [3], bias = True, 
    nonlinearity  = nn.Tanh, dimReadout = [2], dimEdgeFeatures = 1):
    """
    Method returns a LocalGNN given the path for the training hyperparameters. No other information is needed
    as it defaults to the hyperparameters of FlockingGNN.py.
    """
    
    localGNN = architTime.LocalGNN_DB(dimNodeSignals, nFilterTaps, bias, nonlinearity, dimReadout, dimEdgeFeatures)
    localGNN.load_state_dict(state_dic_location)

    return localGNN


hyperparameters_path = ('/home/jcervino/summer-research/constrained-RL/experiments/flockingGNN-015-20210727171842/savedModels/LocalGNNArchitBest.ckpt') # Path to the hyperparameters
localGNN = createTrainedLocalGNN(torch.load(hyperparameters_path, map_location=torch.device('cpu'))) # Model object


n_agents = 15
min_dist = 0.2
n_train = 1
n_valid = 1
n_test = 1


capt = CAPT.CAPT(n_agents, min_dist, n_train, n_valid, n_test, t_f = 3, degree=degree)

sample = 0
pos, vel, accel = capt.pos_all, capt.vel_all, capt.accel_all

print(capt.evaluate(pos, capt.G_all))

pos, vel_gnn, accel_gnn = capt.simulated_trajectory(pos[:,0,:,:], archit=localGNN)

for t in range(0, pos.shape[1]):
    plt.scatter(pos[sample, t, :, 0], 
                pos[sample, t, :, 1], 
                marker='.', 
                color='black',
                label='')

plt.scatter(capt.G_all[sample, :, 0], capt.G_all[sample, :, 1], 
                label="goal", marker='x', color='r')

plt.scatter(pos[sample, 0, :, 0], 
            pos[sample, 0, :, 1], 
            marker='o', 
            color='red',
            label='start')

state = capt.state_all[0]
pos = capt.pos_all[0]
goals = capt.G_all[0]
accel = capt.accel_all[0]

plt.grid()    
plt.title('Trajectories')
plt.legend()
plt.show()



