import CAPT
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import alegnn.modules.architecturesTime as architTime
import alegnn.utils.miscTools as miscTools

#miscTools.loadSeed('/home/jcervino/summer-research/constrained-RL/experiments/flockingGNN-015-20210803102532')
degree = 5

def createTrainedLocalGNN(state_dic_location, dimNodeSignals = [2*(3 * degree + 2)+1, 64], nFilterTaps = [3], bias = True, 
    nonlinearity  = nn.Tanh, dimReadout = [2], dimEdgeFeatures = 1):
    """
    Method returns a LocalGNN given the path for the training hyperparameters. No other information is needed
    as it defaults to the hyperparameters of FlockingGNN.py.
    """
    
    localGNN = architTime.LocalGNN_DB(dimNodeSignals, nFilterTaps, bias, nonlinearity, dimReadout, dimEdgeFeatures)
    localGNN.load_state_dict(state_dic_location)

    return localGNN


hyperparameters_path = ('/home/jcervino/summer-research/constrainedRL/experiments/constrainedRL/savedModels/localGNNArchitLast.ckpt') # Path to the hyperparameters
localGNN = createTrainedLocalGNN(torch.load(hyperparameters_path, map_location=torch.device('cpu'))) # Model object


n_agents = 12
min_dist = 0.1
n_train = 1
n_valid = 1
n_test = 1
duration = 10


capt = CAPT.CAPT(n_agents, min_dist, n_train, n_valid, n_test, max_accel = 0.5, t_f = duration, degree=degree)

sample = 0
pos, vel, accel = capt.pos_all, capt.vel_all, capt.accel_all


pos_gnn, vel_gnn, accel_gnn, collision_gnn = capt.simulated_trajectory(pos[:,0,:,:], archit=localGNN)

print(capt.state_all[0,:,-1,].shape)

fig, axs = plt.subplots(2, figsize=(12, 10))

labels_agents = []
for i in range(1, n_agents + 1):
    labels_agents.append('Agent ' + str(i))

agents_object = axs[0].plot(np.arange(0, duration, 0.1), capt.state_all[0,:,-1,:])
# axs[0].plot(np.arange(0, duration, 0.1), capt.state_all[0,:,-1,1], label='Agent 2')
# axs[0].plot(np.arange(0, duration, 0.1), capt.state_all[0,:,-1,2], label='Agent 3')
axs[0].legend(agents_object, labels_agents)
axs[0].set_ylabel('$\lambda$')
axs[0].set_xlabel('t (sec)')
axs[0].set_title('Dual variable evolution')
axs[0].grid()    

print(capt.state_all[:,:,-1,:])

""" for t in range(0, pos.shape[1]):
    if t == pos_gnn.shape[1] - 1:
        plt.scatter(pos[sample, t, :, 0], 
                    pos[sample, t, :, 1], 
                    marker='.', 
                    color='gray',
                    linewidths=0.01, 
                    alpha=0.5,
                    label="CAPT trajectory")
    else:
        plt.scatter(pos[sample, t, :, 0], 
                    pos[sample, t, :, 1], 
                    marker='.', 
                    color='gray',
                    label='', linewidths=0.01, alpha=0.5) """

for t in range(0, pos_gnn.shape[1]):
    if t == pos_gnn.shape[1] - 1:
        axs[1].scatter(pos_gnn[sample, t, :, 0], 
                    pos_gnn[sample, t, :, 1], 
                    marker='.', 
                    color='black',
                    label='GNN trajectory')
    else: 
        axs[1].scatter(pos_gnn[sample, t, :, 0], 
                    pos_gnn[sample, t, :, 1], 
                    marker='.', 
                    color='black',
                    label='')

axs[1].scatter(capt.G_all[sample, :, 0], capt.G_all[sample, :, 1], 
                label="goal", marker='x', color='blue')


axs[1].set_aspect('equal', adjustable='datalim')

for goal in range(0, n_agents):
    cir = plt.Circle((capt.G_all[sample, goal, 0], capt.G_all[sample, goal, 1]), capt.R, color='blue',fill=True, alpha=0.25)
    axs[1].add_patch(cir)




agent_start = axs[1].scatter(pos[sample, 0, :, 0], 
                             pos[sample, 0, :, 1], 
                             marker='o', 
                             color='red')


for i, agent in enumerate(labels_agents):
    axs[1].annotate(str(i + 1), (pos[sample, 0, i, 0] + 0.05, pos[sample, 0, i, 1] + 0.05))

state = capt.state_all[0]
pos = capt.pos_all[0]
goals = capt.G_all[0]
accel = capt.accel_all
#pos_gnn = np.transpose(pos_gnn, (0, 1, 3, 2))
#capt.saveVideo('/home/jcervino/summer-research/constrained-RL/videos/test', pos_gnn, doPrint=True)


axs[1].grid()    
axs[1].set_title('Trajectories')
plt.show()


