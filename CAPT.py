import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import _needs_add_docstring
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import timeit
from sklearn.neighbors import NearestNeighbors
import torch
import pickle
import os
import sys
from matplotlib.animation import FFMpegWriter

zeroTolerance = 1e-7

def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), expandDims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    
    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.
    
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None
        
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y
    
    def expandDims(self):
        
        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 2)
        
    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired dataType to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

class CAPT(_data):
    """
    A wrapper class to execute the CAPT algorithm by Matthew Turpin
    (https://journals.sagepub.com/doi/10.1177/0278364913515307). Certain 
    parts this code (.compute_agents_initial_positions) are originally 
    from the Alelab GNN library (https://github.com/alelab-upenn).
    ...
    
    Attributes
    ----------
    n_agents : int
        The total number of agents that will take part in the simulation
    min_dist : double
        The minimum distanc between agents
    n_samples : int
        The total number of samples.
    max_vel = double
        Maximum velocity allowed
    t_f = double
        Simulation time
    max_accel : double
        Maximum acceleration allowed
    degree : int
        Number of edges (connections) per node
    """

    def __init__(self, n_agents, min_dist,
                 nTrain, nValid, nTest,
                 max_vel = None, t_f=None, max_accel = 5, degree = 5):

        super().__init__()
        
        self.zeroTolerance = 1e-7 # all values less then this are zero
        self.n_agents = n_agents # number of agents
        self.min_dist = min_dist # minimum initial distance between agents 
        self.n_goals = n_agents # number of goals (same as n_agents by design)
        self.max_accel = max_accel # max allowed acceleration
        self.degree = degree # number of edges for each node (agent)

        # Dataset information
        self.nTrain, self.nValid, self.nTest =  nTrain, nValid, nTest
        self.n_samples = nTrain + nValid + nTest # number of samples
        self.dataType = np.float64
        self.R = 0.2
        
        self.eta = 1
        self.delta = 5

        
        # Max allowed velocity
        if (max_vel is None):
            self.max_vel = 10
        else:
            self.max_vel = max_vel
        
        # Simulation duration
        if (t_f is None):
            self.t_f = 10 / self.max_vel
        else:
            self.t_f = t_f
            
        self.comm_radius = 6 # Legacy code
            
        # Time samples per sample (where 0.1 is the sampling time)    
        self.t_samples = int(self.t_f / 0.1)
        
        # Defining initial positions for agents
        self.X_0_all = self.compute_agents_initial_positions(self.n_agents, 
                                                       self.n_samples, 
                                                       self.comm_radius,
                                                       min_dist = self.min_dist)
        
        # Defining initial positions for goals
        self.G_all = self.compute_goals_initial_positions(self.X_0_all, self.min_dist)
        
        # Compute assignments for agents-goals (using Hungarian Algorithm)
        self.phi = self.compute_assignment_matrix(self.X_0_all, self.G_all)
        
        # Compute complete trajectories (iterated CAPT algorithm)
        self.pos_all, self.vel_all, self.accel_all = self.simulated_trajectory(self.X_0_all)
        
        # Compute communication graphs for the simulated trajectories
        self.comm_graph_all = self.compute_communication_graph(self.pos_all,
                                                           self.degree)

        # Compute the states for the entire dataset
        self.state_all, _ = self.compute_state(self.pos_all, self.G_all, self.vel_all, self.comm_graph_all, self.degree)
 
        # Separate the states into training, validation and testing samples
        # and save them

        # Create the dictionaries
        self.initPos = {}
        self.pos = {}
        self.vel = {}
        self.accel = {}
        self.commGraph = {}
        self.state = {}
        self.goals = {}

        #   Training set
        self.samples['train']['signals'] = self.state_all[0:self.nTrain].copy()
        self.samples['train']['targets'] = np.transpose(self.accel_all[0:self.nTrain].copy(), (0, 1, 3, 2))
        self.initPos['train'] = self.X_0_all[0:self.nTrain]
        self.pos['train'] = self.pos_all[0:self.nTrain]
        self.vel['train'] = self.vel_all[0:self.nTrain]
        self.accel['train'] = self.accel_all[0:self.nTrain]
        self.commGraph['train'] = self.comm_graph_all[0:self.nTrain]
        self.state['train'] = self.state_all[0:self.nTrain]
        self.goals['train'] = self.G_all[0:self.nTrain]


        #   Validation set
        startSample = self.nTrain
        endSample = self.nTrain + self.nValid
        self.samples['valid']['signals'] = self.state_all[startSample:endSample].copy()
        self.samples['valid']['targets'] = np.transpose(self.accel_all[startSample:endSample].copy(), (0, 1, 3, 2))
        self.initPos['valid'] = self.X_0_all[startSample:endSample]
        self.pos['valid'] = self.pos_all[startSample:endSample]
        self.vel['valid'] = self.vel_all[startSample:endSample]
        self.accel['valid'] = self.accel_all[startSample:endSample]
        self.commGraph['valid'] = self.comm_graph_all[startSample:endSample]
        self.state['valid'] = self.state_all[startSample:endSample]
        self.goals['valid'] = self.G_all[startSample:endSample]

        #   Testing set
        startSample = self.nTrain + self.nValid
        endSample = self.nTrain + self.nValid + self.nTest
        self.samples['test']['signals'] = self.state_all[startSample:endSample].copy()
        self.samples['test']['targets'] = np.transpose(self.accel_all[startSample:endSample].copy(), (0, 1, 3, 2))
        self.initPos['test'] = self.X_0_all[startSample:endSample]
        self.pos['test'] = self.pos_all[startSample:endSample]
        self.vel['test'] = self.vel_all[startSample:endSample]
        self.accel['test'] = self.accel_all[startSample:endSample]
        self.commGraph['test'] = self.comm_graph_all[startSample:endSample]
        self.state['test'] = self.state_all[startSample:endSample]
        self.goals['test'] = self.G_all[startSample:endSample]

        # Change data to specified type and device
        self.astype(torch.float64)
        self.to(self.device)
        
    def astype(self, dataType):
        
        # Change all other signals to the correct place
        datasetType = ['train', 'valid', 'test']
        for key in datasetType:
            self.initPos[key] = changeDataType(self.initPos[key], dataType)
            self.pos[key] = changeDataType(self.pos[key], dataType)
            self.vel[key] = changeDataType(self.vel[key], dataType)
            self.accel[key] = changeDataType(self.accel[key], dataType)
            self.commGraph[key] = changeDataType(self.commGraph[key], dataType)
            self.state[key] = changeDataType(self.state[key], dataType)
        
        # And call the parent
        super().astype(dataType)
        
    def to(self, device):
        
        # Check the data is actually torch
        if 'torch' in repr(self.dataType):
            datasetType = ['train', 'valid', 'test']
            # Move the data
            for key in datasetType:
                self.initPos[key].to(device)
                self.pos[key].to(device)
                self.vel[key].to(device)
                self.accel[key].to(device)
                self.commGraph[key].to(device)
                self.state[key].to(device)
            
            super().to(device)
        
    def compute_agents_initial_positions(self, n_agents, n_samples, comm_radius,
                                        min_dist = 0.1, doPrint= True, **kwargs):
        """ 
        Generates a NumPy array with the 
        initial x, y position for each of the n_agents
    
        Parameters
        ----------
        n_agents : int
            The total number of agents that will take part in the simulation
        n_samples : int
            The total number of samples.
        comm_radius : double (legacy code)
            The communication radius between agents (determines initial spacing between agents) 
        min_dist : double
            The minimum distance between each agent
    
        Returns
        -------
        np.array (n_samples x n_agents x 2) 
        """
        
        if (doPrint):
            print('\tComputing initial positions matrix...', end = ' ', flush = True)
        
        assert min_dist * (1.+self.zeroTolerance) <= comm_radius * (1.-self.zeroTolerance)
        
        min_dist = min_dist * (1. + self.zeroTolerance)
        comm_radius = comm_radius * (1. - self.zeroTolerance)
     
            
        # This is the fixed distance between points in the grid
        distFixed = (comm_radius + min_dist)/(2.*np.sqrt(2))
        
        # This is the standard deviation of a uniform perturbation around
        # the fixed point.
        distPerturb = (comm_radius - min_dist)/(4.*np.sqrt(2))
        
        # How many agents per axis
        n_agentsPerAxis = int(np.ceil(np.sqrt(n_agents)))
        
        axisFixedPos = np.arange(-(n_agentsPerAxis * distFixed)/2,
                                    (n_agentsPerAxis * distFixed)/2,
                                    step = distFixed)
        
        # Repeat the positions in the same order (x coordinate)
        xFixedPos = np.tile(axisFixedPos, n_agentsPerAxis)
    
        # Repeat each element (y coordinate)
        yFixedPos = np.repeat(axisFixedPos, n_agentsPerAxis)
        
        # Concatenate this to obtain the positions
        fixedPos = np.concatenate((np.expand_dims(xFixedPos, 0),
                                    np.expand_dims(yFixedPos, 0)),
                                    axis = 0)
        
        # Get rid of unnecessary agents
        fixedPos = fixedPos[:, 0:n_agents]
        
        # Adjust to correct shape
        fixedPos = fixedPos.T
    
        # And repeat for the number of samples we want to generate
        fixedPos = np.repeat(np.expand_dims(fixedPos, 0), n_samples,
                                axis = 0)
        
        # Now generate the noise
        perturbPos = np.random.uniform(low = -distPerturb,
                                        high = distPerturb,
                                        size = (n_samples, n_agents,  2))
        # Initial positions
        initPos = fixedPos + perturbPos
        
        if doPrint:
            print("OK", flush = True)
              
        return initPos
    
    def compute_goals_initial_positions(self, X_0, min_dist):
        """ 
        Generates a NumPy array with the 
        initial x, y position for each of the n_goals
        
        Parameters
        ----------
        X_0 : np.array (n_samples x n_agents x 2) 
            Initial positions of the agents for all samples
        min_dist : double (legacy)
            The minimum distance between each agent
        
        Returns
        -------
        np.array (n_samples x n_goals x 2) 
        """

        
        n_samples = X_0.shape[0]
        n_goals = X_0.shape[1]

        goal_position = np.zeros((n_samples, n_goals, 2))


        for sample in range(0, n_samples):
            for goal in range(0, n_goals):
                x_0 = X_0[sample, goal, 0]
                y_0 = X_0[sample, goal, 1]
                radius = np.random.uniform(1, 1.5)
                phi = np.random.uniform(0, 2*np.math.pi)
                goal_position[sample, goal] = np.array([radius * np.math.cos(phi) + x_0, radius * np.math.sin(phi) + y_0])


        
        # # Find max/min positions
        # x_min = np.min(X_0[0, :, 0])
        # y_min = np.min(X_0[0, :, 1])
        # x_max = np.max(X_0[0, :, 0])
        # y_max = np.max(X_0[0, :, 1])
      
        # # Samples uniform distribution
        # x = np.random.uniform(low = x_min, high = x_max, size=n_goals)
        # y = np.random.uniform(low = y_min, high = y_max, size=n_goals)
      
        
        # # Creates goals array
        # goals = np.stack((x, y), axis=1)  
        # goals = np.repeat(np.expand_dims(goals, 0), n_samples, axis = 0)
        
        # dist_pertub = (min_dist)/(4.*np.sqrt(2))
        
        # # Now generate the noise
        # pertubation = np.random.uniform(low = -dist_pertub,
        #                                 high = dist_pertub,
        #                                 size = (n_samples, n_goals,  2))
        
        # goals = goals + pertubation
      
        return goal_position
    
    def compute_assignment_matrix(self, X_0, G, doPrint = True):
        """ 
        Computes assignment matrix using the Hungarian Algorithm
        
        Parameters
        ----------
        X_0 : np.array (n_samples x n_agents x 2) 
            Initial positions of the agents for all samples
        G : np.array (n_samples x n_agents x 2) 
            goal positions of the agents for all samples
 
        Returns
        -------
        np.array (n_samples x n_agents x n_goals)
        """
        
        n_samples = X_0.shape[0]
        n_agents = X_0.shape[1]

        phi = np.zeros((n_samples, n_agents, n_agents))

        if (doPrint):
            print('\tComputing assignment matrix...', end = ' ', flush = True)
        
        for sample in range(0, n_samples):
            # Obtains the initial posiition arrays
            agents = X_0[sample,:,:]
            goals = G[sample,:,:]
            
            # Calculates distance matrix
            distance_matrix = cdist(agents, goals)
          
            # Obtains optimal linear combination
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
          
            # Obtains assignment matrix (binary)
            phi[sample, row_ind, col_ind] = 1
        
            if (doPrint):
                percentageCount = int(100 * sample + 1) / n_samples
                if sample == 0:
                    # It's the first one, so just print it
                    print("%3d%%" % percentageCount,
                          end = '', flush = True)
                else:
                    # Erase the previous characters
                    print('\b \b' * 4 + "%3d%%" % percentageCount,
                          end = '', flush = True)
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            print("OK", flush = True)

        return phi
    
    def get_beta(self, t_0, t):
        """ 
        Computes the polynomial function of time Beta as described in
        the CAPT paper.
        
        Parameters
        ----------
        t : double
            time index that we define as the starting point
        t : double
            time index such that we obtain β(t) 
        
        Returns
        -------
        double (β(t))
        """
        
        t_f = self.t_f
        
        alpha_0 = -t_0 / (t_f - t_0)
        alpha_1 = 1 / (t_f - t_0)
        
        return (alpha_0 * 1 + alpha_1 * t)

    def compute_trajectory(self, X, G, sample, t, t_0 = 0):
        """ 
        Computes the matrix X(t) (agent location) for the input t
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        G : np.array (n_samples x n_agents x 2) 
            goal positions of the agents for all samples
        t : int
            time integer index such that we obtain X(t). Note that this is an integer which is then converted.
        t_0 : double
            starting time index (i.e. the reference position to obtain X(t_0)),
            we set as default 0.0
        
        Returns
        -------
        np.array (n_agents x 2)
        """
        
        t_0 = int(t_0 * 0.1)
        t = t * 0.1
        
        beta = self.get_beta(t_0, t)
        phi = self.phi[sample,:,:]
        G = G[sample,:,:]
        
        # If the length is 4, we are passing in the entire trajectory; if it is
        # 3, we are only passing the first time step.
        if len(X.shape) == 4:
            X = X[sample,t_0*10 - 1, :,:]
        else:
            X = X[sample, :,:]
        
        N = self.n_agents
        I = np.eye(N)

        trajectory = (1 - beta) * X \
            + beta * (phi @ G + (I - phi @ phi.T) @ X)
        
        return trajectory
    
    def capt_trajectory(self, X = None, doPrint=True, t_0 = 0):
        """ 
        Computes the matrix X(t) (agent location) for all t such
        that t_0 <= t <= t_f and optionally plots it. It will use the CAPT
        algorithm with no modifications; as such, it might produce trajectories
        that require unfeasiable velocities/accelerations. It will, however,
        produce the right *direction* of the trajectories - this can be used
        later with other functions to generate a more realistic trajectory.
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        doPrint : boolean
            determines whether to print the progress or not
        t_0 : integer
            index that corresponds to the time that the trajectory starts
            
        
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x 2)
        """

        t_samples = int((self.t_f - t_0 * 0.1) / 0.1)
        
        complete_trajectory = np.zeros((self.n_samples, 
                                        t_samples, 
                                        self.n_agents, 
                                        2))

        if (X is None):
            X = self.X_0_all
        
        G = self.G_all
        
        if (doPrint):
            print('\tComputing CAPT trajectories...', end = ' ', flush = True)
        
        for sample in range(0, self.n_samples):
            for index in np.arange(0, t_samples):
                complete_trajectory[sample, index, :, :] = \
                    self.compute_trajectory(X, G, sample, index, t_0)
                 
            if (doPrint):
                percentageCount = int(100 * sample + 1) / self.n_samples
                if sample == 0:
                    # It's the first one, so just print it
                    print("%3d%%" % percentageCount,
                          end = '', flush = True)
                else:
                    # Erase the previous characters
                    print('\b \b' * 4 + "%3d%%" % percentageCount,
                          end = '', flush = True)
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            print("OK", flush = True)
            
        return complete_trajectory
    
    def compute_communication_graph(self, X, degree,
                                    doPrint = True):
        """ 
        Computes the communication graphs S for the entire position array at
        each time instant.
        
        Parameters
        ----------
         X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        degree : int
            number of edges for each node (agent)
        doPrint : boolean
            whether to print progress or not.
            
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x n_agents)
        
        """
        
        n_samples = X.shape[0]
        t_samples = X.shape[1]
        n_agents = X.shape[2]

        graphMatrix = np.zeros((n_samples, 
                                t_samples, 
                                n_agents, 
                                n_agents))
        
        if (doPrint):
            print('\tComputing communication graph...', end = ' ', flush = True)
        
        for sample in range(0, n_samples):
            for t in range(0, t_samples):
                neigh = NearestNeighbors(n_neighbors=degree)
                neigh.fit(X[sample, t, :, :])
                graphMatrix[sample, t, :, :] = np.array(neigh.kneighbors_graph(mode='connectivity').todense())    
        
            if (doPrint):
                percentageCount = int(100 * sample + 1) / self.n_samples
                if sample == 0:
                    # It's the first one, so just print it
                    print("%3d%%" % percentageCount,
                          end = '', flush = True)
                else:
                    # Erase the previous characters
                    print('\b \b' * 4 + "%3d%%" % percentageCount,
                          end = '', flush = True)
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            print("OK", flush = True)
            
        return graphMatrix
        
    
    def compute_velocity(self, X, doPrint = True, t_0 = 0):
        """ 
        Computes the matrix with the velocity (v_x, v_y) of each agent for all t such
        that t_0 <= t <= t_f.
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x 2)
        
        """
        complete_trajectory = self.capt_trajectory(X = X, doPrint=False, t_0 = t_0)
        
        # Calculate the difference at each step
        v_x = np.diff(complete_trajectory[:,:,:,0], axis=1) / 0.1
        v_y = np.diff(complete_trajectory[:,:,:,1], axis=1) / 0.1
        
        # Stack the vectors
        vel = np.stack((v_x, v_y), axis=-1)
        
        # Add velocity for t = 0
        v_0 = np.zeros((self.n_samples, 1, self.n_agents, 2))
        vel = np.concatenate((v_0, vel), axis=1)
        
        return vel
    
    def compute_acceleration(self, X, clip=True, t_0 = 0):
        """ 
        Computes the matrix with the acceleration (a_x, a_y) of each agent for 
        all t such that t_0 <= t <= t_f.
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        clip : boolean
            Determines wheter to limit the acceleration to the interval
            [-max_accel, max_accel]
        
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x 2)
        
        """
        complete_velocity = self.compute_velocity(X = X, t_0 = t_0)
        
        # Calculate the difference at each step
        a_x = np.diff(complete_velocity[:,:,:,0], axis=1) / 0.1
        a_y = np.diff(complete_velocity[:,:,:,1], axis=1) / 0.1
        
        # Stack the vectors
        accel = np.stack((a_x, a_y), axis=-1)
        
        # Add velocity for t = 0
        accel_0 = np.zeros((self.n_samples, 1, self.n_agents, 2))
        accel = np.concatenate((accel_0, accel), axis=1)
        
        if (clip):
            accel = np.clip(accel[:,:,:,:], -self.max_accel, self.max_accel)
        
        return accel
    
    def simulated_trajectory(self, X_0, doPrint = True, archit = None):
        """ 
        Calculates trajectory using the calculated acceleration. This function
        is particularly useful when clip is set to True in 
        .compute_acceleration() since it will generate trajectories that are
        physically feasible.
        
        Parameters
        ----------
        X_0 : np.array (n_samples x n_agents x 2) 
            Initial positions of the agents for all samples
        
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x 2)
        
        """
        
        n_samples = X_0.shape[0]
        t_samples = int(self.t_f / 0.1)
        n_agents = X_0.shape[1]
        max_accel = self.max_accel
        
       
        vel_all = np.zeros((n_samples, 
                        t_samples, 
                        n_agents, 
                        2))
        
        pos_all = np.zeros((n_samples, 
                        t_samples, 
                        n_agents, 
                        2))

        accel_all = np.zeros((n_samples, t_samples, n_agents, 2))
        collision_all = np.zeros(t_samples)

        
        pos_all[:, 0, :, :] = X_0

         # If there is no architecture, we use CAPT. Else, we use the GNN.
        if archit == None:
            k = 3
            x_f = self.capt_trajectory(doPrint=False)[:,-1,:,:] - X_0
            #accel = x_f / (0.1*k *(self.t_f - 0.1*k/2))

            accel = np.clip((4*x_f / self.t_f**2), -max_accel, max_accel)

            for t in range(0, int(t_samples / 2)):
                accel_all[:,t,:,:] = accel
                if not t == 0:
                    accel_all[:,-t,:,:] = -accel
                accel_all = np.clip(accel_all, -max_accel, max_accel)
            use_archit = False
        else:
            accel_all = np.zeros((n_samples, t_samples, n_agents, 2))
            graph_all = np.zeros((n_samples, t_samples, n_agents, n_agents))
            state_all = np.zeros((n_samples, t_samples, 2 *(3 * self.degree + 2) + 1, n_agents))
            use_archit = True
        
        if (doPrint):
            print('\tComputing simulated trajectories...', end = ' ', flush = True)
        
        for t in np.arange(1, t_samples):
            curr_pos = np.expand_dims(pos_all[:, t-1, :, :], 1)
            curr_vel = np.expand_dims(vel_all[:, t-1, :, :], 1)
                
            curr_comm_graph = self.compute_communication_graph(curr_pos, self.degree, doPrint=False)
            curr_state, collision_num = self.compute_state(curr_pos, self.G_all, curr_vel, commGraph=curr_comm_graph, degree=self.degree, doPrint=False)
            graph_all[:, t-1, :, :] = curr_comm_graph.squeeze(1)
            state_all[:, t-1, :, :] = curr_state.squeeze(1)
            collision_all[t-1] = collision_num

            x = torch.tensor(state_all[:, 0:t, :, :])
            S = torch.tensor(graph_all[:, 0:t, :, :]) 

            with torch.no_grad():
                new_accel = archit(x, S)
                new_accel = new_accel.numpy()
                new_accel = np.transpose(new_accel, (0, 1, 3, 2))
            
            accel_all[:, t-1, :, :] = new_accel[:, -1, :, :]
                
            vel_all[:, t, :, :] = vel_all[:, t - 1, :, :] \
                        + accel_all[:, t-1, :, :] * 0.1 
                        
            pos_all[:, t, :, :] = pos_all[:, t - 1, :, :] \
                + vel_all[:, t - 1, :, :] * 0.1 \
                + accel_all[:, t - 1, :, :] * 0.1**2 / 2
            
        # Print
        if doPrint:
            # Erase the percentage
            print("OK", flush = True)
            
        return pos_all, vel_all, accel_all, collision_all
    
    def compute_state(self, X, G, V, commGraph, degree, doPrint = True):
        """ 
        Computes the states for all agents at all t_samples and all n_samples.
        The state is a matrix with contents [X_agent, V_agent, X_closest, V_closest, G_closest],
        where X_agent is the position of the agent itself, V_agent is the position of the agent itself,
        X_closest is the position of the n_degree closest agents, V_Closest is the position of the n_degree
        closest agents and G_closest is the position of the n_degree closest goals. 
        Each state, therefore, has 2(3*self.degree + 2) elements
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        G : np.array (n_samples x n_agents x 2) 
            goal positions of the agents for all samples
        commGraph : np.array (n_samples x t_samples x n_agents x n_agents)
            communication graph (adjacency matrix)
        degree : int
            number of edges allowed per node
        
        Returns
        -------
        np.array (n_samples x t_samples x (2 * (3*self.degree + 2)) x n_agents)
        
        """
        
        n_samples = X.shape[0]
        t_samples = X.shape[1]
        n_agents = X.shape[2]

        if (doPrint):
            print('\tComputing states...', end = ' ', flush = True)
        
        d = 3 * degree + 2
        state = np.zeros((n_samples, t_samples, d * 2 + 1, n_agents))
        
        # Finding closest goals
        for sample in range(0, n_samples):
                for t in range(0, t_samples):
                    agents = X[sample, t, :,:]
                    goals = G[sample, :,:]
                    
                    # Calculates distance matrix
                    distance_matrix = cdist(agents, goals)
                    
                    # Lambda update
                    agent_distance = cdist(agents, agents) # Minimum distance between agents
                    neighboorhood_distance = agent_distance + np.eye(self.n_agents) * sys.float_info.max # Removes zeros from itself
                    min_dist = np.min(neighboorhood_distance, axis=1) # Minimum distance to neighboors
                    min_dist = np.min(neighboorhood_distance, axis=1) # Minimum distance to neighboors
                    collision_num = np.sum(neighboorhood_distance <= self.collision_dist)
                    
                    for agent in range(0, n_agents):
                        
                        # Absolute
                        # Own positions, velocity 
                        # state[sample, t, 0:2, agent] = X[sample, t, agent,:].flatten()
                        # state[sample, t, 2:4, agent] = V[sample, t, agent,:].flatten()
                        
                        # # Other agents position, velocity
                        # closest_agents_index = commGraph[sample, t, agent, :] == 1
                        # state[sample, t, 4:(degree+2)*2, agent] = X[sample, t, closest_agents_index].flatten()
                        # state[sample, t, (degree+2)*2:(2*degree + 2)*2, agent] = V[sample, t, closest_agents_index].flatten()

                        # # Goals
                        # distance_to_goals = distance_matrix[agent, :]
                        # closest_goals_index = np.argsort(distance_to_goals)[0:degree]
                        # state[sample, t, -degree * 2:, agent] = goals[closest_goals_index].flatten()

                        # Relative
                        #Own positions, velocity 
                        state[sample, t, 0:2, agent] = X[sample, t, agent,:].flatten()
                        state[sample, t, 2:4, agent] = V[sample, t, agent,:].flatten()
                        
                        # Other agents position, velocity
                        closest_agents_index = commGraph[sample, t, agent, :] == 1
                        state[sample, t, 4:(degree+2)*2, agent] = (X[sample, t, closest_agents_index] - np.tile(state[sample, t, 0:2, agent], (self.degree, 1))).flatten()
                        state[sample, t, (degree+2)*2:(2*degree + 2)*2, agent] = (V[sample, t, closest_agents_index] - np.tile(state[sample, t, 2:4, agent], (self.degree, 1))).flatten() 

                        # Goals
                        distance_to_goals = distance_matrix[agent, :]
                        closest_goals_index = np.argsort(distance_to_goals)[0:degree]
                        state[sample, t, -degree * 2 - 1:-1, agent] = (goals[closest_goals_index] - np.tile(state[sample, t, 0:2, agent], (self.degree, 1))).flatten()
                        
                        
                        # Dual variable
                        state[sample, t, -1, agent] =  max(0, state[sample, t, -1, agent] - self.eta / self.t_samples * (min_dist[agent] - self.delta)) if t != 0 else np.random.uniform(0, 0.5)
    
                        
                        
                if (doPrint):
                    percentageCount = int(100 * sample + 1) / n_samples
                    if sample == 0:
                        # It's the first one, so just print it
                        print("%3d%%" % percentageCount,
                              end = '', flush = True)
                    else:
                        # Erase the previous characters
                        print('\b \b' * 4 + "%3d%%" % percentageCount,
                              end = '', flush = True)
        
        # Print
        if doPrint:
            # Erase the percentage
            print('\b \b' * 4, end = '', flush = True)
            print("OK", flush = True)
            
        return state, collision_num
    
    def evaluate(self, X, G):
        """ 
        Computes the total cost of the trajectory averaged over all samples. 
        The cost is associated with the number of goals with no agent located
        at distance less than R.
        
        Parameters
        ----------
        X : np.array (n_samples x t_samples, n_agents x 2) 
            positions of the agents for all samples for all times t
        G : np.array (n_samples x n_agents x 2) 
            goal positions of the agents for all samples
        R : double
            tolerance regarding goal-agent distance
        
        Returns
        -------
        double
        
        """

        R = self.R
        X = np.array(X)
        final_pos = X[:, -1, :, :]
        n_samples = X.shape[0]
        goals = G
        mean_cost = 0
        t_samples = X.shape[2]
        
        for sample in range(0, n_samples):
            # Calculate distance
            distance_matrix = cdist(final_pos[sample, :, :], goals[sample, :, :])
            
            # Find the closest agent distance
            distance_matrix = np.min(distance_matrix, axis=1)
            
            # Check which goals have no agents at distance R (or greater)
            distance_matrix = distance_matrix > R
            
            # Count the number of goals with no agents at distance R (or greater)
            curr_cost = np.sum(distance_matrix)
                                    
            # Running (iterative) average
            mean_cost = mean_cost + (1 / (sample + 1)) * (curr_cost - mean_cost)
            
        return mean_cost
    
    def getData(self, name, samplesType, *args):
        """ 
        Returns the specified data from the dataset
        
        Parameters
        ----------
        name : str
            data type (pos, vel, etc.)
        sampleType : str
            data category (train, valid, test)
        
        Returns
        -------
        np.array
        
        """
        
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
                    
        # Check that the name is actually an attribute
        assert name in dir(self)
        
        # Get the desired attribute
        thisDataDict = getattr(self, name)
        
        # Check it's a dictionary and that it has the corresponding key
        assert type(thisDataDict) is dict
        assert samplesType in thisDataDict.keys()
        
        # Get the data now
        thisData = thisDataDict[samplesType]
        # Get the dimension length
        thisDataDims = len(thisData.shape)
        
        # Check that it has at least two dimension, where the first one is
        # always the number of samples
        assert thisDataDims > 1
        
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = thisData.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                thisData = thisData[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                thisData = thisData[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(thisData.shape) < thisDataDims:
                if 'torch' in repr(thisData.dtype):
                    thisData =thisData.unsqueeze(0)
                else:
                    thisData = np.expand_dims(thisData, axis = 0)

        return thisData

    def saveVideo(self, saveDir, pos, *args, 
                  commGraph = None, **kwargs):
        
        # Check that pos is a position of shape nSamples x tSamples x 2 x nAgents
        assert len(pos.shape) == 4
        nSamples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        if 'torch' in repr(pos.dtype):
            pos = pos.cpu().numpy()
        
        # Check if there's the need to plot a graph
        if commGraph is not None:
            # If there's a communication graph, then it has to have shape
            #   nSamples x tSamples x nAgents x nAgents
            assert len(commGraph.shape) == 4
            assert commGraph.shape[0] == nSamples
            assert commGraph.shape[1] == tSamples
            assert commGraph.shape[2] == commGraph.shape[3] == nAgents
            if 'torch' in repr(commGraph.dtype):
                commGraph = commGraph.cpu().numpy()
            showGraph = True
        else:
            showGraph = False
        
        if 'doPrint' in kwargs.keys():
            doPrint = kwargs['doPrint']
        else:
            doPrint = self.doPrint
            
        # This number determines how faster or slower to reproduce the video
        if 'videoSpeed' in kwargs.keys():
            videoSpeed = kwargs['videoSpeed']
        else:
            videoSpeed = 1.
            
        if 'showVideoSpeed' in kwargs.keys():
            showVideoSpeed = kwargs['showVideoSpeed']
        else:
            if videoSpeed != 1:
                showVideoSpeed = True
            else:
                showVideoSpeed = False    
                
        if 'vel' in kwargs.keys():
            vel = kwargs['vel']
            if 'showCost' in kwargs.keys():
                showCost = kwargs['showCost']
            else:
                showCost = True
            if 'showArrows' in kwargs.keys():
                showArrows = kwargs['showArrows']
            else:
                showArrows = True
        else:
            showCost = False
            showArrows = False
        
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                selectedIndices = args[0]
                
            # Select the corresponding samples
            pos = pos[selectedIndices]
                
            # Finally, observe that if pos has shape only 3, then that's 
            # because we selected a single sample, so we need to add the extra
            # dimension back again
            if len(pos.shape) < 4:
                pos = np.expand_dims(pos, 0)
                
            if showGraph:
                commGraph = commGraph[selectedIndices]
                if len(commGraph.shape)< 4:
                    commGraph = np.expand_dims(commGraph, 0)
        
        # Where to save the video
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
            
        videoName = 'sampleTrajectory'
        
        xMinMap = np.min(pos[:,:,0,:]) * 1.2
        xMaxMap = np.max(pos[:,:,0,:]) * 1.2
        yMinMap = np.min(pos[:,:,1,:]) * 1.2
        yMaxMap = np.max(pos[:,:,1,:]) * 1.2
        
        # Create video object
        
        videoMetadata = dict(title = 'Sample Trajectory', artist = 'Flocking',
                             comment='Flocking example')
        videoWriter = FFMpegWriter(fps = videoSpeed/0.1,
                                   metadata = videoMetadata)
        
        if doPrint:
            print("\tSaving video(s)...", end = ' ', flush = True)
        
        # For each sample now
        for n in range(pos.shape[0]):
            
            # If there's more than one video to create, enumerate them
            if pos.shape[0] > 1:
                thisVideoName = videoName + '%03d.mp4' % n
            else:
                thisVideoName = videoName + '.mp4'
            
            # Select the corresponding position trajectory
            thisPos = pos[n]
            thisGoal = self.G_all[n]
            
            # Create figure
            videoFig = plt.figure(figsize = (5,5))
            
            # Set limits
            plt.xlim((xMinMap, xMaxMap))
            plt.ylim((yMinMap, yMaxMap))
            plt.axis('equal')
            
            if showVideoSpeed:
                plt.text(xMinMap, yMinMap, r'Speed: $%.2f$' % videoSpeed)
                
            # Create plot handle
            plotAgents, = plt.plot([], [], 
                                   marker = 'o',
                                   markersize = 3,
                                   linewidth = 0,
                                   color = '#01256E',
                                   scalex = False,
                                   scaley = False)
            plotGoals, = plt.plot([], [], 
                                   marker = 'o',
                                   markersize = 3,
                                   linewidth = 0,
                                   color = 'red',
                                   scalex = False,
                                   scaley = False)
            
            # Create the video
            with videoWriter.saving(videoFig,
                                    os.path.join(saveDir,thisVideoName),
                                    dpi=300):
                
                for t in range(tSamples):
                        
                    # Plot the agents
                    plotAgents.set_data(thisPos[t,0,:], thisPos[t,1,:])
                    plotGoals.set_data(thisGoal[:,0], thisGoal[:,1])
                    videoWriter.grab_frame()
                    
                    # Print
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(
                                 100*(t+1+n*tSamples)/(tSamples * pos.shape[0])
                                              )
                        
                        if n == 0 and t == 0:
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
        
            plt.close(fig=videoFig)
            
        # Print
        if doPrint:
            # Erase the percentage and the label
            print('\b \b' * 4 + "OK", flush = True)
            
        if showGraph:
            
            # Normalize velocity
            if showArrows:
                # vel is of shape nSamples x tSamples x 2 x nAgents
                velNormSq = np.sum(vel ** 2, axis = 2)
                #   nSamples x tSamples x nAgents
                maxVelNormSq = np.max(np.max(velNormSq, axis = 2), axis = 1)
                #   nSamples
                maxVelNormSq = maxVelNormSq.reshape((nSamples, 1, 1, 1))
                #   nSamples x 1 x 1 x 1
                normVel = 2*vel/np.sqrt(maxVelNormSq)
            
            if doPrint:
                print("\tSaving graph snapshots...", end = ' ', flush = True)
            
            # Essentially, we will print nGraphs snapshots and save them
            # as images with the graph. This is the best we can do in a
            # reasonable processing time (adding the graph to the video takes
            # forever).
            time = np.arange(0, self.duration, step = self.samplingTime)
            assert len(time) == tSamples
            
            nSnapshots = 5 # The number of snapshots we will consider
            tSnapshots = np.linspace(0, tSamples-1, num = nSnapshots)
            #   This gives us nSnapshots equally spaced in time. Now, we need
            #   to be sure these are integers
            tSnapshots = np.unique(tSnapshots.astype(np.int)).astype(np.int)
            
            # Directory to save the snapshots
            snapshotDir = os.path.join(saveDir,'graphSnapshots')
            # Base name of the snapshots
            snapshotName = 'graphSnapshot'
            
            for n in range(pos.shape[0]):
                
                if pos.shape[0] > 1:
                    thisSnapshotDir = snapshotDir + '%03d' % n
                    thisSnapshotName = snapshotName + '%03d' % n
                else:
                    thisSnapshotDir = snapshotDir
                    thisSnapshotName = snapshotName
                    
                if not os.path.exists(thisSnapshotDir):
                    os.mkdir(thisSnapshotDir)
                
                # Get the corresponding positions
                thisPos = pos[n]
                thisCommGraph = commGraph[n]
                
                for t in tSnapshots:
                    
                    # Get the edge pairs
                    #   Get the graph for this time instant
                    thisCommGraphTime = thisCommGraph[t]
                    #   Check if it is symmetric
                    isSymmetric = np.allclose(thisCommGraphTime,
                                              thisCommGraphTime.T)
                    if isSymmetric:
                        #   Use only half of the matrix
                        thisCommGraphTime = np.triu(thisCommGraphTime)
                    
                    #   Find the position of all edges
                    outEdge, inEdge = np.nonzero(np.abs(thisCommGraphTime) \
                                                               > zeroTolerance)
                    
                    # Create the figure
                    thisGraphSnapshotFig = plt.figure(figsize = (5,5))
                    
                    # Set limits (to be the same as the video)
                    plt.xlim((xMinMap, xMaxMap))
                    plt.ylim((yMinMap, yMaxMap))
                    plt.axis('equal')
                    
                    # Plot the edges
                    plt.plot([thisPos[t,0,outEdge], thisPos[t,0,inEdge]],
                             [thisPos[t,1,outEdge], thisPos[t,1,inEdge]],
                             color = '#A8AAAF', linewidth = 0.75,
                             scalex = False, scaley = False)
                    
                    # Plot the arrows
                    if showArrows:
                        for i in range(nAgents):
                            plt.arrow(thisPos[t,0,i], thisPos[t,1,i],
                                      normVel[n,t,0,i], normVel[n,t,1,i])
                
                    # Plot the nodes
                    plt.plot(thisPos[t,0,:], thisPos[t,1,:],
                             marker = 'o', markersize = 3, linewidth = 0,
                             color = '#01256E', scalex = False, scaley = False)
                    
                    # Add the cost value
                    if showCost:
                        totalCost = self.evaluate(vel = vel[:,t:t+1,:,:])
                        plt.text(xMinMap,yMinMap, r'Cost: $%.4f$' % totalCost)
                    
                    # Add title
                    plt.title("Time $t=%.4f$s" % time[t])
                    
                    # Save figure
                    thisGraphSnapshotFig.savefig(os.path.join(thisSnapshotDir,
                                            thisSnapshotName + '%03d.pdf' % t))
                    
                    # Close figure
                    plt.close(fig = thisGraphSnapshotFig)
                    
                    # Print percentage completion
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(
                                 100*(t+1+n*tSamples)/(tSamples * pos.shape[0])
                                              )
                        if n == 0 and t == 0:
                            # Print new value
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                        
                        
                
            # Print
            if doPrint:
                # Erase the percentage and the label
                print('\b \b' * 4 + "OK", flush = True)



##########
# Driver #
##########

if __name__ == "__main__":
    capt = CAPT(n_agents = 10,
                min_dist = 0.5, 
                nTrain=200,
                nTest=200,
                nValid=30,
                t_f = 3, 
                max_accel = 10,
                degree = 3,)

    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(capt, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # Plotting (uncomment to visualize trajectory)

    sample = 5
    pos, vel, accel = capt.pos_all, capt.vel_all, capt.accel_all

    for t in range(0, pos.shape[1]):
        plt.scatter(pos[sample, t, :, 0], 
                    pos[sample, t, :, 1], 
                    marker='.', 
                    color='gray',
                    label='')

    plt.scatter(capt.G_all[sample, :, 0], capt.G_all[sample, :, 1], 
                    label="goal", marker='x', color='r')

    plt.scatter(pos[sample, 0, :, 0], 
                pos[sample, 0, :, 1], 
                marker='o', 
                color='red',
                label='start')

    # state = capt.state_all[0]
    # pos = capt.pos_all[0]
    # goals = capt.G_all[0]
    # accel = capt.accel_all[0]

    # accel_agent = accel[:, 3, 0]

    # plt.plot(np.arange(0, len(vel[0, :, 3, 0])), vel[0, :, 3, 0])

    plt.grid()    
    plt.title('Trajectories')
    plt.legend()
    #plt.show()

    pos = np.transpose(pos, (0, 1, 3, 2))
    capt.saveVideo('/home/jcervino/summer-research/constrained-RL/videos', pos[0:3], doPrint=True)

