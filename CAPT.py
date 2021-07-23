import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import utils
import timeit
from sklearn.neighbors import NearestNeighbors
import torch
import pickle

zeroTolerance = utils.zeroTolerance

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
        self.R = 0.25

        
        # Max allowed velocity
        if (max_vel is None):
            self.max_vel = 10
        else:
            self.max_vel = max_vel
        
        # Simulation duration
        if (t_f is None):
            self.t_f = 10 / max_vel
        else:
            self.t_f = t_f
            
        # Time samples per sample (where 0.1 is the sampling time)    
        self.t_samples = int(self.t_f / 0.1)

        #start = timeit.default_timer()

        
        # Defining initial positions for agents
        self.X_0_all = self.compute_agents_initial_positions(self.n_agents, 
                                                       self.n_samples, 
                                                       6,
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
        self.state_all = self.compute_state(self.pos_all, self.G_all, self.comm_graph_all, self.degree)

 
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
    

        stop = timeit.default_timer()
        #print('Total time: ', stop - start, 's')

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
        
        # Find max/min positions
        x_min = np.min(X_0[0, :, 0]) - 5
        y_min = np.min(X_0[0, :, 1]) - 5
        x_max = np.max(X_0[0, :, 0]) + 5
        y_max = np.max(X_0[0, :, 1]) + 5
      
        # Samples uniform distribution
        x = np.random.uniform(low = x_min, high = x_max, size=n_goals)
        y = np.random.uniform(low = y_min, high = y_max, size=n_goals)
      
        
        # Creates goals array
        goals = np.stack((x, y), axis=1)  
        goals = np.repeat(np.expand_dims(goals, 0), n_samples, axis = 0)
        
        dist_pertub = (min_dist)/(4.*np.sqrt(2))
        
        # Now generate the noise
        pertubation = np.random.uniform(low = -dist_pertub,
                                        high = dist_pertub,
                                        size = (n_samples, n_goals,  2))
        
        goals = goals + pertubation
      
        return goals
    
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
        
        for sample in range(0, self.n_samples):
            for t in range(0, self.t_samples):
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
        
        if archit == None:
            accel = self.compute_acceleration(X = None, clip=True, t_0 = 0)
            useArchit = False
        else:
            accel = np.zeros((n_samples, t_samples, n_agents, 2))
            useArchit = True
        
        vel = np.zeros((n_samples, 
                        t_samples, 
                        n_agents, 
                        2))
        
        pos = np.zeros((n_samples, 
                        t_samples, 
                        n_agents, 
                        2))
        

        pos[:, 0, :, :] = X_0
        
        if (doPrint):
            print('\tComputing simulated trajectories...', end = ' ', flush = True)
        
        for sample in range(0, n_samples):
            for t in np.arange(1, t_samples):
        
                if (t % 25 == 0):
                    if (not useArchit):
                        new_vel = self.compute_velocity(X = pos, t_0 = t)[sample, 1, :, :]
                        new_accel = (new_vel - vel[sample, t-1, :, :]) / 0.1
                    else:
                        comm_graph = self.compute_communication_graph(pos[:, 0:t, :, :], self.degree)
                        state = self.compute_state(pos[:, 0:t, :, :], self.G, commGraph=comm_graph, doPrint=False)
                        
                        with torch.no_grad():
                            new_accel = archit(state, comm_graph)

                    accel[sample, t-1, :, :] = np.clip(new_accel, -max_accel, max_accel)
                    
                vel[sample, t, :, :] = vel[sample, t - 1, :, :] \
                         + accel[sample, t-1, :, :] * 0.1 
                         
                pos[sample, t, :, :] = pos[sample, t - 1, :, :] \
                    + vel[sample, t - 1, :, :] * 0.1 \
                    + accel[sample, t - 1, :, :] * 0.1**2 / 2
                    
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
            
        return pos, vel, accel
    
    def compute_state(self, X, G, commGraph, degree, doPrint = True):
        """ 
        Computes the states for all agents at all t_samples and all n_samples.
        The state is a matrix with contents [X_agent, X_closest, G_closest],
        where X_agent is the position of the agent itself, X_closest is the
        position of the n_degree closest agents and G_closest is the position
        of the n_degree closest goals. Each state, therefore, has 2(2*self.degree + 1) elements
        
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
        np.array (n_samples x t_samples x (2 * (2*self.degree + 1)) x n_agents)
        
        """
        
        n_samples = X.shape[0]
        t_samples = X.shape[1]
        n_agents = X.shape[2]

        if (doPrint):
            print('\tComputing states...', end = ' ', flush = True)
        
        d = 2 * degree + 1
        state = np.zeros((n_samples, t_samples, d * 2, n_agents))
        
        # Finding closest goals
        for sample in range(0, n_samples):
                for t in range(0, t_samples):
                    agents = X[sample, t, :,:]
                    goals = G[sample, :,:]
                    
                    # Calculates distance matrix
                    distance_matrix = cdist(agents, goals)
                    
                    for agent in range(0, n_agents):
                        distance_to_goals = distance_matrix[agent, :]
                        closest_goals_index = np.argpartition(distance_to_goals, degree)[0:degree]
                        
                        # TODO: relative or absolute position?
                        # distance_to_closest = np.tile(agents[agent], (self.degree, 1)) - goals[closest_goals_index]
                        
                        # Goals
                        state[sample, t, -degree * 2:, agent] = goals[closest_goals_index].flatten()
                        
                        # Own positions  
                        state[sample, t, 0:2, agent] = X[sample, t, agent,:].flatten()
                        
                        # Other agents
                        closest_agents_index = commGraph[sample, t, agent, :] == 1
                        state[sample, t, 2:(degree+1)*2, agent] = X[sample, t, closest_agents_index].flatten()
            
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
            
        return state
    
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
            mean_cost = mean_cost + (1 / (sample*t_samples + 1)) * (curr_cost - mean_cost)
            
        return -mean_cost
    
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



##########
# Driver #
##########

capt = CAPT(n_agents = 50,
            min_dist = 0.5, 
            t_f = 10, 
            max_accel = 5,
            degree = 5,
            nTrain = 1, 
            nValid = 1, 
            nTest = 1)

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(capt, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Plotting (uncomment to visualize trajectory)

# sample = 0
# pos, vel, accel = capt.pos_all, capt.vel_all, capt.accel_all

# for t in range(0, pos.shape[1]):
#     plt.scatter(pos[sample, t, :, 0], 
#                 pos[sample, t, :, 1], 
#                 marker='.', 
#                 color='gray',
#                 label='',
#                 s=0.8, linewidths=0.2)

# plt.scatter(capt.G_all[sample, :, 0], capt.G_all[sample, :, 1], 
#                 label="goal", marker='x', color='r')

# plt.scatter(pos[sample, 0, :, 0], 
#             pos[sample, 0, :, 1], 
#             marker='o', 
#             color='red',
#             label='start')

# plt.grid()    
# plt.title('Trajectories')
# plt.legend()
# plt.show()
# #plt.savefig('/home/jcervino/summer-research/constrained-RL/plots/img-test.png')
