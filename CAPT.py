import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import utils
import timeit
from sklearn.neighbors import NearestNeighbors

zeroTolerance = utils.zeroTolerance

class CAPT:
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

    def __init__(self, n_agents, min_dist, n_samples, 
                 max_vel = None, t_f=None, max_accel = 5, degree = 5):
        
        self.zeroTolerance = 1e-7 # all values less then this are zero
        self.n_agents = n_agents # number of agents
        self.min_dist = min_dist # minimum initial distance between agents 
        self.n_goals = n_agents # number of goals (same as n_agents by design)
        self.n_samples = n_samples # number of samples
        self.max_accel = max_accel # max allowed acceleration
        self.degree = degree # number of edges for each node (agent)
        
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
        
        # Defining initial positions for agents
        self.X_0 = self.compute_agents_initial_positions(n_agents, 
                                                       n_samples, 
                                                       6,
                                                       min_dist = min_dist)
        
        # Defining initial positions for goals
        self.G = self.compute_goals_initial_positions(self.X_0, min_dist)
        
        # Compute assignments for agents-goals (using Hungarian Algorithm)
        self.phi = self.compute_assignment_matrix(self.X_0, self.G)
        
        # Compute complete trajectories (iterated CAPT algorithm)
        self.pos, self.vel, self.accel = self.simulated_trajectory(self.max_accel, self.X_0)
        
        # Compute communication graphs for the simulated trajectories
        self.comm_graph = self.compute_communication_graph(self.pos,
                                                           self.degree)

        
        self.states = self.compute_state(self.pos, self.G, self.comm_graph, self.degree)
        
        
         
        # Unclear if required?    
        # self.Phi = np.kron(self.phi, np.eye(self.n_goals)) 
    
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
            X = self.X_0
        
        G = self.G
        
        
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
    
    def simulated_trajectory(self, max_accel, X_0, doPrint = True):
        """ 
        Calculates trajectory using the calculated acceleration. This function
        is particularly useful when clip is set to True in 
        .compute_acceleration() since it will generate trajectories that are
        physically feasible.
        
        Parameters
        ----------
        max_accel : double
            Maximum acceleration allowed
        X_0 : np.array (n_samples x n_agents x 2) 
            Initial positions of the agents for all samples
        
        Returns
        -------
        np.array (n_samples x t_samples x n_agents x 2)
        
        """

        n_samples = X_0.shape[0]
        t_samples = int(self.t_f / 0.1)
        n_agents = X_0.shape[1]

        
        accel = self.compute_acceleration(X = None, clip=True, t_0 = 0)
        
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
                    new_vel = self.compute_velocity(X = pos, t_0 = t)[sample, 1, :, :]
                   
                    new_accel = (new_vel - vel[sample, t-1, :, :]) / 0.1
                    
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
    
    def compute_state(self, X, G, comm_graph, degree, doPrint = True):
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
        comm_graph : np.array (n_samples x t_samples x n_agents x n_agents)
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
        state = np.zeros((n_samples, t_samples, d, n_agents, 2))
        
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
                        state[sample, t, -degree:, agent, :] = goals[closest_goals_index]
                        
                        # Own positions  
                        state[sample, t, 0, agent, :] = X[sample, t, agent,:]
                        
                        # Other agents
                        closest_agents_index = comm_graph[sample, t, agent, :] == 1
                        state[sample, t, 1:degree+1, agent, :] = X[sample, t, closest_agents_index]
            
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
    
    def evaluate(self, X, G, R):
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
        final_pos = X[:,-1, :, :]
        n_samples = X.shape[0]
        goals = G
        mean_cost = 0
        
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
        

start = timeit.default_timer()
print('Starting...')

sample = 0 # sample to graph    

capt = CAPT(n_agents = 30, 
            min_dist=2, 
            n_samples=1, 
            t_f = 10, 
            max_accel = 5, 
            degree = 3)

# Plotting
pos, vel, accel = capt.pos, capt.vel, capt.accel

for t in range(0, pos.shape[1]):
    plt.scatter(pos[sample, t, :, 0], 
                pos[sample, t, :, 1], 
                marker='.', 
                color='gray',
                label='',
                s=0.8, linewidths=0.2)

plt.scatter(capt.G[sample, :, 0], capt.G[sample, :, 1], 
                label="goal", marker='x', color='r')

plt.scatter(pos[sample, 0, :, 0], 
            pos[sample, 0, :, 1], 
            marker='o', 
            color='red',
            label='start')

plt.grid()    
plt.title('Trajectories')
plt.legend()
plt.show()
#plt.savefig('/home/jcervino/summer-research/constrained-RL/plots/img-test.png')


stop = timeit.default_timer()

print(capt.evaluate(0.5))

print()
print('Total time: ', stop - start, 's')