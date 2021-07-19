import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class CAPT:
    """
    A wrapper class to execute the CAPT algorithm by Matthew Turpin
    (https://journals.sagepub.com/doi/10.1177/0278364913515307). Certain 
    parts of this code are originally from the Alelab GNN library 
    (https://github.com/alelab-upenn).
    ...
    
    Attributes
    ----------
    n_agents : int
              The total number of agents that will take part in the simulation
    n_samples : int
        The total number of samples.
    comm_radius : double
        The communication radius between agents (determines initial spacing between agents)
    min_dist : double
        The minimum distance between each agent
    
    Methods
    -------
    computeInitialPositions : np.array (n_samples x 2 x n_agents)
    """

    def __init__(self, n_agents, comm_radius, min_dist, n_samples, t_f):
        self.zeroTolerance = 1e-7
        self.n_agents = n_agents
        self.comm_radius = comm_radius
        self.min_dist = min_dist
        self.n_goals = n_agents
        
        self.t_f = t_f
    
        self.X = self.compute_agents_initial_positions(n_agents, 
                                                                 n_samples, 
                                                                 comm_radius)
        
        
        self.G = self.compute_goals_initial_positions(self.n_goals, 
                                                                  n_samples)
        
        self.phi = self.compute_assignment_matrix()
        
        self.Phi = np.kron(self.phi, np.eye(self.n_goals))
    
    def compute_agents_initial_positions(self, n_agents, n_samples, comm_radius,
                                        min_dist = 0.1, **kwargs):
        """ 
        Generates a NumPy array with the 
        initial x, y position for each of the n_agents
    
        Parameters
        ----------
        n_agents : int
            The total number of agents that will take part in the simulation
        n_samples : int
            The total number of samples.
        comm_radius : double
            The communication radius between agents (determines initial spacing between agents)
        min_dist : double
            The minimum distance between each agent
    
        Returns
        -------
        np.array (n_samples x 2 x n_agents) 
        """
        
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
              
        return initPos
        # return initPos.reshape(n_samples, n_agents, 2)
    
    def compute_goals_initial_positions(self, n_goals, n_samples):
        """ 
        Generates a NumPy array with the 
        initial x, y position for each of the n_goals
        
        Parameters
        ----------
        n_agents : int
            The total number of goals that will take part in the simulation
        n_samples : int
            The total number of samples.
        comm_radius : double
            The communication radius between agents (determines initial spacing between agents)
        min_dist : double
            The minimum distance between each agent
        
        Returns
        -------
        np.array (n_samples x 2 x n_agents) 
        """
        
        # Find max/min positions
        x_min = np.min(self.X[0, :, 0]) - 10
        y_min = np.min(self.X[0, :, 1]) - 10
        x_max = np.max(self.X[0, :, 0]) + 10
        y_max = np.max(self.X[0, :, 1]) + 10
      
        # Samples uniform distribution
        x = np.random.uniform(low = x_min, high = x_max, size=n_goals)
        y = np.random.uniform(low = y_min, high = y_max, size=n_goals)
      
        
        # Creates goals array
        goals = np.stack((x, y), axis=1)  
        goals = np.repeat(np.expand_dims(goals, 0), n_samples, axis = 0)
      
        return goals
    
    def plot_initial_positions(self):
        """ 
        Plots initial positions of the goals and agents
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        N/A
        """
        
        plt.scatter(self.X[0, :, 0], self.X[0, :, 1], label="goals", marker='.')
        plt.scatter(self.G[0, :, 0], self.G[0, :, 1], label="agents", marker='x')
        plt.title("Initial Positions")
        plt.legend()
        plt.show()
    
    def compute_assignment_matrix(self):
        """ 
        Computes assignment matrix using the Hungarian Algorithm
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_agents x n_goals)
        """
        
        # Obtains the initial posiition arrays
        agents = self.X[0,:,:]
        goals = self.G[0,:,:]
        
        
        # Calculates distance matrix
        distance_matrix = cdist(agents, goals)
      
        # Obtains optimal linear combination
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
      
        # Obtains assignment matrix (binary)
        phi = np.zeros((self.n_agents, self.n_agents))
        phi[row_ind, col_ind] = 1
      
        return phi
    
    def get_beta(self, t):
        """ 
        Computes the polynomial function of time Beta as described in
        the CAPT paper.
        
        Parameters
        ----------
        t : double
            time index such that we obtain β(t)
        
        Returns
        -------
        double (β(t))
        """
        
        t_0 = 0
        t_f = self.t_f
        
        alpha_0 = -t_0 / (t_f - t_0)
        alpha_1 = 1 / (t_f - t_0)
        
        return (alpha_0 * 1 + alpha_1 * t)

    def compute_trajectory(self, t):
        """ 
        Computes the matrix X(t) (agent location) for the input t
        
        Parameters
        ----------
        t : double
            time index such that we obtain X(t)
        
        Returns
        -------
        np.array (n_agents x 2)
        """
        
        beta = self.get_beta(t)
        phi = self.phi
        G = self.G[0,:,:]
        X = self.X[0,:,:]
        N = self.n_agents
        I = np.eye(N)
        
        trajectory = (1 - beta) * X \
            + beta * (phi @ G + (I - phi @ phi.T) @ X)

        return trajectory
    
    def plot_trajectories(self):
        trajectories = []

        for t in np.arange(0, 5, 0.1):
            trajectories.append(self.compute_trajectory(t))
            plt.scatter(self.G[0, :, 0], self.G[0, :, 1], 
                        label="agents", marker='x', color='r')
            plt.scatter(trajectories[-1][:, 0], 
                        trajectories[-1][:, 1], marker='.', color='k')
          
        plt.grid()    
        plt.title('Trajectories')

    



