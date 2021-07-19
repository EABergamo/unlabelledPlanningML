import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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
        self.n_samples = n_samples
        
        self.t_f = t_f
    
        self.X = self.compute_agents_initial_positions(n_agents, 
                                                       n_samples, 
                                                       comm_radius)
        
        
        self.G = self.compute_goals_initial_positions(self.n_goals, 
                                                      n_samples)
        
        self.phi = self.compute_assignment_matrix(0)
        
        self.Phi = np.kron(self.phi, np.eye(self.n_goals)) # TODO: required?
    
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
    
    def compute_assignment_matrix(self, sample):
        """ 
        Computes assignment matrix using the Hungarian Algorithm
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_samples x n_agents x n_goals)
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
        
        # And repeat for the number of samples we want to generate
        phi = np.repeat(np.expand_dims(phi, 0), 
                        self.n_samples,
                        axis = 0)
      
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

    def compute_trajectory(self, sample, t):
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
        phi = self.phi[sample,:,:]
        G = self.G[sample,:,:]
        X = self.X[sample,:,:]
        N = self.n_agents
        I = np.eye(N)
        
        trajectory = (1 - beta) * X \
            + beta * (phi @ G + (I - phi @ phi.T) @ X)

        return trajectory
    
    def compute_full_trajectory(self, plot=False):
        """ 
        Computes the matrix X(t) (agent location) for all t such
        that t_0 <= t <= t_f and optionally plots it.
        
        Parameters
        ----------
        plot : boolean
            determines whether to plot the trajectory or not
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        last_index = int(self.t_f / 0.1)
        
        complete_trajectory = np.zeros((self.n_samples, 
                                        last_index, 
                                        self.n_agents, 
                                        2))
        
        for sample in range(0, self.n_samples):
            for index in np.arange(0, last_index):
                t = index * 0.1
                complete_trajectory[sample, index, :, :] = (self.compute_trajectory(sample, t))
                
                if (plot and sample == self.n_samples - 1):
                    plt.scatter(complete_trajectory[sample, index, :, 0], 
                                complete_trajectory[sample, index, :, 1], 
                                marker='.', 
                                color='k')
          
        if (plot):    
            plt.scatter(self.G[0, :, 0], self.G[0, :, 1], 
                            label="agents", marker='x', color='r')
            plt.grid()    
            plt.title('Trajectories')
        
        return complete_trajectory
    
    def compute_velocity(self):
        """ 
        Computes the matrix with the velocity (v_x, v_y) of each agent for all t such
        that t_0 <= t <= t_f.
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        complete_trajectory = self.compute_full_trajectory(plot=False)
        
        # Calculate the difference at each step
        v_x = np.diff(complete_trajectory[:,:,:,0], axis=1) / 0.1
        v_y = np.diff(complete_trajectory[:,:,:,1], axis=1) / 0.1
        
        # Stack the vectors
        vel = np.stack((v_x, v_y), axis=-1)
        
        # Add velocity for t = 0
        v_0 = np.zeros((self.n_samples, 1, 50, 2))
        vel = np.concatenate((v_0, vel), axis=1)
        
        return vel
    
    def compute_acceleration(self):
        """ 
        Computes the matrix with the acceleration (a_x, a_y) of each agent for 
        all t such that t_0 <= t <= t_f.
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        complete_velocity = self.compute_velocity()
        
        # Calculate the difference at each step
        a_x = np.diff(complete_velocity[:,:,:,0], axis=1) / 0.1
        a_y = np.diff(complete_velocity[:,:,:,1], axis=1) / 0.1
        
        # Stack the vectors
        accel = np.stack((a_x, a_y), axis=-1)
        
        # Add velocity for t = 0
        accel_0 = np.zeros((self.n_samples, 1, 50, 2))
        accel_0 = np.concatenate((accel_0, accel), axis=1)
        
        return accel
    
capt = CAPT(50, 6, 2, 3, 5)
traj = capt.compute_full_trajectory(plot=True)[0]

vel = capt.compute_velocity()[0]
accel = capt.compute_acceleration()[0]



