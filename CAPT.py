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

    def __init__(self, n_agents, comm_radius, min_dist, n_samples, 
                 max_vel = None, t_f=None, max_accel = 5, degree = 5):
        self.zeroTolerance = 1e-7
        self.n_agents = n_agents
        self.comm_radius = comm_radius
        self.min_dist = min_dist
        self.n_goals = n_agents
        self.n_samples = n_samples
        self.max_accel = max_accel
        self.degree = degree
        
        if (max_vel is None):
            self.max_vel = 10
        else:
            self.max_vel = max_vel
        
        if (t_f is None):
            self.t_f = 10 / max_vel
        else:
            self.t_f = t_f
            
        self.t_samples = int(self.t_f / 0.1)
        
        self.X = np.zeros((n_samples, self.t_samples, n_agents, 2))
        
        self.X[:, 0, :, :] = self.compute_agents_initial_positions(n_agents, 
                                                       n_samples, 
                                                       comm_radius,
                                                       min_dist = min_dist)
        
        self.G = self.compute_goals_initial_positions(self.n_goals, 
                                                      n_samples,
                                                      min_dist)
        
        self.phi = self.compute_assignment_matrix(self.X, self.G)
        
        self.X = self.capt_trajectory(doPrint = True)
        
        self.comm_graph = self.compute_communication_graph(self.X,
                                                           comm_radius)
        
        self.pos, self.vel, self.accel = self.simulated_trajectory()
        
        self.states = self.compute_state()
        
        
         
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
        comm_radius : double
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
    
    def compute_goals_initial_positions(self, n_goals, n_samples, min_dist):
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
        np.array (n_samples x n_goals x 2) 
        """
        
        # Find max/min positions
        x_min = np.min(self.X[0, 0, :, 0]) - 5
        y_min = np.min(self.X[0,0, :, 1]) - 5
        x_max = np.max(self.X[0, 0,:, 0]) + 5
        y_max = np.max(self.X[0, 0,:, 1]) + 5
      
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
    
    def compute_assignment_matrix(self, X, G, doPrint = True):
        """ 
        Computes assignment matrix using the Hungarian Algorithm
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_samples x n_agents x n_goals)
        double (max distance)
        """
        
        n_samples = self.n_samples
        phi = np.zeros((n_samples, self.n_agents, self.n_agents))

        if (doPrint):
            print('\tComputing assignment matrix...', end = ' ', flush = True)
        
        for sample in range(0, n_samples):
            # Obtains the initial posiition arrays
            agents = self.X[sample, 0, :,:]
            goals = self.G[sample, :,:]
            
            # Calculates distance matrix
            distance_matrix = cdist(agents, goals)
          
            # Obtains optimal linear combination
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
          
            # Obtains assignment matrix (binary)
            phi[sample, row_ind, col_ind] = 1
        
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

        return phi
    
    def get_beta(self, t_0, t):
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
        
        t_f = self.t_f
        
        alpha_0 = -t_0 / (t_f - t_0)
        alpha_1 = 1 / (t_f - t_0)
        
        return (alpha_0 * 1 + alpha_1 * t)

    def compute_trajectory(self, sample, index, t_0 = 0):
        """ 
        Computes the matrix X(t) (agent location) for the input t
        
        Parameters
        ----------
        index : double
            time index such that we obtain X(t)
        t_0 : double
            starting time index (i.e. the reference position to obtain X(t_0)),
            we set as default 0.0
        
        Returns
        -------
        np.array (n_agents x 2)
        """
        
        t_0 = int(t_0 * 0.1)
        t = index * 0.1
        
        beta = self.get_beta(t_0, t)
        phi = self.phi[sample,:,:]
        G = self.G[sample,:,:]
        X = self.X[sample,t_0*10, :,:]
        
        
        N = self.n_agents
        I = np.eye(N)

        trajectory = (1 - beta) * X \
            + beta * (phi @ G + (I - phi @ phi.T) @ X)
            

        return trajectory
    
    def capt_trajectory(self, doPrint=True, t_0 = 0):
        """ 
        Computes the matrix X(t) (agent location) for all t such
        that t_0 <= t <= t_f and optionally plots it. It will use the CAPT
        algorithm with no modifications; as such, it might produce trajectories
        that require unfeasiable velocities/accelerations. It will, however,
        produce the right *direction* of the trajectories - this can be used
        later with other functions to generate a more realistic trajectory.
        
        Parameters
        ----------
        doPrint : boolean
            determines whether to print the progress or not
        t_0 : integer
            index that corresponds to the time that the trajectory starts
            
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        t_samples = int((self.t_f - t_0 * 0.1) / 0.1)
        
        complete_trajectory = np.zeros((self.n_samples, 
                                        t_samples, 
                                        self.n_agents, 
                                        2))
        
        
        if (doPrint):
            print('\tComputing CAPT trajectories...', end = ' ', flush = True)
        
        for sample in range(0, self.n_samples):
            for index in np.arange(0, t_samples):
                complete_trajectory[sample, index, :, :] = \
                    self.compute_trajectory(sample, index, t_0)
                 
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
    
    def compute_communication_graph(self, pos, comm_radius, 
                                    normalize_graph = True,
                                    doPrint = True):
        """ 
        Computes the communication graphs S for the entire position array at
        each time instant.
        
        Parameters
        ----------
        pos : np.array (n_samples x t_samples x n_agents, 2)
            position array for the agents
        comm_radius : double
            communication radius
        normalize_graph : boolean
            whether to normalize all elements by the largest eigenvalue
        doPrint : boolean
            whether to print progress or not.
            
            
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        
        graphMatrix = np.zeros((self.n_samples, 
                                self.t_samples, 
                                self.n_agents, 
                                self.n_agents))
        
        if (doPrint):
            print('\tComputing communication graph...', end = ' ', flush = True)
        
        for sample in range(0, self.n_samples):
            for t in range(0, self.t_samples):
                neigh = NearestNeighbors(n_neighbors=self.degree)
                neigh.fit(pos[sample, t, :, :])
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
        
    
    def compute_velocity(self, doPrint = True, t_0 = 0):
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
        complete_trajectory = self.capt_trajectory(doPrint=False, t_0 = t_0)
        
        # Calculate the difference at each step
        v_x = np.diff(complete_trajectory[:,:,:,0], axis=1) / 0.1
        v_y = np.diff(complete_trajectory[:,:,:,1], axis=1) / 0.1
        
        # Stack the vectors
        vel = np.stack((v_x, v_y), axis=-1)
        
        # Add velocity for t = 0
        v_0 = np.zeros((self.n_samples, 1, self.n_agents, 2))
        vel = np.concatenate((v_0, vel), axis=1)
        
        return vel
    
    def compute_acceleration(self, clip=True, t_0 = 0):
        """ 
        Computes the matrix with the acceleration (a_x, a_y) of each agent for 
        all t such that t_0 <= t <= t_f.
        
        Parameters
        ----------
        clip : boolean
            Determines wheter to limit the acceleration to the interval
            [-max_accel, max_accel]
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        complete_velocity = self.compute_velocity(t_0 = t_0)
        
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
    
    def simulated_trajectory(self, doPrint = True):
        """ 
        Calculates trajectory using the calculated acceleration. This function
        is particularly useful when clip is set to True in 
        .compute_acceleration since it will generate trajectories that are
        physically feasible.
        
        Parameters
        ----------
        N/A
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x n_agents x 2)
        
        """
        t_samples = int(self.t_f / 0.1)
        
        accel = self.compute_acceleration(clip=True, t_0 = 0)
        
        vel = np.zeros((self.n_samples, 
                        t_samples, 
                        self.n_agents, 
                        2))
        
        pos = np.zeros((self.n_samples, 
                        t_samples, 
                        self.n_agents, 
                        2))
        
        pos = self.X
        
        if (doPrint):
            print('\tComputing simulated trajectories...', end = ' ', flush = True)
        
        for sample in range(0, self.n_samples):
            for t in np.arange(1, t_samples):
        
                if (t % 25 == 0):
                    new_vel = self.compute_velocity(t_0 = t)[sample, 1, :, :]
                   
                    new_accel = (new_vel - vel[sample, t-1, :, :]) / 0.1
                    
                    accel[sample, t-1, :, :] = np.clip(new_accel, -self.max_accel, self.max_accel)
                    
                vel[sample, t, :, :] = vel[sample, t - 1, :, :] \
                         + accel[sample, t-1, :, :] * 0.1 
                         
                pos[sample, t, :, :] = pos[sample, t - 1, :, :] \
                    + vel[sample, t - 1, :, :] * 0.1 \
                    + accel[sample, t - 1, :, :] * 0.1**2 / 2
                    
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
            
        return pos, vel, accel
    
    def compute_state(self, doPrint = True):
        """ 
        Computes the states for all agents at all t_samples and all n_samples.
        The state is a matrix with contents [X_agent, X_closest, G_closest],
        where X_agent is the position of the agent itself, X_closest is the
        position of the n_degree closest agents and G_closest is the position
        of the n_degree closest goals.
        
        Parameters
        ----------
        doPrint : boolean
            whether to print progress or not.
        
        Returns
        -------
        np.array (n_samples x (t_f / 0.1) x (2 * self.degree + 10 x n_agents x 2)
        
        """
        
        if (doPrint):
            print('\tComputing states...', end = ' ', flush = True)
        
        d = 2 * self.degree + 1
        state = np.zeros((self.n_samples, self.t_samples, d, self.n_agents, 2))
        
        # Finding closest goals
        for sample in range(0, self.n_samples):
                for t in range(0, self.t_samples):
                    agents = self.X[sample, t, :,:]
                    goals = self.G[sample, :,:]
                    
                    # Calculates distance matrix
                    distance_matrix = cdist(agents, goals)
                    
                    for agent in range(0, self.n_agents):
                        distance_to_goals = distance_matrix[agent, :]
                        closest_goals_index = np.argpartition(distance_to_goals, self.degree)[0:self.degree]
                        
                        # TODO: relative or absolute position?
                        # distance_to_closest = np.tile(agents[agent], (self.degree, 1)) - goals[closest_goals_index]
                        
                        # Goals
                        state[sample, t, -self.degree:, agent, :] = goals[closest_goals_index]
                        
                        # Own positions
                        state[sample, t, 0, agent, :] = self.X[sample, t, agent,:]
                        
                        # Other agents
                        closest_agents_index = self.comm_graph[sample, t, agent, :] == 1
                        state[sample, t, 1:self.degree+1, agent, :] = self.X[sample, t, closest_agents_index]
            
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
            
        return state
    
    def evaluate(self, R):
        """ 
        Computes the total cost of the trajectory averaged over all samples. 
        The cost is associated with the number of goals with no agent located
        at distance less than R.
        
        Parameters
        ----------
        R : double
            tolerance regarding goal-agent distance
        
        Returns
        -------
        double
        
        """
        final_pos = self.X[:,-1, :, :]
        goals = self.G
        mean_cost = 0
        
        for sample in range(0, self.n_samples):
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
            
        

start = timeit.default_timer()
print('Starting...')

sample = 0 # sample to graph    

capt = CAPT(n_agents = 30, 
            comm_radius=6, 
            min_dist=2, 
            n_samples=1, 
            t_f = 10, 
            max_accel = 10, 
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