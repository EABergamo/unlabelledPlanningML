import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import utils
import timeit


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
                 max_vel = None, t_f=None, max_accel = 5):
        self.zeroTolerance = 1e-7
        self.n_agents = n_agents
        self.comm_radius = comm_radius
        self.min_dist = min_dist
        self.n_goals = n_agents
        self.n_samples = n_samples
        self.max_accel = max_accel
        
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
            print('\tComputing trajectories...', end = ' ', flush = True)
        
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
        
        if (doPrint):
            print('\tComputing communication graph...', end = ' ', flush = True)
        
        pos = np.transpose(pos, (0, 1, 3, 2))   
        
        maxBatchSize = 100
        maxTimeSamples = 200
        kernelScale = 1
                
        assert comm_radius > 0
        assert len(pos.shape) == 4
        n_samples = pos.shape[0]
        tSamples = pos.shape[1]
        assert pos.shape[2] == 2
        nAgents = pos.shape[3]
        
        # Compute the number of samples, and split the indices accordingly
        if n_samples < maxBatchSize:
            nBatches = 1
            batchSize = [n_samples]
        elif n_samples % maxBatchSize != 0:
            # If we know it's not divisible, then we do floor division and
            # add one more batch
            nBatches = n_samples // maxBatchSize + 1
            batchSize = [maxBatchSize] * nBatches
            # But the last batch is actually smaller, so just add the 
            # remaining ones
            batchSize[-1] = n_samples - sum(batchSize[0:-1])
        # If they fit evenly, then just do so.
        else:
            nBatches = int(n_samples/maxBatchSize)
            batchSize = [maxBatchSize] * nBatches
        # batchIndex is used to determine the first and last element of each
        # batch. We need to add the 0 because it's the first index.
        batchIndex = np.cumsum(batchSize).tolist()
        batchIndex = [0] + batchIndex
        
        # Create the output state variable
        graphMatrix = np.zeros((n_samples, tSamples, nAgents, nAgents))
        
        for b in range(nBatches):
            
            # Pick the batch elements
            posBatch = pos[batchIndex[b]:batchIndex[b+1]]
                
            if tSamples > maxTimeSamples:
                # If the trajectories are longer than 200 points, then do it 
                # time by time.
                
                # For each time instant
                for t in range(tSamples):
                    # Let's start by computing the distance squared
                    _, distSq = utils.computeDifferences(posBatch[:,t,:,:])
                    # Apply the Kernel
                    graphMatrixTime = np.exp(-kernelScale * distSq)
                  
                    # Now let's place zeros in all places whose distance is greater
                    # than the radius
                    graphMatrixTime[distSq > (comm_radius ** 2)] = 0.
                    # Set the diagonal elements to zero
                    graphMatrixTime[:,\
                                    np.arange(0,nAgents),np.arange(0,nAgents)]\
                                                                           = 0.
                    # If it is unweighted, force all nonzero values to be 1
                    graphMatrixTime = (graphMatrixTime > zeroTolerance)\
                                                          .astype(distSq.dtype)
                                                              
                    if normalize_graph:
                        isSymmetric = np.allclose(graphMatrixTime,
                                                  np.transpose(graphMatrixTime,
                                                               axes = [0,2,1]))
                        # Tries to make the computation faster, only the 
                        # eigenvalues (while there is a cost involved in 
                        # computing whether the matrix is symmetric, 
                        # experiments found that it is still faster to use the
                        # symmetric algorithm for the eigenvalues)
                        if isSymmetric:
                            W = np.linalg.eigvalsh(graphMatrixTime)
                        else:
                            W = np.linalg.eigvals(graphMatrixTime)
                        maxEigenvalue = np.max(np.real(W), axis = 1)
                        #   batchSize[b]
                        # Reshape to be able to divide by the graph matrix
                        maxEigenvalue=maxEigenvalue.reshape((batchSize[b],1,1))
                        # Normalize
                        graphMatrixTime = graphMatrixTime / maxEigenvalue
                                                              
                    # And put it in the corresponding time instant
                    graphMatrix[batchIndex[b]:batchIndex[b+1],t,:,:] = \
                                                                graphMatrixTime
                    
                    if doPrint:
                        # Sample percentage count
                        percentageCount = int(100*(t+1+b*tSamples)\
                                                          /(nBatches*tSamples))
                        
                        if t == 0 and b == 0:
                            # It's the first one, so just print it
                            print("%3d%%" % percentageCount,
                                  end = '', flush = True)
                        else:
                            # Erase the previous characters
                            print('\b \b' * 4 + "%3d%%" % percentageCount,
                                  end = '', flush = True)
                
            else:
                # Let's start by computing the distance squared
                _, distSq = utils.computeDifferences(posBatch)
                # Apply the Kernel
                graphMatrixBatch = np.exp(-kernelScale * distSq)
                # Now let's place zeros in all places whose distance is greater
                # than the radius
                graphMatrixBatch[distSq > (comm_radius ** 2)] = 0.
                # Set the diagonal elements to zero
                graphMatrixBatch[:,:,
                                 np.arange(0,nAgents),np.arange(0,nAgents)] =0.
                                 
                # If it is unweighted, force all nonzero values to be 1
                graphMatrixBatch = (graphMatrixBatch > zeroTolerance)\
                                                          .astype(distSq.dtype)
                    
                if normalize_graph:
                    isSymmetric = np.allclose(graphMatrixBatch,
                                              np.transpose(graphMatrixBatch,
                                                            axes = [0,1,3,2]))
                    # Tries to make the computation faster
                    if isSymmetric:
                        W = np.linalg.eigvalsh(graphMatrixBatch)
                    else:
                        W = np.linalg.eigvals(graphMatrixBatch)
                    maxEigenvalue = np.max(np.real(W), axis = 2)
                    #   batchSize[b] x tSamples
                    # Reshape to be able to divide by the graph matrix
                    maxEigenvalue = maxEigenvalue.reshape((batchSize[b],
                                                           tSamples,
                                                           1, 1))
                    # Normalize
                    graphMatrixBatch = graphMatrixBatch / maxEigenvalue
                    
                # Store
                graphMatrix[batchIndex[b]:batchIndex[b+1]] = graphMatrixBatch
                
                if doPrint:
                    # Sample percentage count
                    percentageCount = int(100*(b+1)/nBatches)
                    
                    if b == 0:
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
        
        for sample in range(0, self.n_samples):
            for t in np.arange(1, t_samples):
        
                if (t % 25 == 0):
                    new_vel = self.compute_velocity(t_0 = t)[sample, 1, :, :]
                    vel[sample, t-1:, :, :] = new_vel
                    
                vel[sample, t, :, :] = vel[sample, t - 1, :, :] \
                         + accel[sample, t-1, :, :] * 0.1 
                pos[sample, t, :, :] = pos[sample, t - 1, :, :] \
                    + vel[sample, t - 1, :, :] * 0.1 \
                    + accel[sample, t - 1, :, :] * 0.1**2 / 2
              
        return pos

start = timeit.default_timer()

sample = 0 # sample to graph    

#np.random.seed(55)


capt = CAPT(n_agents = 5, comm_radius=6, min_dist=2, n_samples=1, t_f = 10, max_accel = 5)
X_c = capt.compute_velocity()[0]
X_t = capt.simulated_trajectory()
X_sample = X_t[sample]

for t in range(0, X_t.shape[1]):
    plt.scatter(X_t[sample, t, :, 0], 
                X_t[sample, t, :, 1], 
                marker='.', 
                color='gray',
                label='')

plt.scatter(capt.G[sample, :, 0], capt.G[sample, :, 1], 
                label="goal", marker='x', color='r')
plt.grid()    
plt.title('Trajectories')
plt.legend()
plt.savefig('/home/jcervino/summer-research/constrained-RL/plots/img-test.png')
stop = timeit.default_timer()

accel = capt.compute_acceleration()[0]

print()
print('\tTotal time: ', stop - start, 's')




