import numpy as np
# import utils
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def computeDifferences(u):
        
        # Takes as input a tensor of shape
        #   nSamples x tSamples x 2 x nAgents
        # or of shape
        #   nSamples x 2 x nAgents
        # And returns the elementwise difference u_i - u_j of shape
        #   nSamples (x tSamples) x 2 x nAgents x nAgents
        # And the distance squared ||u_i - u_j||^2 of shape
        #   nSamples (x tSamples) x nAgents x nAgents
        
        # Check dimensions
        assert len(u.shape) == 3 or len(u.shape) == 4
        # If it has shape 3, which means it's only a single time instant, then
        # add the extra dimension so we move along assuming we have multiple
        # time instants
        if len(u.shape) == 3:
            u = np.expand_dims(u, 1)
            hasTimeDim = False
        else:
            hasTimeDim = True
        
        # Now we have that pos always has shape
        #   nSamples x tSamples x 2 x nAgents
        nSamples = u.shape[0]
        tSamples = u.shape[1]
        assert u.shape[2] == 2
        nAgents = u.shape[3]
        
        # Compute the difference along each axis. For this, we subtract a
        # column vector from a row vector. The difference tensor on each
        # position will have shape nSamples x tSamples x nAgents x nAgents
        # and then we add the extra dimension to concatenate and obtain a final
        # tensor of shape nSamples x tSamples x 2 x nAgents x nAgents
        # First, axis x
        #   Reshape as column and row vector, respectively
        uCol_x = u[:,:,0,:].reshape((nSamples, tSamples, nAgents, 1))
        uRow_x = u[:,:,0,:].reshape((nSamples, tSamples, 1, nAgents))
        #   Subtract them
        uDiff_x = uCol_x - uRow_x # nSamples x tSamples x nAgents x nAgents
        # Second, for axis y
        uCol_y = u[:,:,1,:].reshape((nSamples, tSamples, nAgents, 1))
        uRow_y = u[:,:,1,:].reshape((nSamples, tSamples, 1, nAgents))
        uDiff_y = uCol_y - uRow_y # nSamples x tSamples x nAgents x nAgents
        # Third, compute the distance tensor of shape
        #   nSamples x tSamples x nAgents x nAgents
        uDistSq = uDiff_x ** 2 + uDiff_y ** 2
        # Finally, concatenate to obtain the tensor of differences
        #   Add the extra dimension in the position
        uDiff_x = np.expand_dims(uDiff_x, 2)
        uDiff_y = np.expand_dims(uDiff_y, 2)
        #   And concatenate them
        uDiff = np.concatenate((uDiff_x, uDiff_y), 2)
        #   nSamples x tSamples x 2 x nAgents x nAgents
            
        # Get rid of the time dimension if we don't need it
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDistSq = uDistSq.squeeze(1)
            #   nSamples x nAgents x nAgents
            uDiff = uDiff.squeeze(1)
            #   nSamples x 2 x nAgents x nAgents
            
        return uDiff, uDistSq


class CAPT:
  """
  A wrapper class to execute the CAPT algorithm by Matthew Turpin
  (https://journals.sagepub.com/doi/10.1177/0278364913515307). 
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

  def __init__(self, n_agents, comm_radius, min_dist, n_samples) -> None:
      self.zeroTolerance = 1e-7
      self.n_agents = n_agents
      self.comm_radius = comm_radius
      self.min_dist = min_dist
      self.n_goals = n_agents

      self.agent_init_pos = self.compute_agents_initial_positions(n_agents, 
                                                               n_samples, 
                                                               comm_radius)
      
      self.goal_init_pos = self.compute_goals_initial_positions(self.n_goals, 
                                                                n_samples)

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
      
      # If there are other keys in the kwargs argument, they will just be
      # ignored

      # Let's start by setting the fixed position
          
      # This grid has a distance that depends on the desired min_dist and
      # the comm_radius
      distFixed = (comm_radius + min_dist)/(2.*np.sqrt(2))
      #   This is the fixed distance between points in the grid
      distPerturb = (comm_radius - min_dist)/(4.*np.sqrt(2))
      #   This is the standard deviation of a uniform perturbation around
      #   the fixed point.
      # This should guarantee that, even after the perturbations, there
      # are no agents below min_dist, and that all agents have at least
      # one other agent within comm_radius.
      
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

      # And repeat for the number of samples we want to generate
      fixedPos = np.repeat(np.expand_dims(fixedPos, 0), n_samples,
                              axis = 0)
      
      # Now generate the noise
      perturbPos = np.random.uniform(low = -distPerturb,
                                      high = distPerturb,
                                      size = (n_samples, 2, n_agents))
      # Initial positions
      initPos = fixedPos + perturbPos
      
      # Now, check that the conditions are met:
      #   Compute square distances
      _, distSq = computeDifferences(np.expand_dims(initPos, 1))

      # Get rid of the "time" dimension that arises from using the 
      # method to compute distances
      distSq = distSq.squeeze(1)

      # Compute the minimum distance (don't forget to add something in
      # the diagonal, which otherwise is zero)
      min_distSq = np.min(distSq + 2 * comm_radius * 
                          np.eye(distSq.shape[1]).reshape(1, 
                          distSq.shape[1], distSq.shape[2]))

      assert min_distSq >= min_dist ** 2
              
      return initPos.reshape(n_samples, 2, n_agents)

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
    x_min = np.min(self.agent_init_pos[0, 0, :]) - 3
    y_min = np.min(self.agent_init_pos[0, 1, :]) - 3
    x_max = np.max(self.agent_init_pos[0, 0, :]) + 3
    y_max = np.max(self.agent_init_pos[0, 1, :]) + 3

    x = np.random.uniform(low = x_min, high = x_max, size=30)
    y = np.random.uniform(low = y_min, high = y_max, size=30)

    
    goals = np.stack((x, y), axis=0)  
    goals = np.repeat(np.expand_dims(goals, 0), n_samples, axis = 0)

    return goals

  def plot_initial_posiitons(self):
    """ 
    Plots initial positions of the goals and agents
    
    Parameters
    ----------
    N/A
    
    Returns
    -------
    N/A
    """
    
    plt.scatter(self.goal_init_pos[0, 0, :], self.goal_init_pos[0, 1, :], label="goals")
    plt.scatter(self.agent_init_pos[0, 0, :], self.agent_init_pos[0, 1, :], label="agents")
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
    
    agents = self.agent_init_pos[0,:,:].T
    goals = self.goal_init_pos[0,:,:].T
    print(agents.shape)

    distance_matrix = cdist(agents, goals)

    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    phi = np.zeros((self.n_agents, self.n_agents))
    phi[row_ind, col_ind] = 1

    return phi

capt = CAPT(30, 5, 0.1, 3)
capt.plot()
phi = capt.compute_assignment_matrix()