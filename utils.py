from numpy import expand_dims, concatenate

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
            u = expand_dims(u, 1)
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
        uDiff_x = expand_dims(uDiff_x, 2)
        uDiff_y = expand_dims(uDiff_y, 2)
        #   And concatenate them
        uDiff = concatenate((uDiff_x, uDiff_y), 2)
        #   nSamples x tSamples x 2 x nAgents x nAgents
            
        # Get rid of the time dimension if we don't need it
        if not hasTimeDim:
            # (This fails if tSamples > 1)
            uDistSq = uDistSq.squeeze(1)
            #   nSamples x nAgents x nAgents
            uDiff = uDiff.squeeze(1)
            #   nSamples x 2 x nAgents x nAgents
            
        return uDiff, uDistSq
