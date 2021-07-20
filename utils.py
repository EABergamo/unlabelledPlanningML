import numpy as np
import torch

zeroTolerance = 1e-9

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

def rotateVector(v_1, v_2):
    dot_product = np.sum((v_1 * v_2), axis=1)
    lengths_multiplied = np.sqrt(np.sum((v_1 * v_1), axis=1)) * \
        np.sqrt(np.sum((v_2 * v_2), axis=1))
        
        
    cos = np.arccos(dot_product / lengths_multiplied)    
    angles = np.arccos(dot_product / lengths_multiplied)
    
    rotations = np.array([v_1[:,0] * np.cos(angles) - v_1[:,1] * np.sin(angles), 
                 v_1[:,0] * np.sin(angles) + v_1[:,1] * np.cos(angles)]).T
    
    return rotations

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

def invertTensorEW(x):
    
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    xInv = x.copy() # Copy the matrix to invert
    # Replace zeros for ones.
    xInv[x < zeroTolerance] = 1. # Replace zeros for ones
    xInv = 1./xInv # Now we can invert safely
    xInv[x < zeroTolerance] = 0. # Put back the zeros
    
    return xInv
