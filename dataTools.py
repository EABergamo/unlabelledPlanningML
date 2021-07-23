import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

import utils
import numpy as np
import torch

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

