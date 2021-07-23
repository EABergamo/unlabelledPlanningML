from CAPT import CAPT
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.dataTools as dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architecturesTime as architTime
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import CAPT

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'flockingGNN' # This is the general name of all related files

nAgents = 50 # Number of agents at training time

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-%03d-' % nAgents + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

useGPU = False # If true, and GPU is available, use it.

degree = 5
nAgentsMax = nAgents # Maximum number of agents to test the solution
nSimPoints = 1 # Number of simulations between nAgents and nAgentsMax
    # At test time, the architectures trained on nAgents will be tested on a
    # varying number of agents, starting at nAgents all the way to nAgentsMax;
    # the number of simulations for different number of agents is given by
    # nSimPoints, i.e. if nAgents = 50, nAgentsMax = 100 and nSimPoints = 3, 
    # then the architectures are trained on 50, 75 and 100 agents.
commRadius = 2. # Communication radius
repelDist = 1. # Minimum distance before activating repelling potential
nTrain = 400 # Number of training samples
nValid = 20 # Number of valid samples
nTest = 20 # Number of testing samples
duration = 8 # Duration of the trajectory
samplingTime = 0.01 # Sampling time
initGeometry = 'circular' # Geometry of initial positions
initVelValue = 3. # Initial velocities are samples from an interval
    # [-initVelValue, initVelValue]
initMinDist = 0.1 # No two agents are located at a distance less than this
accelMax = 10. # This is the maximum value of acceleration allowed

nRealizations = 1 # Number of data realizations
    # How many times we repeat the experiment

#\\\ Save values:
writeVarValues(varsFile,
               {'nAgents': nAgents,
                'nAgentsMax': nAgentsMax,
                'repelDist': repelDist,
                'nTrain': nTrain,
                'nValid': nValid,
                'nTest': nTest,
                'duration': duration,
                'initMinDist': initMinDist,
                'accelMax': accelMax,
                'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss

#\\\ Training algorithm
trainer = training.Trainer

#\\\ Evaluation algorithm
evaluator = evaluation.evaluateFlocking

#\\\ Overall training options
#probExpert = 0.993 # Probability of choosing the expert in DAGger
#DAGgerType = 'fixedBatch' # 'replaceTimeBatch', 'randomEpoch'
nEpochs = 30 # Number of epochs
batchSize = 20 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimizationAlgorithm': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'trainer': trainer,
                'evaluator': evaluator,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N'
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

#nFeatures = 32 # Number of features in all architectures
#nFilterTaps = 4 # Number of filter taps in all architectures
# [[The hyperparameters are for each architecture, and they were chosen 
#   following the results of the hyperparameter search]]
nonlinearityHidden = torch.tanh
nonlinearityOutput = torch.tanh
nonlinearity = nn.Tanh # Chosen nonlinearity for nonlinear architectures

modelList = []

# Note: designed to work with localGNN only.
doLocalGNN = True # Local GNN (include nonlinearity)

#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for the Local GNN architecture

    hParamsLocalGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

    hParamsLocalGNN['name'] = 'LocalGNN'
    # Chosen architecture
    hParamsLocalGNN['archit'] = architTime.LocalGNN_DB
    hParamsLocalGNN['device'] = 'cuda:0' \
                                    if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu'

    # Graph convolutional parameters
    hParamsLocalGNN['dimNodeSignals'] = [2*(2*degree + 1), 64] # Features per layer
    hParamsLocalGNN['nFilterTaps'] = [3] # Number of filter taps
    hParamsLocalGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLocalGNN['nonlinearity'] = nonlinearity # Selected nonlinearity
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLocalGNN['dimReadout'] = [2] # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor 
        # considering all nodes at once, making the architecture entirely
        # local.
    # Graph structure
    hParamsLocalGNN['dimEdgeFeatures'] = 1 # Scalar edge weights

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLocalGNN)
    modelList += [hParamsLocalGNN['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = False # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 1 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 10 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 2 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers
videoSpeed = 0.5 # Slow down by half to show transitions
nVideos = 3 # Number of videos to save

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize,
                'videoSpeed': videoSpeed,
                'nVideos': nVideos})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ If CUDA is selected, empty cache:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        hParamsDict = eval('hParams' + thisModel)
        print("\t%s: %s" % (thisModel, hParamsDict['device']))

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from alegnn.utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')
    
#\\\ Number of agents at test time
nAgentsTest = np.linspace(nAgents, nAgentsMax, num = nSimPoints,dtype = np.int)
nAgentsTest = np.unique(nAgentsTest).tolist()
nSimPoints = len(nAgentsTest)
writeVarValues(varsFile, {'nAgentsTest': nAgentsTest}) # Save list

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# The first list is one for each value of nAgents that we want to simulate 
# (i.e. these are test results, so if we test for different number of agents,
# we need to save the results for each of them). Each element in the list will
# be a dictionary (i.e. for each testing case, we have a dictionary).
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
# We're saving the cost of the full trajectory, as well as the cost at the end
# instant.
costBestFull = [None] * nSimPoints
costBestEnd = [None] * nSimPoints
costLastFull = [None] * nSimPoints
costLastEnd = [None] * nSimPoints
costOptFull = [None] * nSimPoints
costOptEnd = [None] * nSimPoints
for n in range(nSimPoints):
    costBestFull[n] = {} # Accuracy for the best model (full trajectory)
    costBestEnd[n] = {} # Accuracy for the best model (end time)
    costLastFull[n] = {} # Accuracy for the last model
    costLastEnd[n] = {} # Accuracy for the last model
    for thisModel in modelList: # Create an element for each split realization,
        costBestFull[n][thisModel] = [None] * nRealizations
        costBestEnd[n][thisModel] = [None] * nRealizations
        costLastFull[n][thisModel] = [None] * nRealizations
        costLastEnd[n][thisModel] = [None] * nRealizations
    costOptFull[n] = [None] * nRealizations # Accuracy for optimal controller
    costOptEnd[n] = [None] * nRealizations # Accuracy for optimal controller


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options (aka 'DAGger'), then
# we create a separate dictionary per model.

trainingOptsPerModel= {}

# Create relevant dirs: we need directories to save the videos of the dataset
# that involve the optimal centralized controllers, and we also need videos
# for the learned trajectory of each model. Note that all of these depend on
# each realization, so we will be saving videos for each realization.
# Here, we create all those directories.
datasetTrajectoryDir = os.path.join(saveDir,'datasetTrajectories')
if not os.path.exists(datasetTrajectoryDir):
    os.makedirs(datasetTrajectoryDir)
    
datasetTrainTrajectoryDir = os.path.join(datasetTrajectoryDir,'train')
if not os.path.exists(datasetTrainTrajectoryDir):
    os.makedirs(datasetTrainTrajectoryDir)
    
datasetTestTrajectoryDir = os.path.join(datasetTrajectoryDir,'test')
if not os.path.exists(datasetTestTrajectoryDir):
    os.makedirs(datasetTestTrajectoryDir)

datasetTestAgentTrajectoryDir = [None] * nSimPoints
for n in range(nSimPoints):    
    datasetTestAgentTrajectoryDir[n] = os.path.join(datasetTestTrajectoryDir,
                                                    '%03d' % nAgentsTest[n])
    
if nRealizations > 1:
    datasetTrainTrajectoryDirOrig = datasetTrainTrajectoryDir
    datasetTestAgentTrajectoryDirOrig = datasetTestAgentTrajectoryDir.copy()

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data realization for each number of total realizations

for realization in range(nRealizations):

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if nRealizations > 1:
        trainingOptions['realizationNo'] = realization
        
        # Create new directories (specific for this realization)
        datasetTrainTrajectoryDir = os.path.join(datasetTrainTrajectoryDirOrig,
                                                 '%03d' % realization)
        if not os.path.exists(datasetTrainTrajectoryDir):
            os.makedirs(datasetTrainTrajectoryDir)
            
        for n in range(nSimPoints):
            datasetTestAgentTrajectoryDir[n] = os.path.join(
                                          datasetTestAgentTrajectoryDirOrig[n],
                                          '%03d' % realization)
            if not os.path.exists(datasetTestAgentTrajectoryDir[n]):
                os.makedirs(datasetTestAgentTrajectoryDir[n])

    if doPrint:
        print("", flush = True)

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############

    if doPrint:
        print("Generating data", end = '')
        if nRealizations > 1:
            print(" for realization %d" % realization, end = '')
        print("...", flush = True)

    #   Generate the dataset

    print(nTrain)
    data = CAPT.CAPT(nAgents, initMinDist, nTrain, nValid, nTest, t_f=duration, max_accel=accelMax, degree=degree)

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # This is the dictionary where we store the models (in a model.Model
    # class).
    modelsGNN = {}

    # If a new model is to be created, it should be called for here.

    if doPrint:
        print("Model initialization...", flush = True)

    for thisModel in modelList:

        #  the corresponding parameter dictionary
        hParamsDict = deepcopy(eval('hParams' + thisModel))
        # and training options
        trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)

        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture, but it also has the 'name' and 'archit' that
        # we do not need to pass them. So we are going to get them out of
        # the dictionary
        thisName = hParamsDict.pop('name')
        callArchit = hParamsDict.pop('archit')
        thisDevice = hParamsDict.pop('device')

        # If more than one graph or data realization is going to be carried out,
        # we are going to store all of thos models separately, so that any of
        # them can be brought back and studied in detail.
        if nRealizations > 1:
            thisName += 'G%02d' % realization

        if doPrint:
            print("\tInitializing %s..." % thisName,
                  end = ' ',flush = True)
        ##############
        # PARAMETERS #
        ##############

        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisOptimAlg = optimAlg
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2

        ################
        # ARCHITECTURE #
        ################

        thisArchit = callArchit(**hParamsDict)
        thisArchit.to(thisDevice)

        #############
        # OPTIMIZER #
        #############

        if thisOptimAlg == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate,
                                   betas = (beta1, beta2))
        elif thisOptimAlg == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(),
                                  lr = learningRate)
        elif thisOptimAlg == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)

        ########
        # LOSS #
        ########

        thisLossFunction = lossFunction()
        
        ###########
        # TRAINER #
        ###########

        thisTrainer = trainer
        
        #############
        # EVALUATOR #
        #############

        thisEvaluator = evaluator

        #########
        # MODEL #
        #########

        modelCreated = model.Model(thisArchit,
                                   thisLossFunction,
                                   thisOptim,
                                   thisTrainer,
                                   thisEvaluator,
                                   thisDevice,
                                   thisName,
                                   saveDir)

        modelsGNN[thisName] = modelCreated

        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisOptimizationAlgorithm': thisOptimAlg,
                        'thisTrainer': thisTrainer,
                        'thisEvaluator': thisEvaluator,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})

        if doPrint:
            print("OK") 

    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################

    ############
    # TRAINING #
    ############

    print("")

    for thisModel in modelsGNN.keys():

        if doPrint:
            print("Training model %s..." % thisModel)
            
        for m in modelList:
            if m in thisModel:
                modelName = m

        thisTrainVars = modelsGNN[thisModel].train(data,
                                                   nEpochs,
                                                   batchSize,
                                                   **trainingOptsPerModel[m])

    # And we also need to save 'nBatch' but is the same for all models, so
    if doFigs:
        nBatches = thisTrainVars['nBatches']     


