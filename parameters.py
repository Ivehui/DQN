'''
@author Ivehui
@time 2016/06/05
@function: the parameters of the agent
'''


# game name
gameName = 'Breakout-v0'

# net directary
actionTestNetPath = 'models/action_test_net.prototxt'
actionTrainNetPath = 'models/action_train_net.prototxt'

# solver directory
actionSolverPath = 'models/action_solver.prototxt'

# models
newModel = './models/random_model.caffemodel'

# # mean file
# mean_file = './models/ilsvrc_2012_mean.npy'
# # test result file
# result_file = './models/result.txt'

# net input --- frame
batchSize = 32
frameChannel = 4
frameHeight = 84
frameWidth = 84
# buffer to save transition
bufferSize = 100000
startSize = 50000
#
baseLr = 0.00025
# parameter--update the target network
updateParam = 1e-3
# discount factor
discount = 0.99
# max_iter
maxIter = 10000000
# episode_count
episodeCount = 200000
# episode_count
episodeTestCount = 10
# size
actionSize = 6

# layer name
layers = ('conv1', 'conv2', 'conv3', 'fc4', 'value_q')
# update_layer = ('fc7n', 'fc8n', 'value_q')

# e greedy to choose the way getting action
eGreedyFinal = 0.1
finalNum = 100000
# alpha: radio of every action
tou = 0.99
updateStep = 1000
# net update parameter
updateParam = 1
# train number in every step
trainNum = 1
