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
newModel = './models/new_model.caffemodel'

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
bufferSize = 10000
startSize = batchSize * 3
#
baseLr = 0.00025
# parameter--update the target network
updateParam = 1e-3
# discount factor
discount = 0.99
# max_iter
maxIter = 10000000
# episode_count
episodeCount = 500
# size
actionSize = 6

# layer name
layers = ('conv1', 'conv2', 'conv3', 'fc4', 'value_q')
# update_layer = ('fc7n', 'fc8n', 'value_q')

# e greedy to choose the way getting action
eGreedy = 0.1
# alpha: radio of every action
alpha = 0.1
tou = 0.6
updateStep = 10
# net update parameter
updateParam = 1e-3
# reward setting
successReward = 3
stepReward = 1
