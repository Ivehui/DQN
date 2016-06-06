'''
@author Ivehui
@time 2016/05/18
@function copy caffenet model to other net
'''

import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2
import parameters as pms
import random

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=1, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def overall_net(batch_size, channels, height, width, action_size, net_type):

    # param = learned_param
    n=caffe.NetSpec()
    # action
    n.frames = L.Input(shape=dict(dim=[batch_size, channels, height, width]))

    # Image feature
    if net_type == 'action':
        param = learned_param
    else:
        param = frozen_param

    n.conv1, n.relu1 = conv_relu(n.frames, 8, 32, stride=4, param=param)
    n.conv2, n.relu2 = conv_relu(n.relu1, 4, 64, stride=2, param=param)
    n.conv3, n.relu3 = conv_relu(n.relu2, 3, 64, stride=1, param=param)
    n.fc4, n.relu4 = fc_relu(n.relu3, 512, param=param)

    n.value_q = L.InnerProduct(n.relu4, num_output=action_size, param=param,
                               weight_filler=dict(type='gaussian', std=0.005),
                               bias_filler=dict(type='constant', value=1))

    if net_type == 'test':
        return n.to_proto()

    n.filter = L.Input(shape=dict(dim=[batch_size, action_size]))
    # operation 0: PROD
    n.filtered_value_q = L.Eltwise(n.value_q, n.filter, operation=0)

    n.target = L.Input(shape=dict(dim=[batch_size, action_size]))

    n.loss = L.EuclideanLoss(n.filtered_value_q, n.target)

    return n.to_proto()

### define solver
def solver(train_net_path, net_type, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = pms.maxIter  # Test after every 1000 training iterations.
        s.test_iter.append(1)  # Test on 1 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    s.max_iter = pms.maxIter  # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'fixed'

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.95
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 100000
    s.snapshot_prefix = 'models/'+net_type

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    return s

# Write the solver to a temporary file and return its filename.

with open(pms.actionTestNetPath, 'w') as f:
    f.write('name: "action test net"\n')
    f.write(str(overall_net(pms.batchSize, pms.frameChannel, pms.frameHeight,
                            pms.frameWidth, pms.actionSize, 'test')))

with open(pms.actionTrainNetPath, 'w') as f:
    f.write('name: "action train net"\n')
    f.write(str(overall_net(pms.batchSize, pms.frameChannel, pms.frameHeight,
                            pms.frameWidth, pms.actionSize, 'action')))

with open(pms.actionSolverPath, 'w') as f:
    f.write(str(solver(pms.actionTrainNetPath, 'action',
                       test_net_path=pms.actionTestNetPath,
                       base_lr=pms.baseLr)))


# create new net
random.seed(0)
newNet = caffe.Net(pms.actionTrainNetPath, caffe.TEST)
# save the model, weights of last 3 layers are random init
newNet.save(pms.newModel)