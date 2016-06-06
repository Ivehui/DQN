'''
@author Ivehui
@time 2016/06/05
@function: reinforcement agent
'''
import random
import caffe
import parameters as pms
import numpy as np

class DqnAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

        actionSolver = None
        actionSolver = caffe.get_solver(pms.actionSolverPath)
        actionSolver.net.copy_from(pms.newModel)
        # test net share weights with train net
        actionSolver.test_nets[0].share_with(actionSolver.net)
        self.solver = actionSolver

        self.targetNet = caffe.Net(pms.actionTestNetPath, pms.newModel, caffe.TEST)

    def act(self, frame, greedy):
        if random.uniform(0, 1) < greedy:
            return self.action_space.sample()
        else:
            self.solver.test_nets[0].blobs['frames'].data[...] = frame.copy()
            netOut = self.solver.test_nets[0].forward()
            return np.where(netOut['value_q'][0] == max(netOut['value_q'][0]))

    def train(self, tran, selected):
        self.targetNet.blobs['frames'].data[...] \
            = tran.frames[selected + 1].copy()
        netOut = self.targetNet.forward()
        target = tran.reward[selected] \
                 + pms.discount \
                   * tran.n_lasr[selected] \
                   * netOut['value_q'].max(0)

        self.solver.net.blobs['target'].data[...] = target
        self.solver.net.blobs['frames'].data[...] = tran.frames[selected].copy()
        self.solver.net.blobs['filter'].data[...] = tran.action[selected].copy()
        self.solver.step(1)

    def updateTarget(self):
        for layer in pms.layers:
            self.targetNet.params[layer][0].data[...] \
                = self.targetNet.params[layer][0].data * (1 - pms.updateParam) + \
                  self.solver.net.params[layer][0].data * pms.updateParam
            self.targetNet.params[layer][1].data[...] \
                = self.targetNet.params[layer][1].data * (1 - pms.updateParam) + \
                  self.solver.net.params[layer][1].data * pms.updateParam

