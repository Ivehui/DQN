'''
@author Ivehui
@time 2016/06/05
@function save tuple data
'''


import numpy as np
import parameters as pms


class Transition(object):
    def __init__(self, max_size=10000):
        self.size = 0
        self.max_size = int(max_size)
        self.isFull = False

        self.frames = np.zeros((max_size, pms.frameChannel,
                                pms.frameHeight, pms.frameWidth))
        self.action = np.zeros((max_size, pms.actionSize))
        self.reward = np.zeros((max_size, 1))
        self.n_last = np.zeros((max_size, 1))

    def saveTran(self, frame, actionNum, reward, done):
        action = np.zeros(pms.actionSize)
        action[actionNum] = 1
        if self.size < self.max_size:
            self.frames[self.size] = frame
            self.action[self.size] = action
            self.reward[self.size] = reward
            self.n_last[self.size] = 0 if done else 1
            self.size += 1
        else:
            self.frames[0] = frame
            self.action[0] = action
            self.reward[0] = reward
            self.n_last[0] = 0 if done else 1
            self.isFull = True
            self.size = 1

    def getBufferSize(self):
        if self.isFull:
            return self.max_size
        else:
            return self.size

    def getSize(self):
        return self.size

    def getIsFull(self):
        return self.isFull

    def getCurActionHistory(self):
        return self.curActionHistory
