import caffe
import random
import parameters as pms
from environment import Environment as Envment
from transition import Transition as Tran
import numpy as np
import time
import os

actionSolver = None
actionSolver = caffe.get_solver(pms.actionSolverPath)
result_save = True
result_file = pms.result_file
for i in range(199, 200):
    trained_model = './models/action_iter_'+str(i)+'000.caffemodel'
    # trained_model = pms.new_model

    # copy params from caffemodel
    actionSolver.net.copy_from(trained_model)
    # test net share weights with train net
    actionSolver.test_nets[0].share_with(actionSolver.net)
    caffe.set_mode_gpu()

    success_cnt = 0
    for episode in range(50):
        envment = Envment(episode)
        enable = 1
        actionHistory = np.zeros((pms.history_num * pms.actionSize,))
        num = 0
        if result_save:
            folder = 'test/Images' + str(episode)
            os.makedirs(folder)
            image_file = './'+folder + '/' + str(num) + '-' + str(int(envment.curIOU*100)) + '.jpg'
            envment.show_image(file=image_file)
        while enable:
            curFrame = envment.get_curFrame()
            actionSolver.test_nets[0].blobs['frames'].data[...]\
                = curFrame.copy()
            actionSolver.test_nets[0].blobs['actionHistory'].data[...]\
                = actionHistory.copy()
            net_out = actionSolver.test_nets[0].forward()
            curAction = np.zeros(pms.actionSize)
            # print(net_out['value_q'][0])
            curAction[np.where(net_out['value_q'][0] == max(net_out['value_q'][0]))] = 1

            envment.take_action(curAction)
            curReward = envment.get_reward()
            actionHistory = np.hstack((curAction, actionHistory[pms.actionSize:]))

            num = num + 1
            if result_save:
                image_file = './' + folder + '/' + str(num) + '-' + str(int(envment.curIOU * 100)) + '.jpg'
                envment.show_image(file=image_file)
            # time.sleep(5)
            if curAction[pms.actionSize-1] == 1 or num > 30:
                enable = 0
                if curReward == pms.successReward:
                    success_cnt = success_cnt +1
        # print((i, episode + 1, success_cnt, float(success_cnt) / (episode + 1)))
    with open(result_file, 'a') as f:
        f.write("%d, %d, %d, %f" % (i, episode + 1, success_cnt, float(success_cnt)/(episode + 1)))
    print((i, episode+1, success_cnt, float(success_cnt)/(episode+1)))

    success_cnt = 0
    for episode in range(600, 700):
        envment = Envment(episode)
        enable = 1
        actionHistory = np.zeros((pms.history_num * pms.actionSize,))
        num = 0
        if result_save:
            folder = 'test/Images' + str(episode)
            os.makedirs(folder)
            image_file = './' + folder + '/' + str(num) + '-' + str(int(envment.curIOU * 100)) + '.jpg'
            envment.show_image(file=image_file)
        while enable:
            curFrame = envment.get_curFrame()
            actionSolver.test_nets[0].blobs['frames'].data[...] \
                = curFrame.copy()
            actionSolver.test_nets[0].blobs['actionHistory'].data[...] \
                = actionHistory.copy()
            net_out = actionSolver.test_nets[0].forward()
            curAction = np.zeros(pms.actionSize)
            # print(net_out['value_q'][0])
            curAction[np.where(net_out['value_q'][0] == max(net_out['value_q'][0]))] = 1

            envment.take_action(curAction)
            curReward = envment.get_reward()
            actionHistory = np.hstack((curAction, actionHistory[pms.actionSize:]))

            num = num + 1
            if result_save:
                image_file = './' + folder + '/' + str(num) + '-' + str(int(envment.curIOU * 100)) + '.jpg'
                envment.show_image(file=image_file)
            # time.sleep(5)e
            if curAction[pms.actionSize - 1] == 1 or num > 30:
                enable = 0
                if curReward == pms.successReward:
                    success_cnt = success_cnt + 1
        # print((i, episode + 1, success_cnt, float(success_cnt) / (episode + 1 - 600)))
    with open(result_file, 'a') as f:
        f.write(", %d, %d, %d, %f\n" % (i, episode + 1 - 600, success_cnt, float(success_cnt) / (episode + 1 - 600)))
    print((i, episode + 1 - 600, success_cnt, float(success_cnt) / (episode + 1 - 600)))



