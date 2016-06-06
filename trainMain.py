'''
@author Ivehui
@time 2016/06/05
@function: train the agent
'''
import logging
import os, sys
import dqn
import parameters as pms
import gym
import numpy as np
import caffe
from skimage.transform import resize
from transition import Transition as Tran


def transfer(rgbImage, new_dims):
    im = np.dot(rgbImage[..., :3],
                [0.229, 0.587, 0.144])

    im_min, im_max = im.min(), im.max()
    if im_max > im_min:
        # skimage is fast but only understands {1,3} channel images
        # in [0, 1].
        im_std = (im - im_min) / (im_max - im_min)
        resized_std = resize(im_std, new_dims, order=1)
        resized_im = resized_std * (im_max - im_min) + im_min
    else:
        # the image is a constant -- avoid divide by 0
        ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                       dtype=np.float32)
        ret.fill(im_min)
        return ret
    return resized_im.astype(np.float32)

if __name__ == '__main__':
    # # if isDisplsy == 0: no image plot
    # isDisplay = 1

    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make(pms.gameName)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/DQN-' + pms.gameName
    env.monitor.start(outdir, force=True, seed=0)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = dqn.DqnAgent(env.action_space)
    tran = Tran(max_size=pms.bufferSize)
    caffe.set_mode_gpu()

    imageDim = np.array((pms.frameHeight,
                         pms.frameWidth))
    curFrame = np.zeros((pms.frameChannel,
                         pms.frameHeight,
                         pms.frameWidth))
    nextFrame = np.zeros((pms.frameChannel,
                          pms.frameHeight,
                          pms.frameWidth))

    testStep = 0
    update_step = 0

    for i in range(pms.episodeCount):
        rgbImage = env.reset()
        env.render()
        done = False
        for j in range(pms.frameChannel):
            curFrame[j, ...] = transfer(rgbImage, imageDim)

        while(done == False):
            eGreedy = max(pms.eGreedy, 1)
            actionNum = agent.act(curFrame, eGreedy)
            reward = 0
            for j in range(pms.frameChannel):
                if(done == False):
                    rgbImage, rewardTemp, done, _ = env.step(actionNum)
                    env.render()
                nextFrame[j, ...] = transfer(rgbImage, imageDim)
                reward += rewardTemp
            reward /= pms.frameChannel
            tran.saveTran(curFrame, actionNum, reward, done)
            curFrame = nextFrame.copy()
            testStep += 1

            # training
            overallSize = tran.getBufferSize()
            if overallSize > pms.startSize:
                selected = np.random.choice(overallSize - 1, pms.batchSize, replace=False)
                if tran.getIsFull():
                    selected = selected - overallSize + tran.getSize()
                # calculate the q_target
                agent.train(tran, selected)
                if update_step > pms.updateStep:
                    update_step = 0
                    agent.updateTarget()
                else:
                    update_step += 1

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    gym.upload(outdir)
