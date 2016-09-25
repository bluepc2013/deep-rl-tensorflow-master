#-*- coding:utf-8 -*-

import gym
import random
import logging
import numpy as np
import pdb

from .corridor import CorridorEnv

try:
  import scipy.misc
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave
except:
  import cv2
  imresize = cv2.resize
  imwrite = cv2.imwrite

logger = logging.getLogger(__name__)

class Environment(object):
  def __init__(self, env_name, n_action_repeat, max_random_start,
               observation_dims, data_format, display):
    self.env = gym.make(env_name)

    self.n_action_repeat = n_action_repeat # 在两步间有很多帧，这里指示在步间行动被重复执行的次数
    self.max_random_start = max_random_start # 当选择任意前进一些步时，最大允许前进的步数
    self.action_size = self.env.action_space.n

    self.display = display
    self.data_format = data_format # NHWC(CPU) or NCHW(GPU)
    self.observation_dims = observation_dims

    if hasattr(self.env, 'get_action_meanings'):
      logger.info("Using %d actions : %s" % (self.action_size, ", ".join(self.env.get_action_meanings())))

  def new_game(self):
    return self.preprocess(self.env.reset()), 0, False

  def new_random_game(self):
    return self.new_game()

  def step(self, action, is_training=False):
    observation, reward, terminal, info = self.env.step(action)
    if self.display: self.env.render()
    return self.preprocess(observation), reward, terminal, info

  def preprocess(self):
    """This is a abstract method.
    This is a trick for define a abstract method in python. this function must be 
    override, or it will raise a Exception.
    """
    raise NotImplementedError()

class ToyEnvironment(Environment):
  def preprocess(self, obs):
    """ ?
    """
    new_obs = np.zeros([self.env.observation_space.n])
    new_obs[obs] = 1
    return new_obs

class AtariEnvironment(Environment):
  def __init__(self, env_name, n_action_repeat, max_random_start,
               observation_dims, data_format, display):
    super(AtariEnvironment, self).__init__(env_name, 
        n_action_repeat, max_random_start, observation_dims, data_format, display)

  def new_game(self, from_random_game=False):
    screen = self.env.reset()
    screen, reward, terminal, _ = self.env.step(0)

    if self.display:
      self.env.render()

    if from_random_game:
      return screen, 0, False
    else:
      self.lives = self.env.ale.lives()
      terminal = False
      return self.preprocess(screen, terminal), 0, terminal


  def new_random_game(self):
    screen, reward, terminal = self.new_game(True)

    for idx in range(random.randrange(self.max_random_start)):
      screen, reward, terminal, _ = self.env.step(0)

      if terminal: logger.warning("warning: terminal signal received after %d 0-steps", idx)

    if self.display:
      self.env.render()

    self.lives = self.env.ale.lives()

    terminal = False
    return self.preprocess(screen, terminal), 0, terminal


  def step(self, action, is_training):
    """ 采取一次行动并返回执行结果
    """
    if action == -1:
      # Step with random action
      action = self.env.action_space.sample()

    cumulated_reward = 0

    # 一次行动持续n_action_repeat帧，有死亡事件时除外
    for _ in range(self.n_action_repeat):
      screen, reward, terminal, _ = self.env.step(action)
      cumulated_reward += reward
      current_lives = self.env.ale.lives()

      # 如果遇到死亡事件，则终止本次行动
      if is_training and self.lives > current_lives:
        terminal = True

      if terminal: break

    # 每隔n_action_repeat帧刷新一次图像
    if self.display:
      self.env.render()

    if not terminal:
      self.lives = current_lives

    return self.preprocess(screen, terminal), reward, terminal, {}


  def preprocess(self, raw_screen, terminal):
    """image preprocess.
    """
    # merge three channels to build a new one.
    y = 0.2126 * raw_screen[:, :, 0] + 0.7152 * raw_screen[:, :, 1] + 0.0722 * raw_screen[:, :, 2]
    y = y.astype(np.uint8) # cast int64 to uint8,
    y_screen = imresize(y, self.observation_dims)
    return y_screen


