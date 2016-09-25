#-*- coding:utf-8 -*-

import os
import time
import pprint
import tensorflow as tf
from six.moves import range
from logging import getLogger

logger = getLogger(__name__)
pp = pprint.PrettyPrinter().pprint

def get_model_dir(config, exceptions=None):
  """根据设置的参数生成一个路径.
  这个路径将会作为存放训练数据的目录地址，所以不同的参数的训练数据必须保证路径唯一。
  """
  attrs = config.__dict__['__flags'] # 
  pp(attrs)

  keys = attrs.keys()
  keys.sort()
  keys.remove('env_name')
  keys = ['env_name'] + keys

  names = [config.env_name]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'  # os.path.join(path,*paths) 是一种特殊用法

def timeit(f):
  """time for a function. This is a decorator.
  """
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    logger.info("%s : %2.2f sec" % (f.__name__, end_time - start_time))
    return result
  return timed
