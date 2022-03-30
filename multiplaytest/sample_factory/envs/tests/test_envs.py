import os
import time
import unittest
from os.path import join
from unittest import TestCase

from sample_factory.algorithms.utils.algo_utils import num_env_steps
from sample_factory.algorithms.utils.arguments import default_cfg

from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict


def default_doom_cfg():
    return default_cfg(env='doom_env')
