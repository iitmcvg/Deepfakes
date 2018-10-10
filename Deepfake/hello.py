import numpy as np
import os, sys
import argparse
from PIL import Image
from freeze_graph import freeze_graph
import tensorflow as tf
import time

from net import *
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
from custom_vgg16 import *


data_dict = loadWeightsData('./vgg16.npy')
