# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:44:40 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import style
style.use('ggplot')

"""
Setting working device to CPU
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'