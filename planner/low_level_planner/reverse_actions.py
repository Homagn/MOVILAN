import os
import sys
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

import numpy as np
import math



def find_reverse(action):
    if action=="OpenObject":
        return "CloseObject"
    if action=="CloseObject":
        return "OpenObject"


