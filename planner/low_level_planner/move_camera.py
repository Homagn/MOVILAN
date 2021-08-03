import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from planner import params

def set_default_tilt(env):
    #set the vertical head tilt to default

    #x,y,z = env.get_position()
    rot = env.get_rotation()
    env.set_rotation(params.camera_horizon_angle, rot)


    #done setting vertical head tilt to default
    return env