import os
import sys
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))
import planner.params as params

import numpy as np
import cv2
import copy
import math
import random
from skimage.measure import regionprops, label
import time as t
import argparse

#all submodules in this folder
import planner.low_level_planner.grid_planning as grid_planning
import planner.low_level_planner.visibility_check as visibility_check
import planner.low_level_planner.object_localization as object_localization
import planner.low_level_planner.astar_search as astar_search
import planner.low_level_planner.refinement as refinement
import planner.low_level_planner.exploration as exploration
import planner.low_level_planner.resolve as resolve


occupancy_grid = grid_planning.occupancy_grid #occupancy grid function from grid_planning
target_visible = visibility_check.target_visible
location_in_fov = object_localization.location_in_fov
target_visible = visibility_check.target_visible
graph_search = astar_search.search
unit_refinement = refinement.unit_refinement
nudge = refinement.nudge
random_explore = exploration.random_explore
resolve_confusions = resolve.resolve_confusions
resolve_refinement = resolve.resolve_refinement