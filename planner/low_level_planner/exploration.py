import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

import planner.low_level_planner.visibility_check as visibility_check
target_visible = visibility_check.target_visible

def random_explore(env, targ_obj, localize_params):
    print("(exploration.py -> random_explore)")
    goal_vis = False
    grid = []
    face_grids = []

    for _ in range(4):
        forwards = 0
        for _ in range(4):#random exploration 4 steps forward
            env.step(dict({"action": "MoveAhead"}))
            if env.actuator_success():
                forwards+=1

        g,f = [],[]
        try:
            g,f = occupancy_grid(targ_obj, localize_params)
        except:#target may be too far for depth map
            v, diff = target_visible(env, targ_obj)
            if v!=-1:#target object is not in map but is visible far away
                grid, face_grids = occupancy_grid(targ_obj, localize_params, hallucinate = diff)
                env = search(grid,face_grids, env, targ_obj, localize_params)
                #just make sure agent knows its position perfectly before planning
                #after one round of hallucination agent might have goten close enough to get the actual map
                grid, face_grids = occupancy_grid(targ_obj, localize_params)
                goal_vis = True
        if goal_vis:
            break

        if f!=[]:
            grid, face_grids = occupancy_grid(targ_obj, localize_params)
            goal_vis = True
            break
        
        for _ in range(forwards): #backtrack
            env.step(dict({"action": "MoveBack"}))
        env.step(dict({"action": "RotateLeft",'forceAction': True}))
    return goal_vis, env, grid, face_grids