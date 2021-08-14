import numpy as np
from skimage.measure import regionprops, label
import copy
import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from mapper import test_gcn
from planner import params

import planner.low_level_planner.object_localization as object_localization
import planner.low_level_planner.move_camera as move_camera
import planner.low_level_planner.resolve as resolve

set_default_tilt = move_camera.set_default_tilt 
field_of_view = resolve.field_of_view
# sometimes the robot cannot ind the receptacle object in its field of view so swivel and search
def swivel_search(env,obj):

    env.custom_rotation(params.camera_horizon_angle,30)
    field = field_of_view(env)
    print("swivel searching for object ",obj)
    print("swivel search got field ",field)
    #obj_vis = obj+'|' in field.keys()
    obj_vis = any([obj+'|' in a for a in list(field.keys())])
    if obj_vis:
        print("Found ",obj," in swivel left ")
        env.custom_rotation(params.camera_horizon_angle,-30)
        event = env.step(dict({"action": "MoveLeft", 'forceAction': True}))
    else:
        env.custom_rotation(params.camera_horizon_angle,-30)
        obj_vis = obj+'|' in field.keys()
        if obj_vis:
            print("Found ",obj," in swivel right ")
            env.custom_rotation(params.camera_horizon_angle,30)
            event = env.step(dict({"action": "MoveRight", 'forceAction': True}))
        else:
            return env #object was not visible in either swivel leftt or right

    #if after slight movement left/right the object again recedes out of view, recursively swivel again
    field = field_of_view(env)
    #obj_vis = obj+'|' in field.keys()
    obj_vis = any([obj+'|' in a for a in list(field.keys())])
    if not obj_vis:
        return swivel_search(env,obj)
    else:
        return env
    


def nudge(areas, key, env, act1,act2, rollback = False):
    #NOTE!
    #In this case we are not adding steps to the entire trajectory in the algorithm
    #This is because this is an observable target object area maximization step
    #Agent takes a step in the direction which will maximize observed area of target object
    #This can be implemented much more efficiently with object tracking models that wont require the agent to brute force as in this code below
    
    change = 0
    #count_step is addtional functionality by me which can disable number of steps counter towards the entire trajectory
    env.step(dict({"action": act1, 'forceAction': True}), count_step = False) 
    change+=1

    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    _,_,_,areas_new,_ = object_localization.location_in_fov(env,mask_image,depth_image)

    if key not in areas_new.keys():
        areas_new[key] = 0
    if areas_new[key]<=areas[key]:
        env.step(dict({"action": act2, 'forceAction': True}), count_step = False)

        env.step(dict({"action": act2, 'forceAction': True}), count_step = False)
        change-=2

        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        _,_,_,areas_new,_ = object_localization.location_in_fov(env,mask_image,depth_image)

        if key not in areas_new.keys():
            areas_new[key] = 0
        if areas_new[key]<=areas[key]:
            env.step(dict({"action": act1, 'forceAction': True}), count_step = False)
            change+=1
        else:
            print("nudged ",act2)
    else:
        print("nudged ",act1)
    
    if rollback: #cancel nudge refinement
        if change==-1:
            env.step(dict({"action": act1, 'forceAction': True}), count_step = False)
        if change==1:
            env.step(dict({"action": act2, 'forceAction': True}), count_step = False)

    return env

def unit_refinement(env, obj, numtry = 0): 
    print("refinement.py -> unit_refinement")
    print("Trying this for the ",numtry," time")
    #on top of final navigation using graph search , brute force search over the unit grid of the last grid in the path
    #add a final translation and final rotation that increases the area of the detected target object in segmentation image
    max_area = 0
    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    _,_,_,areas,_ = object_localization.location_in_fov(env,mask_image,depth_image) #HERE/ face_touch = object_localization.location_in_fov
    
    key = ""
    for k in areas.keys():
        if obj+'|' in k:
            print("Able to see the object ",k)
            key = k
            print(k," has area ",areas[k])
        elif '|' in obj: #this means a resolve function has been called earlier to remove confusion about the object name and now we have the precise name
            if obj in k:
                print("Able to see the object ",k)
                key = k
                print(k," has area ",areas[k])
    if key=="":
        print("Object went out of sight trying to realign")
        
        #event = env.step(dict({"action": "RotateLeft"}))
        #t.sleep(2)
        #event = env.step(dict({"action": "RotateRight"}))
        
        #write a function for spiralling upwards camera view formation and put that function in move_camera.py
        if numtry<=3:
            env.step(dict({"action": "RotateLeft", 'forceAction': True}))
            return unit_refinement(env, obj, numtry = numtry+1)
        elif numtry>3 and numtry<6:
            env.step(dict({"action": "LookUp", 'forceAction': True}))
            return unit_refinement(env, obj, numtry = numtry+1)
        elif numtry==6:
            env = set_default_tilt(env)
            return unit_refinement(env, obj, numtry = numtry+1)
        elif numtry>6 and numtry<9:
            env.step(dict({"action": "LookDown", 'forceAction': True}))
            return unit_refinement(env, obj, numtry = numtry+1)
        
        else:
            env = set_default_tilt(env)
            return env
        




    env = nudge(areas, key, env, "MoveLeft","MoveRight")
    env = nudge(areas, key, env, "MoveAhead","MoveBack")
    env = nudge(areas, key, env, "RotateLeft","RotateRight")
    env = nudge(areas, key, env, "LookUp","LookDown")
    
    return env