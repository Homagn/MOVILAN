import numpy as np
from skimage.measure import regionprops, label
import copy
import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

import planner.low_level_planner.object_localization as object_localization
import planner.low_level_planner.object_type as object_type
from planner.low_level_planner import move_camera

toggleables = object_type.toggleables
location_in_fov = object_localization.location_in_fov

def toggle(target_object,refinement_obj,action,env, numtries = 0):
    print("(manipulation_signatures.py -> toggle)")
    #move_camera.set_default_tilt(env)
    print("Trying this for ",numtries," time")
    if action=="turnon":
        action='ToggleObjectOn'
    if action=="turnoff":
        action='ToggleObjectOff'

    o_target_object = copy.copy(target_object)
    o_refinement_obj = copy.copy(refinement_obj)
    o_action = copy.copy(action)


    obj = ""
    t = target_object.split(',')
    r = refinement_obj.split(',')
    target_object = toggleables(t)

    if target_object!=[]:
        target_object = toggleables(t)[0]
    else:
        target_object = toggleables(t+r)[0]
    print("Mentioned toggleable object ",target_object)

    print("Tilting head all the way up and scanning top to bottom")
    pans = 0
    for i in range(4): # was 3 earlier, made 4 to increase fov !!!!!!!!!!!!!!!!
        env.step(dict({"action": "LookUp",'forceAction': True}))
        if env.actuator_success():
            #print("looking upwards success ")
            pans+=1

    #pans = 3
    cpans = 0
    env_toggle_error = False
    for i in range(pans+2): #extra +1 to look completely down towards the floor !!!!!!!!!!!!!!!
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,areas,_ = location_in_fov(env, mask_image, depth_image)
        all_keys = list(lf.keys())+list(mf.keys())+list(rf.keys())

        for k in all_keys:
            k1 = target_object
            if k1+'|' in k:
                print("Exact object iD for ",k1," is ",k)
                obj = k
                break
        
        #three things are problem in simulator
        #1. Even if the lamp turns on, the agent has the see the lamp turned on in the final image other it will think failure
        #2. Lamp can be turned on only when its visible to the agent (big objects unnecessarily block fov)
        #3. turnOn command even when objectID= "" can sometimes turn on possible objects that can be turned on

        if obj!="": #!!!!!!!!!!!
            print("Trying to ",action," ",obj)
            #event = env.step(dict({"action": "MoveAhead"}))
            try:
                env.step(dict({'action': action, 'objectId': obj, 'forceAction': True}))
                #env.step(dict({'action': action, 'objectId': obj}))
            except:
                print("Env returned error in toggle")
                env_toggle_error = True
            #env.step(dict({'action': action, 'objectId': obj}))
            if env.actuator_success() and env_toggle_error==False:# and obj!="":  # !!!!!!!!!!!!!!!!!!!!!
                #print("Need to restore original pan")
                return env,False
                '''
                for i in range(cpans,pans):
                    print("restoring original pan ")
                    event = env.step(dict({"action": "LookDown",'forceAction': True}))
                return env,event,False
                '''
            #break
        env.step(dict({"action": "LookDown",'forceAction': True}))
        cpans+=1
    
    

    #try to turn it on for the last time
    #event = env.step(dict({'action': action, 'objectId': obj, 'forceAction': True}))

    if not env.actuator_success() or obj=="":
        print("WARNING! failed to turn on a light source")
        if numtries<4:
            print("Rotating and trying to align to the toggleable object ")
            env.step(dict({"action": "RotateLeft",'forceAction': True}))
            return toggle(o_target_object, o_refinement_obj, o_action,env,numtries = numtries+1)
            #return toggle(o_target_object, o_action,env,event,numtries = numtries+1)

        warn = True
    else:
        warn = False
    return env,warn