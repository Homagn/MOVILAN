import os
import sys
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))
import planner.params as params

import numpy as np
import math

from planner.low_level_planner import object_localization
from language_understanding import equivalent_concepts as eqc

TEXTURES = eqc.TEXTURES

def target_visible(env, obj):
    #agent keeps rotating until it sees the segmented color corresponding to the object

    print("(visibility_check.py -> target_visible)")
    v = -1
    diff = [0,0]
    rot_step = 36
    rot_init = env.get_rotation()

    for i in range(10):
        env.custom_rotation(params.camera_horizon_angle, rot_step)

        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        _,_,_,areas,_ = object_localization.location_in_fov(env,mask_image,depth_image)

        key = ""
        for k in areas.keys():
            if obj+'|' in k:
                #print("Able to see the object ",k)
                key = k
                #print(k," has area ",areas[k])
            elif '|' in obj or obj in TEXTURES: #this means a resolve function has been called earlier to remove confusion about the object name and now we have the precise name
                # or its a textures object which is very tricky to find so need manual check with lists
                if obj in k:
                    #print("Able to see the object ",k)
                    key = k
                    #print(k," has area ",areas[k])
        

        r = env.get_rotation()
        #print("Rotation of the agent is ",r)
        if key!="":
            print("Target object ",key," is visible to the agent from this position")
            v = i
            ct = math.fabs(math.cos(math.radians(v*36)))
            st = math.fabs(math.sin(math.radians(v*36)))

            r = env.get_rotation()
            #event = env.step(dict({"action": "MoveAhead"}))
            #event = env.step(dict({"action": "TeleportFull","x": a1+2.0*st,"y": y,"z": c1+2.0*ct}))
            print("Rotation of the agent is ",r)
            if r>=0 and r<90:
                #facing direction of the agent is between 1 and 4
                print("Able to see the object between facing directions 1 and 4")
                st = math.cos(math.radians(r))
                ct = math.sin(math.radians(r))
                diff = [ct,st]
            if r>=90 and r<180:
                #facing direction of the agent is between 1 and 4
                print("Able to see the object between facing directions 3 and 4")
                r = r-90
                st = math.cos(math.radians(r))
                ct = math.sin(math.radians(r))
                diff = [st,-ct]
            if r>=180 and r<270:
                #facing direction of the agent is between 1 and 4
                print("Able to see the object between facing directions 2 and 3")
                st = math.cos(math.radians(r))
                ct = math.sin(math.radians(r))
                diff = [-ct,-st]
            if r>=270:
                #facing direction of the agent is between 1 and 4
                print("Able to see the object between facing directions 1 and 2")
                st = math.cos(math.radians(r))
                ct = math.sin(math.radians(r))
                diff = [-st,ct]



            break
    

    print("Got v ",v)

    print("Got diff ",diff)
    print("Resetting rotation of agent ")
    env.set_rotation(params.camera_horizon_angle, rot_init)


    return v, diff