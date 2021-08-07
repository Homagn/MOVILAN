import numpy as np
import sys
import os
import copy
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from language_understanding import equivalent_concepts as eqc
import planner.low_level_planner.object_localization as object_localization
import planner.low_level_planner.object_type as object_type
import planner.low_level_planner.move_camera as move_camera

location_in_fov = object_localization.location_in_fov
openables = object_type.openables
set_default_tilt = move_camera.set_default_tilt

CONFUSIONS = eqc.CONFUSIONS


def check_pick(env):
    if env.check_inventory()!=[]:
        print("Pick was successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        return True
    elif env.check_inventory()==[]:
        print("Pick was not successful !")
        return False

def refined_pick(manip_action,targ_obj,refinement_rel,refinement_obj,env, numtries = 0, preactions = '', nudgexy = [0,0]): 

    #psedocode 
    '''
    1. Store initial arguments to facilitate backtrack during recursion
    2. Decide whether to close an object before picking it up - eg laptop
    3. Disambiguate what is the object to be picked from what is the object from top of which to pick/ 
        also resolve common ambiguities with object naming such as mug/cup, plate/bowl, etc based on what is visible 
    4. Decide a preferrential order (look-costs) to search for the object to pick up 
        because the field of view of agent is limited, it needs to tilt its head up and down and also focus on 
        left/right, top-left/top-right, etc regions of the visible image
    5. Since there is a possibility of the object (to pick up) appearing in several regions of the image
        (a table might have multiple mugs placed on top of it), but the user may specify the relative location of the mug to pick up
        so assigning "visual-distances" that differentiate these same category pickup objects
    6. Now if its said pick up the lower left mug or mug to the right, the agent can understand based on visual-distances
        by aligning visual-distances with look-costs (the preferential order obtained in 4)
    7. There is a chance that the object misses out the field of view of the agent, so nudge left right, 
        rotate left right, and each time start fresh from step 1 using recursion
    8. Sometimes object visible but still cannot pick up due to lack of precise positioning, so try to rotate 
    '''


    #1
    print("(manipulation_signatures.py -> refined_pick)")
    print("Trying this for ",numtries," time")
    o_manip_action = copy.copy(manip_action)
    o_targ_obj = copy.copy(targ_obj)
    o_refinement_rel = copy.copy(refinement_rel)
    o_refinement_obj = copy.copy(refinement_obj)

    #2
    if preactions=='close': #currently this will close all objects of similar type that are visible
        print("Want to close an object before picking")
        env.step(dict({"action": "LookDown"}))
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()

        lf,mf,rf,_,cents = location_in_fov(env,mask_image,depth_image)
        #again as usual doing a search for visible objects matching with the target name
        #bacause closeobject operation has to be applied on that precise name
        '''
        for k in lf.keys():
            if targ_obj+'|' in k:
                print("Will try to close the ",k)
                event = env.step(dict({'action': 'CloseObject', 'objectId': k}))
        for k in mf.keys():
            if targ_obj+'|' in k:
                print("Will try to close the ",k)
                event = env.step(dict({'action': 'CloseObject', 'objectId': k}))
        for k in rf.keys():
            if targ_obj+'|' in k:
                print("Will try to close the ",k)
                event = env.step(dict({'action': 'CloseObject', 'objectId': k}))
        '''

        all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
        for k in all_vis:
            if targ_obj+'|' in k:
                print("Will try to close the ",k)
                env.step(dict({'action': 'CloseObject', 'objectId': k}))

        env.step(dict({"action": "LookUp"}))


    #3
    targ_obj = targ_obj.split(',') #split entry into a list by commas
    things2pick, _ = openables(targ_obj) 

    #sometimes user may confuse Mug as a cup and order to pick up the cup which is not visible in scene
    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
    all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
    small = copy.copy(things2pick)
    n_small = copy.copy(small)
    for sm in small:
        if sm in CONFUSIONS.keys():
            print("Possible confused object, all visibles are ", all_vis)
            csma = [CONFUSIONS[sm]+'|' in a for a in all_vis]
            sma = [sm+'|' not in a for a in all_vis]
            if any(csma) and all(sma):
                print(CONFUSIONS[sm]," is visible but ",sm," is not visible so replacing as a related object")
                n_small[n_small.index(sm)] = CONFUSIONS[sm]
    small = n_small
    things2pick = copy.copy(small)


    #4
    targ_obj = things2pick[0]
    manip_action = manip_action.split(',')
    relative = refinement_rel.split(',')
    ref_obj = refinement_obj.split(',')[0]

    act1 = manip_action[0]
    if act1=="pick":
        act1 = 'PickupObject'
    if act1=="place":
        act1 = 'PutObject'
    env.step(dict({"action": "LookUp"}))
    env.step(dict({"action": "LookUp"}))

    look_cost = [0,0] #mid, mid of field of vision by default
    for r in relative:
        if r =='up':
            look_cost[0] = -1000
        if r =='mid':
            look_cost[0] = 0
        if r =='bottom':
            look_cost[0] = 1000
        
        if r=='left':
            look_cost[1] = -1000
        if r=='mid':
            look_cost[1] = 0
        if r=='right':
            look_cost[1] = 1000

    #5
    visual_distances = {}
    for i in range(3): #perform a basic 3 grid search for visible objects around the agent
        env.step(dict({"action": "LookDown"}))
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,_,cents = location_in_fov(env,mask_image,depth_image)
        
        for k in lf.keys():
            
            if targ_obj+'|' in k:
                print(k)
                for kk in lf.keys():
                    checks = [r +'|' in kk for r in ref_obj]
                    if any(checks) or numtries>0: #ref_obj=='' or ref_obj in kk:
                        if ref_obj!="":
                            print("Found ",targ_obj," in ",k," which satisfies neighborhood ",ref_obj)
                        visual_distances[k] =[cents[k][0]-150+(i-1)*100, cents[k][1]-150 - 100] #150,150 is the center pixel coordinates of image
                        
                
        for k in mf.keys():
            
            if targ_obj+'|' in k:
                print(k)
                for kk in mf.keys():
                    checks = [r +'|' in kk for r in ref_obj]
                    if any(checks) or numtries>0: #ref_obj=='' or ref_obj in kk:
                        if ref_obj!="":
                            print("found ",targ_obj," in ",k," which satisfies neighborhood ",ref_obj)
                        visual_distances[k] =[cents[k][0]-150+(i-1)*100, cents[k][1]-150 ]
        
        for k in rf.keys():
            
            if targ_obj+'|' in k:
                print(k)
                for kk in rf.keys():
                    checks = [r +'|' in kk for r in ref_obj]
                    if any(checks) or numtries>0: #ref_obj=='' or ref_obj in kk:
                        if ref_obj!="":
                            print("found ",targ_obj," in ",k," which satisfies neighborhood ",ref_obj)
                        visual_distances[k] =[cents[k][0]-150+(i-1)*100, cents[k][1]-150 + 100]
    
    env.step(dict({"action": "LookUp"}))


    #6
    print("Visual_distances ",visual_distances)
    minkey = ""
    minv = 10000000
    for k in visual_distances.keys():
        c = visual_distances[k]
        d = (c[0]-look_cost[0])**2 + (c[1]-look_cost[1])**2
        if d<minv:
            minv=d
            minkey = k
    if minkey!="":
        print("Found the suggested object ",minkey)
    
    #7
    if minkey=="":
        print("Could not find the object, nudging left right and trying to find it")
        #if numtries<=4: #keep rotating and trying to find the object
        if numtries==0: #nudge left right to expand field of view
            env.step(dict({"action": "MoveLeft",'forceAction': True}))
            nx,ny = 0,0
            if env.actuator_success():
                nx = -1
            return refined_pick(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])

        if numtries==1: #nudge left right to expand field of view
            nx,ny = 0,0
            if nudgexy[0]==-1:
                env.step(dict({"action": "MoveRight",'forceAction': True}))
            env.step(dict({"action": "MoveRight",'forceAction': True}))
            if env.actuator_success():
                nx = 1
            return refined_pick(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])

        print("Could not find the object, rotating and trying to find it")
        if numtries>1 and numtries<=4: #keep rotating and trying to find the object
            nx,ny = 0,0
            if nudgexy[0]==1:
                env.step(dict({"action": "MoveLeft",'forceAction': True}))
                nx = 0
            env.step(dict({"action": "RotateLeft",'forceAction': True}))
            o_refinement_obj = ''
            return refined_pick(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])
        else:
            return env

    env.step(dict({'action': act1, 'objectId': minkey}))

    
    #8

    if not env.actuator_success(): #although found the object picking it up may still require presice positioning
        print("Found object but still need precise positioning to pick it up ")
        for _ in range(4):
            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))
            for _ in range(4):
                env.step(dict({'action': act1, 'objectId': minkey}))
                if check_pick(env):
                    env = set_default_tilt(env)
                    return env, event
                env.step(dict({"action": "LookDown"}))

            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))

            env.step(dict({"action": "RotateLeft",'forceAction': True}))

        #env,event = set_default_tilt(env,event)
        #return env, event

    
    return env