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
import planner.low_level_planner.resolve as resolve

location_in_fov = object_localization.location_in_fov

openables = object_type.openables
sliceables = object_type.sliceables

set_default_tilt = move_camera.set_default_tilt
field_of_view = resolve.field_of_view

CONFUSIONS = eqc.CONFUSIONS_M

'''
def check_sliced(env, obj):
    if env.check_inventory()!=[]:
        print("Pick was successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        return True
    elif env.check_inventory()==[]:
        print("Pick was not successful !")
        return False
'''



    


def refined_slice(manip_action,targ_obj,refinement_rel,refinement_obj,env, numtries = 0, preactions = '', nudgexy = [0,0]): 

    #psedocode 
    '''
    #Follows very closesly to refined_pick 
    '''


    #1
    print("(manipulation_signatures.py -> refined_slice)")
    print("Trying this for ",numtries," time")
    o_manip_action = copy.copy(manip_action)
    o_targ_obj = copy.copy(targ_obj)
    o_refinement_rel = copy.copy(refinement_rel)
    o_refinement_obj = copy.copy(refinement_obj)

    
    field = field_of_view(env)
    targ_obj = targ_obj.split(',') #split entry into a list by commas
    ref_obj = refinement_obj.split(',')[0]
    print("ref object ",ref_obj)
    print("refinement_obj  ",refinement_obj)
    things2slice = sliceables(targ_obj) 
    print("Got things to slice ",things2slice)
    
    obj_vis = any([things2slice[0]+'|' in a for a in list(field.keys())])
    #obj_vis = things2slice[0]+'|' in field.keys()
    _, things2open = openables(ref_obj) 
    
    if not obj_vis:
        print("Warning ! object to slice is not visible, attempting to open something")
        #2
        container_object = things2open[0]
        print("Decided ",container_object," as the container object")
        #if preactions=='open': #objects inside microwave/fridge (sometimes wont explicitly tell to open and then slice, so open first)
        print("Want to open an object before picking")
        #env.step(dict({"action": "LookDown"}))
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()

        lf,mf,rf,_,cents = location_in_fov(env,mask_image,depth_image)


        all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
        for k in all_vis:
            if container_object+'|' in k:
                print("Will try to open the ",k, "before trying to slice")
                env.step(dict({'action': 'OpenObject', 'objectId': k}))



    #3
    #block kept as placeholder for slice confusions - eg 1 unique valid object told to slice, but another unique valid object that can also be sliced is the only object present in scene
    '''
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
    '''


    #4
    things2slice = sliceables(targ_obj)
    targ_obj = things2slice[0]
    manip_action = manip_action.split(',')
    relative = refinement_rel.split(',')
    ref_obj = refinement_obj.split(',')[0]

    act1 = manip_action[0]
    if act1=="slice":
        act1 = 'SliceObject'

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
            return refined_slice(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])

        if numtries==1: #nudge left right to expand field of view
            nx,ny = 0,0
            if nudgexy[0]==-1:
                env.step(dict({"action": "MoveRight",'forceAction': True}))
            env.step(dict({"action": "MoveRight",'forceAction': True}))
            if env.actuator_success():
                nx = 1
            return refined_slice(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])

        print("Could not find the object, rotating and trying to find it")
        if numtries>1 and numtries<=4: #keep rotating and trying to find the object
            nx,ny = 0,0
            if nudgexy[0]==1:
                env.step(dict({"action": "MoveLeft",'forceAction': True}))
                nx = 0
            env.step(dict({"action": "RotateLeft",'forceAction': True}))
            o_refinement_obj = ''
            return refined_slice(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,numtries = numtries+1, nudgexy = [nx,ny])
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