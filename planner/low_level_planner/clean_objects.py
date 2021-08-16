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
import planner.low_level_planner.handle_appliance as handle_appliance

import planner.low_level_planner.place_objects as place_objects
import planner.low_level_planner.pick_objects as pick_objects


refined_place = place_objects.refined_place
refined_pick = pick_objects.refined_pick


location_in_fov = object_localization.location_in_fov
receps_toggles = object_type.receps_toggles
toggle = handle_appliance.toggle


set_default_tilt = move_camera.set_default_tilt
field_of_view = resolve.field_of_view

CONFUSIONS = eqc.CONFUSIONS_M
CONTEXTUALS = eqc.CONTEXTUALS
'''
def check_sliced(env, obj):
    if env.check_inventory()!=[]:
        print("Pick was successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        return True
    elif env.check_inventory()==[]:
        print("Pick was not successful !")
        return False
'''



    


def clean(manip_action,targ_obj,refinement_rel,refinement_obj,env, numtries = 0, preactions = '', nudgexy = [0,0]): 

    #psedocode 
    '''
    #Follows very closesly to refined_pick 
    '''


    #1
    print("(manipulation_signatures.py -> clean)")
    print("Trying this for ",numtries," time")
    o_manip_action = copy.copy(manip_action)
    o_targ_obj = copy.copy(targ_obj)
    o_refinement_rel = copy.copy(refinement_rel)
    o_refinement_obj = copy.copy(refinement_obj)

    
    field = field_of_view(env)
    ref_obj = refinement_obj.split(',')[0]
    
    rec, _ = receps_toggles(ref_obj) #things that opens and toggles like a microwave
    if rec[0] in list(CONTEXTUALS["toggle"].keys()):
        contextual_toggle = CONTEXTUALS["toggle"][rec[0]]

        print("Found ",rec," which comprises of objects where we can clean ")
        print("Found ",contextual_toggle," which comprises of objects we need to toggle contextual to complete clean operation ")
        
        env = refined_place("place",targ_obj,refinement_rel, refinement_obj, env) 

        #event = env.step(dict({"action": "MoveBack"}))
        #act = input("Enter the action ")
        #objid = input("Enter the object_id ")
        #event = env.step(dict({"action": act, "objectId": objid}))
        
        env,warn = toggle(contextual_toggle,"","turnon",env)
        env,warn = toggle(contextual_toggle,"","turnoff",env)
        env = refined_pick("pick",targ_obj,refinement_rel, refinement_obj, env) 

    return env