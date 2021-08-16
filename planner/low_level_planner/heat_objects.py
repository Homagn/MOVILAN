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

import planner.low_level_planner.drawer_manipulation as drawer_manipulation


drawer_manipulation_place = drawer_manipulation.drawer_manipulation_place
drawer_manipulation_remove = drawer_manipulation.drawer_manipulation_remove


location_in_fov = object_localization.location_in_fov
opens_toggles = object_type.opens_toggles
toggle = handle_appliance.toggle


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



    


def cook(manip_action,targ_obj,refinement_rel,refinement_obj,env, numtries = 0, preactions = '', nudgexy = [0,0]): 

    #psedocode 
    '''
    #Follows very closesly to refined_pick 
    '''


    #1
    print("(manipulation_signatures.py -> cook)")
    print("Trying this for ",numtries," time")
    o_manip_action = copy.copy(manip_action)
    o_targ_obj = copy.copy(targ_obj)
    o_refinement_rel = copy.copy(refinement_rel)
    o_refinement_obj = copy.copy(refinement_obj)

    
    field = field_of_view(env)
    ref_obj = refinement_obj.split(',')[0]
    
    opto = opens_toggles(ref_obj) #things that opens and toggles like a microwave

    if opto!=[]:
        print("Found ",opto," which comprises of objects that opens and toggles ")
        env = drawer_manipulation_place("place,close",targ_obj,refinement_rel, refinement_obj, env) 
        env,warn = toggle(refinement_obj,"","turnon",env)
        env,warn = toggle(refinement_obj,"","turnoff",env)
        env = drawer_manipulation_remove("open,pick,close",targ_obj,refinement_rel, refinement_obj, env) 

    return env