#This code designed to fetch all the trajectory data from the alfred folder in an organized fashion
import os
from os import path
import sys
#os.environ['ALFRED_ROOT'] = '/home/hom/alfred'
os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
from robot.sensing import sensing
from language_understanding import equivalent_concepts as eqc
from language_understanding import parse_funcs as pf

from planner.low_level_planner import navigation_signatures as ns
from planner.low_level_planner import manipulation_signatures as ms

import planner.low_level_planner.resolve as resolve
field_of_view = resolve.field_of_view

from planner import params

import json
import csv
import glob
import os
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
import time as t
from skimage.measure import regionprops, label
from word2number import w2n
import math



#from low_level_planner import manipulation_signatures as ms
#from low_level_planner import navigation_signatures as ns

from language_understanding import naming 
from mapper import gtmaps as gtm
import traceback



CARRY = eqc.CARRY
PICK = eqc.PICK # ! Get can also be used as a pointer for navigation- eg - get to the desk near bed
WALK = eqc.WALK
WALK_ST = eqc.WALK_ST
INVEN_OBJS = eqc.INVEN_OBJS

replace_missing_slots = pf.replace_missing_slots #is a helper language parsing function





def writelog(fname, values):
    file1 = open(fname, 'a')
    writer = csv.writer(file1)
    fields1=values #is a list
    writer.writerow(fields1)
    file1.close()


def get_demo_traj_length(r,t,tr):
    traj_file = get_file(rn = r, task_index = t, trial_num = tr)
    
    
    #extracting the expert trajectory length in this block
    try:
        with open(traj_file[0]) as f:
            traj_data = json.load(f)

        trajectory = traj_data['plan']['low_actions']
        Cmd = []
        for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
            # next cmd under the current hl_action
            cmd = ll_action['api_action']
            #because we are assuming panorama image
            if cmd['action']!='RotateRight' and cmd['action']!='RotateLeft' and cmd['action']!='LookUp' and cmd['action']!='LookDown':
                Cmd.append(cmd)
        
        return len(cmd)
    except:
        print("Error encountered in reading expert actions !")
        return -1


def other_side_attempt(GRIDS,FACE_GRIDS,target_obj,ref_obj,localize_params, env):
    print("(high_level_planner.py -> other_side_attempt)")
    print("Since pick/place was not succeful will try to navigate to a better position ")
    #try to nudge away from the target object to cancel the effect of unit refinement earlier
    env.step(dict({"action": "MoveBack", "moveMagnitude" : 0.25}))
    env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
    if not env.actuator_success():
        env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
    
    
    targ_obj = ns.resolve_refinement(ref_obj,target_obj)
    if targ_obj==[] or targ_obj==None: #for example floor is not considered an object
        #if the language misguides to navigate to small objects that are to be manipulated
        #then instead go to the last successfully navigated target from agent memory
        if target_obj in INVEN_OBJS:
            target_obj = env.memory['navigated'][-1]
        targ_obj = target_obj
        print("Improper refinement, changed target to ",targ_obj)

    try:
        grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)


        env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params, other_side = 1)
        env = ns.unit_refinement(env, targ_obj)
    except:
        traceback.print_exc()
        print("Probably ",targ_obj," is not visible at all from this position")
        print("Trying to find out from previous images in trajectory ")
        try:
            grid, face_grids = GRIDS[targ_obj], FACE_GRIDS[targ_obj]
        except:
            print("Seems like ",targ_obj," was never navigated to earlier, so may not exist in the room")
            print("In that case, resolving confusion to see if another object is meant")
            targ_obj = ns.resolve_confusions(targ_obj, env)
            grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)

        env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params, other_side = 1)
        env = ns.unit_refinement(env, targ_obj)

    
    print(" ")
    print(" ")
    return env




def get_file(rn = 302, task_index = 1, trial_num = 0):
    #folders = sorted(glob.glob('/alfred/data/json_2.1.0/train/*'+repr(rn))) #for home computer
    folders = sorted(glob.glob(params.trajectory_data_location+repr(rn)))
    #folders = sorted(glob.glob('/home/microway/Desktop/hsaha/ai2thor/alfred/data/json_2.1.0/train/*'+repr(rn))) #for cluster
    #print("glob folders ",folders)
    #use sorted(glob.glob()) for systematic order in accesing the files
    #sys.exit(0)
    print("Number of demonstrated tasks for this room ",len(folders))
    trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
    print("Number of different trials (language instr) for the same task ",len(trials))
    traj = glob.glob(trials[trial_num]+'/*.json')

    print("got trajectory file ",traj)
    return traj



def run(env, sentences, intents, slot_dicts, ground_truth_map = -1, interactive = False):
    
    rewards = []
    grids = []
    orts = []

    lang_dict = {}
    lang_dict["instr"] = env.traj_data['turk_annotations']['anns'][0]["high_descs"][0]


    GRIDS, FACE_GRIDS = {},{}

    slot_dicts = replace_missing_slots(slot_dicts)

    task_tracker = {}
    task_tracker["task"] = sentences
    task_tracker["intents"] = intents
    task_tracker["slots"] = slot_dicts
    

    try:
        for i in range(len(intents)):
            if interactive:
                pause_input = input("paused for user input type anything and press enter ")
            intent = intents[i]
            slots = slot_dicts[i]
            sent = sentences[i]

            goalsat0 = env.get_goal_satisfied()
            goalsat1 = env.get_subgoal_idx()
            goalsat2 = env.get_postconditions_met() #for home computer
            #goalsat2 = env.get_goal_conditions_met() #for cluster

            n_navi_actions = pf.fnl(slots['action_n_navi'])
            any_carry = any([nna in CARRY for nna in n_navi_actions])
            any_pick = any([nna in PICK for nna in n_navi_actions])
            print("got n_navi_actions ",n_navi_actions)
            print("got any_carry ",any_carry)
            print("got any_pick ",any_pick)
            #sys.exit(0)

            if intent=="navigation" and any_carry:
                intent = "n_navigation"
            if intent=="navigation" and any_pick and slots['target_obj']!='':
                print("There seems to be a mismatch in intent and slots !")
                print("Analyzing whether this is navigation or manipulation -> ",sent)
                targ_objs = naming.name_objects(slots['target_obj'],sent)
                t = targ_objs.split(',')
                if t[0] not in eqc.GOTOOBJS:
                    intent = "n_navigation"
                    print("changed intent to ->",intent)

            if intent=="navigation":
                print("********** In navigation **********\n")
                print("Instruction -> ",sent)


                Slots,par_sents = pf.split_slots(slots,sent)
                slot_count = 0
                for slots in Slots:
                    print("Partial sentence -> ",par_sents[slot_count])
                    print("Slots ->",Slots[slot_count])
                    print(" ")

                    targ_objs = naming.name_objects(slots['target_obj'],sent) #is a string seperating targets by commas
                    print("Recognized target objects -> ",targ_objs, "/ ",slots['target_obj'])
                    ref_obj = naming.name_objects(slots['refinement_obj'],sent)
                    print("Recognized refinement objects -> ",ref_obj," / ",slots['refinement_obj'])
                    ref_rel = naming.name_directions(slots['refinement_rel'])
                    print("Recognized refinement relative -> ",ref_rel," / ",slots['refinement_rel'])
                    tar_rel = naming.name_directions(slots['target_rel'])
                    print("Recognized target relative -> ",tar_rel," / ",slots['target_rel'])
                    action_navi = naming.name_movements(slots['action_navi'])
                    print("Recognized navigation actions -> ",action_navi, "/ ",slots['action_navi'])
                    action_desc = naming.name_movements(slots['action_desc'])
                    print("Recognized navigation action descs -> ",action_desc, "/ ",slots['action_desc'])
                    action_int_desc = naming.intensity2digits(slots['action_intensity']) #is a flat list of words
                    print("Recognized action intensities -> ",action_int_desc, "/ ",slots['action_intensity'])
                    print(" ")


                    action_navi = action_navi.split(',')#is a flat list of words
                    action_desc = action_desc.split(',')#is a flat list of words

                    #insert target splitter function_here
                    #now a for loop over this next part
                    slot_count+=1
                    inten_count = 0
                    desc_count = 0
                    
                    #print("obtained action navi ",action_navi)
                    #print("obtained action_int_desc ",action_int_desc)
                    print("Executing hard instructions ")
                    for act_n in range(len(action_navi)):
                        try:
                            if action_navi[act_n]=='turn' and action_desc[desc_count]=='around': #this is a hard instruction execute it blindly
                                print("blind instruction turn around")
                                event = env.step(dict({"action": "RotateRight"}))

                                event = env.step(dict({"action": "RotateRight"}))
                                desc_count+=1
                            elif action_navi[act_n]=='turn' and action_desc[desc_count]=='left':
                                print("blind instruction turn left")
                                event = env.step(dict({"action": "RotateLeft"}))
                                desc_count+=1
                            elif action_navi[act_n]=='turn' and action_desc[desc_count]=='right':
                                print("blind instruction turn right")

                                event = env.step(dict({"action": "RotateRight"}))
                                desc_count+=1
                            elif (action_navi[act_n] in WALK) and action_int_desc[inten_count]!=-1 and action_int_desc!=['']:
                                print("Blind instruction walk a few mentioned steps")

                                for _ in range(action_int_desc[inten_count]):
                                    print("Executing move ahead ")
                                    event = env.step(dict({"action": "MoveAhead"}))
                                inten_count+=1
                            elif (action_navi[act_n] in WALK) and action_desc[desc_count]=='through':
                                print("Blind instruction walk through a space")
                                for _ in range(20):
                                    event,col = env.check_collision("MoveAhead")
                                    if col:
                                        print("reached end of room")
                                        break
                                desc_count+=1
                            elif (action_navi[act_n] in WALK) and action_desc[desc_count] in WALK_ST: #for instructions like go forward without specifying any number of steps
                                print("Blind instruction arbitrary walk")
                                for _ in range(4): #lets keep the arbitrary steps to 4
                                    print("Executing move ahead ")
                                    event = env.step(dict({"action": "MoveAhead"}))
                                desc_count+=1

                            elif action_navi[act_n] in WALK and action_desc[desc_count]=='right': #for commands like take a right
                                print("blind instruction take right")

                                event = env.step(dict({"action": "RotateRight"}))
                                desc_count+=1
                            elif action_navi[act_n] in WALK and action_desc[desc_count]=='left': #for commands like take a left
                                print("blind instruction take left")

                                event = env.step(dict({"action": "RotateLeft"}))
                                desc_count+=1
                        except:
                            print("Exception caught ! ")
                            #traceback.print_exc()
                            break
                    print(" ")
                    print("Executing arbitrary walks across the room (if mentioned)")
                    for targ_obj in targ_objs.split(','):
                        if targ_obj=='Wall' or targ_obj=='Room':
                            print("Blind movement walking towards wall or the end of room")
                            for i in range(20):
                                event,col = env.check_collision("MoveAhead")
                                if col:
                                    print("reached end of room")
                                    break
                    print(" ")
                    print("Executing targetted navigation ")
                    for targ_obj in targ_objs.split(','):
                        
                        if targ_obj!='Wall' and targ_obj!='Room' and targ_obj!='': # HERE start changing
                            
                            if ground_truth_map>0:
                                x,y,z = env.get_position()
                                #print("Got agent position in room ",x,y,z)
                                localize_params = {"room":ground_truth_map, 'position': [x, z]}
                            else:
                                localize_params = {"room":env}
                            
                            try:
                                grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)
                            
                            except:
                                traceback.print_exc()

                                if targ_obj not in eqc.INVEN_OBJS:
                                    v, diff = ns.target_visible(env,targ_obj) 
                                    
                                    if v!=-1:
                                        grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params, hallucinate = diff)
                                        #when halucinating, the proposed location of the object could be way outside border of map
                                        #so need to recursively expand the location borders until it gets inside the map
                                        #so need inf number of recursion stacks
                                        env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params, recursion_stacks = "inf") #HERE

                                        #just make sure agent knows its position perfectly before planning
                                        #after one round of hallucination agent might have goten close enough to get the actual map
                                        if targ_obj not in ns.TEXTURES:
                                            #texture objects like doors and mats cannot just be localized in any way
                                            grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)


                                    else:
                                        print("The agent might be stuck in between walls nudging to change views ")
                                        goal_vis, env, g, f = ns.random_explore(env, targ_obj, localize_params) 
                                        
                                        if goal_vis==False:
                                            print("WARNING ! target object mentioned in the language not found in the map, looking for similar objects ")
                                            targ_obj = ns.resolve_confusions(targ_obj, env, ref_obj = ref_obj)
                                            

                                            grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)
                                        else:
                                            grid, face_grids = g, f
                                else:
                                    print("Target object ",targ_obj," is too small to map")
                                    print("swapping target and refinement objects ")
                                    #sometimes for cases like go to the pot on top of table, the language understanding detects pot as target and table as ref
                                    #but it needs to be opposite, navigate to the bigger object
                                    targ_obj = copy.copy(ref_obj)
                                    ref_obj = copy.copy(targ_obj)
                                    grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)


                            
                            env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params, notarget = False)

                            
                            print("\n\n\n\n")
                            env = ns.unit_refinement(env, targ_obj)
                            print("\n\n\n\n")
                            
                            GRIDS[targ_obj] = grid
                            FACE_GRIDS[targ_obj] = face_grids

                    print(" ")
                    print(" ")
                    print("Executing post navigation corrections if any ")
                    
                    #for example telling to face opposite something
                    field = field_of_view(env)
                    obj_vis = any([ref_obj.split(',')[0]+'|' in a for a in list(field.keys())])
                    if ref_rel.split(',')[0]=='opposite' and obj_vis and tar_rel.split(',')[0]=='face':
                        print("caught object visible ",ref_obj.split(',')[0])
                        print("caught target relative ",tar_rel.split(',')[0])
                        print("caught instruction opposite so turning around")
                        event = env.step(dict({"action": "RotateLeft"}))
                        event = env.step(dict({"action": "RotateLeft"}))

            
            if intent=="n_navigation": 
                print("********** In manipulation **********\n")
                print("Instruction -> ",sent)
                manip_action = naming.name_actions(slots['action_n_navi'])
                print("Recognized manipulation -> ",manip_action," / ",slots['action_n_navi'])
                target_obj = naming.name_objects(slots['target_obj'],sent)
                print("Recognized target object -> ",target_obj," / ",slots['target_obj'])
                ref_attri = naming.name_directions(slots['refinement_attri'])
                print("Recognized refinement attribute -> ",ref_attri," / ",slots['refinement_attri'])
                ref_rel = naming.name_directions(slots['refinement_rel'])
                print("Recognized refinement relative -> ",ref_rel," / ",slots['refinement_rel'])
                ref_obj = naming.name_objects(slots['refinement_obj'],sent)
                print("Recognized refinement objects -> ",ref_obj," / ",slots['refinement_obj'])

                print(" ")
                
                if manip_action=="open,pick,close" or manip_action=="open,pick":
                    ms.drawer_manipulation_remove(manip_action,target_obj,ref_rel, ref_obj, env) 
                    print(" ")
                if manip_action=="look,open,pick":
                    #Look up and open the cabinet on the right side and then grab the cd that's in there before closing the door again.- 310-5
                    manip_action = "open,pick"
                    ms.drawer_manipulation_remove(manip_action,target_obj,ref_rel, ref_obj, env, event)
                    print(" ")

                #elif manip_action=="carry": 
                if "carry" in manip_action: 
                    targ_obj = ns.resolve_refinement(ref_obj,target_obj) 
                    try:
                        grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)
                    except:
                        print("WARNING ! target object ",targ_obj, " not found in the map, looking for similar objects ")
                        targ_obj = ns.resolve_confusions(targ_obj, env)
                        grid, face_grids = ns.occupancy_grid(env, targ_obj, ref_obj, localize_params)

                    env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params)
                    print("\n\n\n\n")
                    env = ns.unit_refinement(env, targ_obj)
                    print("\n\n\n\n") #just some gap for the debuggers eye
                    

                    ms.carry(ref_obj,ref_rel,env)
                    print("\n\n\n\n")

                if "clean" in manip_action:
                    env = ms.clean(manip_action,target_obj,ref_rel, ref_obj, env) 
                    print("\n\n\n\n")

                    env= ms.set_default_tilt(env)







                elif manip_action=='turnon': 
                    env,warn = ms.toggle(target_obj,ref_obj,manip_action,env)
                    print("\n\n\n\n")
                    if warn: #could not turn on the lamp for some reason
                        #set agent head tilt (vertical to default)
                        env = ms.set_default_tilt(env)
                        print("Setting navigation target to a DeskLamp and going towards it")
                        if params.room_type=="Bedroom":
                            target_obj = "DeskLamp"
                        if params.room_type=="Kitchen":
                            target_obj = "FloorLamp"
                        grid, face_grids = ns.occupancy_grid(env, target_obj, ref_obj, localize_params)
                        env = ns.graph_search(grid,face_grids, env, targ_obj, localize_params)
                        print("\n\n\n\n")
                        
                        env,warn = ms.toggle(target_obj,ref_obj,manip_action,env)
                        print("\n\n\n\n")


                #sometimes sentences like pick up the slice of apple gets together like pick,slice
                elif manip_action=='pick' or manip_action=="pick,carry" or manip_action=="pick,slice": 
                    env = ms.refined_pick(manip_action,target_obj,ref_rel,ref_obj,env)
                    print("\n\n\n\n")
                    if not ms.check_pick(env): 

                        env = other_side_attempt(GRIDS,FACE_GRIDS,target_obj,ref_obj, localize_params, env) 
                        #now try to pick up again
                        env = ms.refined_pick(manip_action,target_obj,ref_rel,ref_obj,env)
                        print("\n\n\n\n")

                elif manip_action=='close,pick':
                    ma = manip_action.split(',')
                    manip_action = ma[-1]
                    env = ms.refined_pick(manip_action,target_obj,ref_rel,ref_obj,env,preactions = ma[0])
                    print("\n\n\n\n")
                    if not ms.check_pick(env):
                        env = other_side_attempt(GRIDS,FACE_GRIDS,target_obj,ref_obj,localize_params,env)
                        #now try to pick up again
                        env = ms.refined_pick(manip_action,target_obj,ref_rel,ref_obj,env)
                        print("\n\n\n\n")


                elif manip_action=='place': 

                    if ms.resolve_place(manip_action,target_obj,ref_rel,ref_obj): #sometimes put stuff in something entails opening the thing and then put it
                        manip_action="open,place,close"
                        env = ms.drawer_manipulation_place(manip_action,target_obj,ref_rel, ref_obj, env) 
                        print("\n\n\n\n")

                    elif len(ref_obj.split(','))<=2:
                        env = ms.refined_place(manip_action,target_obj,ref_rel,ref_obj,env) 
                        print("\n\n\n\n")

                        if not ms.check_place(env):
                            env = ms.set_default_tilt(env)

                            env = other_side_attempt(GRIDS,FACE_GRIDS,target_obj,ref_obj,localize_params,env)
                            #now try to pick up again
                            env = ms.refined_place(manip_action,target_obj,ref_rel,ref_obj,env)
                            print("\n\n\n\n")


                    elif len(ref_obj.split(','))>=3: 
                        env = ms.refined_place2(manip_action,target_obj,ref_rel,ref_obj,env)
                        print("\n\n\n\n")
                        if not ms.check_place(env):

                            env = other_side_attempt(GRIDS,FACE_GRIDS,target_obj,ref_obj,localize_params, env)
                            #now try to pick up again
                            env = ms.refined_place2(manip_action,target_obj,ref_rel,ref_obj,env)
                            print("\n\n\n\n")

                    env= ms.set_default_tilt(env)

                elif manip_action=='slice': 
                    env = ms.refined_slice(manip_action,target_obj,ref_rel, ref_obj, env) 
                    print("\n\n\n\n")

                    env= ms.set_default_tilt(env)

                elif manip_action=="cook" or manip_action=="cook,pick": 
                    env = ms.cook(manip_action,target_obj,ref_rel, ref_obj, env) 
                    print("\n\n\n\n")

                    env= ms.set_default_tilt(env)

                elif manip_action=="open,place,close" or manip_action=="open,place" or manip_action=="place,close":
                    ms.drawer_manipulation_place(manip_action,target_obj,ref_rel, ref_obj, env)
                    print("\n\n\n\n")

                elif manip_action=="look": 
                    ms.gaze(action_desc,env)
                    print(" ")

                print("\n\n\n\n")
            
            #event = env.step(dict({"action":'Stand'})) #does nothing actually, but makes sure the event metadata is updated to latest
            goalsat0 = env.get_goal_satisfied()
            goalsat1 = env.get_subgoal_idx()
            goalsat2 = env.get_postconditions_met() #for home computer
            #goalsat2 = env.get_goal_conditions_met() #for cluster
            print("goals satisfied ",goalsat0,goalsat1,goalsat2)
            task_tracker["goal_satisfied"] = goalsat0
            task_tracker["subgoal_idx"] = goalsat1
            task_tracker["post_conditions"] = goalsat2
            task_tracker["trajectory_length"] = env.cur_traj_len #some additional considerations needed because we are assuming panorama
            alg_length = task_tracker["trajectory_length"]


            exp_length = get_demo_traj_length(env.rn, env.task_index, env.trial_num) #they sould be set earlier in main file
            task_tracker["exp_length"] = exp_length

            print("Expert trajectory length ",exp_length)
            print("Algorithm trajectory length ",alg_length)

        '''
        #save a video of algorithm executed trajectory
        #need to do this later


        if args.make_video:
            import create_video as vid
            condition = ""

            if task_tracker['goal_satisfied']==1:
                vid.video_make('sample_traj/success/'+repr(args.room)+'_'+repr(args.task))
                condition = 'success'
            else: 
                if task_tracker["post_conditions"][0] > 0:
                    vid.video_make('sample_traj/partial/'+repr(args.room)+'_'+repr(args.task))
                    condition = 'partial'
                if task_tracker["post_conditions"][0] == 0:
                    vid.video_make('sample_traj/failed/'+repr(args.room)+'_'+repr(args.task))
                    condition = 'failed'

            with open('sample_traj/'+condition+'/task_info/'+'_room_'+repr(args.room)+'_task_'+repr(args.task)+'.json', 'w') as fp:
                json.dump(task_tracker, fp, indent = 4)
            #clean the folder that stores the images
            shutil.rmtree('/home/hom/Desktop/ai2thor/sample_traj/frames') #clear junk data first
            os.system('mkdir sample_traj/frames')
        '''
        env.init_memory()
        return task_tracker

    except:
        print("failed to execute complete action for some error")
        traceback.print_exc()
        exp_length = get_demo_traj_length(env.rn, env.task_index, env.trial_num) #they sould be set earlier in main file
        task_tracker["exp_length"] = exp_length
        task_tracker["goal_satisfied"] = 0
        task_tracker["subgoal_idx"] = -1
        task_tracker["post_conditions"] = env.get_postconditions_met()
        task_tracker["trajectory_length"] = -1
        return task_tracker

    

