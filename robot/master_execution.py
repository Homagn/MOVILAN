#This code designed to fetch all the trajectory data from the alfred folder in an organized fashion
import os
import sys
os.environ['ALFRED_ROOT'] = '/alfred'

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'data/json_2.1.0/train'))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
from utils.video_util import VideoSaver
from utils.py_util import walklevel
from env.thor_env import ThorEnv
import time as t


#my imports 
import math
#import instruction_parser as ip
#import graph_viz as gv



#Data parameters
IMAGE_WIDTH = 300 #rendering
IMAGE_HEIGHT = 300
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data")
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_false') #can also try 'store_false' to see smooth trajectories
parser.add_argument('--time_delays', dest='time_delays', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_true') 
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--reward_config', type=str, default='data/config/rewards.json')

#my arguments
parser.add_argument('--room', type=int, default=301)
parser.add_argument('--task', type=int, default=1)
parser.add_argument('--gendata', dest='gendata', action='store_true')
parser.add_argument('--numexec', type=int, default=-1)
args = parser.parse_args()


render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True


#data generation control parameters
#args.task = 6
'''
room number 301- #0-15 1-18 2-16 6-6 7-13 8-13 9-10
'''
#for 7-13 dont know if its Chair ir ArmChair
numexec = args.numexec #make it -1 for finding out the number of actions after which it starts the 2nd instruction

def get_file(rn = 302, task_index = 1, trial_num = 0):
    folders = sorted(glob.glob('/alfred/data/json_2.1.0/train/*-'+repr(rn)))
    print("Number of demonstrated tasks for this room ",len(folders))
    trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
    traj = glob.glob(trials[trial_num]+'/*.json')

    print("got trajectory file ",traj)
    return traj

def parse_instr(lang):
    parse_tree = ip.parse(lang,weights = "parse_weights")
    print("Got parse tree ",parse_tree)
    print("Take a look at the constructed parse tree")
    gv.visualize_projection_tree(language = parse_tree)

def inspect_lang_dict(rn = 301, task = 0):
    d = np.load('panorama_data/language_data/room_number_'+repr(rn)+'_task'+repr(task)+'.npy',allow_pickle = 'TRUE').item()
    print("loaded Dictionary ")
    print(d)

def example_run_from_traj(json_file, numexec = numexec):
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT)
    #open the expert demonstration trajectory file
    nav_dict = {}
    with open(json_file) as f:
        traj_data = json.load(f)

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data['scene']['init_action']))
    #print("Task: %s" % (traj_data['template']['task_desc']))
    print("Task: %s" % (traj_data['turk_annotations']['anns'][0]["high_descs"]))

    nav_dict["instructions"] = traj_data['turk_annotations']['anns'][0]["high_descs"]
    nav_dict["commands"] = []
    nav_dict["grids"]=[]
    nav_dict["orts"]=[]
    #print("First navigation instruction: %s" % (traj_data['turk_annotations']['anns'][0]["high_descs"][0]))

    #parse_instr(traj_data['turk_annotations']['anns'][0]["high_descs"][0])

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    rewards = []
    grids = []
    orts = []

    lang_dict = {}
    lang_dict["instr"] = traj_data['turk_annotations']['anns'][0]["high_descs"][0]

    event = env.step(dict(action = 'GetReachablePositions'))
    reach_pos = event.metadata['actionReturn'] #stores all reachable positions for the current scene
    reach_x = [i['x'] for i in reach_pos]
    reach_z = [i['z'] for i in reach_pos]

    m_x = min(reach_x)
    m_z = min(reach_z)


    #print(enumerate(traj_data['plan']['low_actions']))
    n = 0
    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        # next cmd under the current hl_action
        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]

        # remove unnecessary keys
        cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}
        print("command ",cmd)
        nav_dict["commands"].append(cmd)
        

        x = event.metadata['agent']['position']['x']
        y = event.metadata['agent']['position']['y']
        z = event.metadata['agent']['position']['z']
        #print("x ",x," y ",y," z ",z)
        a = int(math.fabs((x - m_x)/0.25))
        b = int(math.fabs((z - m_z)/0.25))

        nav_dict["grids"].append(repr(a)+"_"+repr(b))
        nav_dict["orts"].append(event.metadata['agent']['rotation'])

        if "MoveAhead" in cmd['action']:
            if args.smooth_nav:
                #save_image(env.last_event, root_dir) #will do our own function
                events = env.smooth_move_ahead(cmd, render_settings)
                #save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                event = env.step(cmd)
                #save_image(event, root_dir)

        elif "Rotate" in cmd['action']:
            if args.smooth_nav:
                #save_image(env.last_event, root_dir)
                events = env.smooth_rotate(cmd, render_settings)
                #save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                event = env.step(cmd)
                #save_image(event, root_dir)

        elif "Look" in cmd['action']:
            if args.smooth_nav:
                #save_image(env.last_event, root_dir)
                events = env.smooth_look(cmd, render_settings)
                #save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                event = env.step(cmd)
                #save_image(event, root_dir)

        # handle the exception for CoolObject tasks where the actual 'CoolObject' action is actually 'CloseObject'
        # TODO: a proper fix for this issue
        elif "CloseObject" in cmd['action'] and \
             "CoolObject" in hl_action['planner_action']['action'] and \
             "OpenObject" in traj_data['plan']['low_actions'][ll_idx + 1]['api_action']['action']:
            if args.time_delays:
                cool_action = hl_action['planner_action']
                #save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(cmd)
                #save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.MIDDLE)
                #save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.AFTER)
            else:
                event = env.step(cmd)
                #save_image(event, root_dir)

        else:
            if args.time_delays:
                #save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(cmd)
                #save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.MIDDLE)
                #save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.AFTER)
            else:
                event = env.step(cmd)
                #save_image(event, root_dir)

        # update image list
        '''
        new_img_idx = get_image_index(high_res_images_dir)
        last_img_idx = len(traj_data['images'])
        num_new_images = new_img_idx - last_img_idx
        for j in range(num_new_images):
            traj_data['images'].append({
                'low_idx': ll_idx,
                'high_idx': ll_action['high_idx'],
                'image_name': '%09d.png' % int(last_img_idx + j)
            })
        '''

        if not event.metadata['lastActionSuccess']:
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

        n+=1
        if n>=numexec and numexec!=-1: #-1 means testing 
            break
        x = event.metadata['agent']['position']['x']
        y = event.metadata['agent']['position']['y']
        z = event.metadata['agent']['position']['z']
        #print("x ",x," y ",y," z ",z)
        a = int(math.fabs((x - m_x)/0.25))
        b = int(math.fabs((z - m_z)/0.25))

        grids.append(repr(a)+"_"+repr(b))
        orts.append(event.metadata['agent']['rotation'])


        reward, _ = env.get_transition_reward()
        rewards.append(reward)

        goalsat1 = env.get_subgoal_idx()
        goalsat2 = env.get_goal_conditions_met() #env.get_postconditions_met()
        print("subgoal index ",goalsat1)
        print("postconditions met ",goalsat2)
        print("reward ",reward)

    #np.save('expert_demo_'+repr()+'_'+repr()+'.npy',event.metadata)

    #print("This is the list of rewards for each action ",rewards)
    #print("This is the list of grids ",grids)
    lang_dict["grids"] = grids
    lang_dict["orts"] = orts
    if numexec!=-1:
        x = input("Enter the refinement object1 (press enter if doesnt exist ")
        rf1 = x
        y = input("Enter the refinement object2 (press enter if doesnt exist ")
        rf2 = y
        z = input("Enter the target object (press enter if doesnt exist ")
        rf2 = z
        #ex- Turn around and walk over to the white desk on your right.
        #ex(navigation cues)- turn around,walk right
        #no refinement objects 1 and 2
        #target object is Desk
        #sometimes there is no worthy navigation cue for ex- Move to the bottom right side of the large wood dresser (no cues here)
        c = input("Enter the navigation cues implied from the sentence (use keyword pairs seperated by comma) ")

        lang_dict["objects"] ={}
        lang_dict["objects"]["ref1"] = x
        lang_dict["objects"]["ref2"] = y
        lang_dict["objects"]["target"] = z
        lang_dict["nav_cues"] = c
        lang_dict["traj_file_name"] = json_file


        #np.save('panorama_data/language_data/room_number_'+repr(scene_num)+'_task'+repr(args.task)+'.npy',lang_dict)


if __name__ == '__main__':
    gen_data = args.gendata
    print("got gendata ",gen_data)

    if gen_data:
        traj_file = get_file(rn = args.room, task_index = args.task)
        example_run_from_traj(traj_file[0])
    else:
        inspect_lang_dict(rn = args.room, task = args.task)

'''
#simple run whole trajectory
python3 master_execution.py --room 1 --task 1 --gendata

#to check the complete trajectory
python annotate_traj.py --room 301 --task 1 --numexec -1 --gendata

#to generate and store language dictionary data
python annotate_traj.py --room 301 --task 1 --numexec 10 --gendata

#to verify stored dictionary
python annotate_traj.py --room 301 --task 1 
'''