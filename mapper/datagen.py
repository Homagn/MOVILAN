import numpy as np
import math
import sys
import glob
import os
import json
import random
import copy
import argparse
import cv2

#sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
#sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
#from env.thor_env import ThorEnv
os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
from robot.sensing import sensing


import panorama as pan
import projection as proj
import gtmaps as gtm
import mapper.params as params



parser = argparse.ArgumentParser()



#my arguments
parser.add_argument('--room', type=int, default=301)
parser.add_argument('--task', type=int, default=1)

parser.add_argument('--inputs', dest='inputs', action='store_true')
parser.add_argument('--targets', dest='targets', action='store_true') #if --nomap is passed agent has to approximate bev proj using graph conv 
parser.add_argument('--correct', dest='correct', action='store_true')


parser.add_argument('--allinputs', dest='allinputs', action='store_true')
parser.add_argument('--alltargets', dest='alltargets', action='store_true')
parser.add_argument('--aliasinput', dest='aliasinput', action='store_true')

parser.add_argument('--checkpan', dest='checkpan', action='store_true')
parser.add_argument('--checkinput', dest='checkinput', action='store_true')
parser.add_argument('--checktarget', dest='checktarget', action='store_true')
parser.add_argument('--specific_input', dest='specific_input', action='store_true')
parser.add_argument('--live_explore', dest='live_explore', action='store_true')

args = parser.parse_args()




def get_file(rn = 302, task_index = 1, trial_num = 0):
    #folders = sorted(glob.glob('/home/hom/alfred/data/json_2.1.0/train/*'+repr(rn))) #for home computer
    #folders = sorted(glob.glob('/home/hom/alfred/data/json_2.1.0/train/*-'+repr(rn))) #for home computer
    #folders = sorted(glob.glob('/alfred/data/json_2.1.0/train/*-'+repr(rn))) #for home computer
    folders = sorted(glob.glob(params.trajectory_data_location+repr(rn))) #for home computer
    print("Number of demonstrated tasks for this room ",len(folders))
    trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
    print("Number of different trials (language instr) for the same task ",len(trials))
    traj = glob.glob(trials[trial_num]+'/*.json')
    print("got trajectory file ",traj)
    return traj

def list_nicely(l):
    c = 0
    for i in l:
        print(i,"----",c)
        c+=1

def set_env(json_file, env = []):
    if env==[]:
        #if no env passed, initialize an empty environment first
        #IMAGE_WIDTH = 300 #rendering- can change this in robot/params.py
        #IMAGE_HEIGHT = 300
        #env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment
        env = sensing()
    
    with open(json_file[0]) as f:
        traj_data = json.load(f)
    #print("loaded traj file")
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    print("setting orientation of the agent to facing north ")
    traj_data['scene']['rotation'] = 0
    event = env.step(dict(traj_data['scene']['init_action']))

    return env,event,traj_data

if __name__ == '__main__':

    def init_once(room  = 301, task = 0):
        #load information from training trajectory files and set the environment accordingly
        traj_file = get_file(rn = room, task_index = task, trial_num = 0)
        env,event,traj_data = set_env(traj_file)
        #orient the agent to always facing north
        custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": 0,
                        "x": event.metadata['agent']['position']['x'],
                        "y": event.metadata['agent']['position']['y'],
                        "z": event.metadata['agent']['position']['z']}
        event = env.step(dict(custom_rot))
        return env,event


    if args.inputs:
        #obtain input vision panorama images and approx projection map from that

        #load the ground truth BEV map obtained using other functions
        #from this map, we get to know the actual minimum coordinate positions in the room
        o_grids = np.load('data/targets/'+repr(args.room)+'.npy',allow_pickle = 'TRUE').item()
        env,event = init_once(room = args.room, task = args.task)

        x,y,z = event.metadata['agent']['position']['x'], event.metadata['agent']['position']['y'], event.metadata['agent']['position']['z']
        print("In task ",args.task," the agent starting position is ",x,y,z)
        grid_coord = [int((z-o_grids['min_pos']['mz'])/0.25), int((x-o_grids['min_pos']['mx'])/0.25)]
        print("This is equivalent to grid coordinate ",grid_coord)

        debug = False # True-load premade panorama image/ False- Dont
        #gridsize = 33
        gridsize = params.grid_size
        panorama = pan.rotation_image(env, objects_to_visit = [], debug = debug) #gets a panorama image of everything thats visible
        bev = proj.bevmap(panorama,grid_size = gridsize, debug = debug)
        #save the obtained bev projection
        np.save('data/inputs/bev_'+repr(args.room)+'_'+repr(grid_coord)+'.npy',bev)

        #sample check the created projection map for a random object in the room
        proj.displaymap(bev,'Desk')
        nav_map = proj.input_navigation_map(bev, 'Desk', grid_size = gridsize, unk_id = 0,flr_id = 1, tar_id = 2, obs_id = 3)
        print("Now displaying input navigation map")
        proj.prettyprint(nav_map,argmax = True)
        #print(nav_map) #should be grid_size x grid_size x 4 matrix contain floats between 0 and 1

    if args.aliasinput:
        #change any key in any room for semantic maps in the inputs bev projections
        #say for eg- FP301:Cube is renamed as Shelf1| (a more meaningful name)
        print("For room ",args.room)
        try:
            with open('data/alias/'+repr(args.room)+'.json') as f:
                aliases = json.load(f)
            print("loaded previous aliases ")
            print(aliases)
        except:
            aliases = {}
        while True:
            obj = input("Enter the actual object id you want to alias")
            obj_alis = input("Enter the alias name ")
            aliases[obj_alis] = obj
            
            i = input("Continue ?(y/n) ")
            if i=="n":
                break
        
        with open('data/alias/'+repr(args.room)+'.json', 'w') as fp:
            json.dump(aliases, fp, indent = 4)

    if args.targets:
        # creating ground truth BEV maps
        #room  = 0
        #task = 164
        env,event = init_once(room  = args.room, task  = args.task)
        #o_grids stores BEV map for all objects as indexed in the event metadata
        #fname = '../ai2thor/mapping/gcdata/'+repr(room)+'.npy'
        fname = 'data/targets/'+repr(args.room)+'.npy'
        o_grids = gtm.gtmap(env,event) # Obtains ground truth occupancy grids using Ai2Thor functions / try-> Dresser|-01.33|+00.00|-00.75 for room 301 
        np.save(fname,o_grids)

    if args.correct:
        #manual labeling for objects that are unable to be disabled and are like fixed parts in the room
        gtm.manual_label(args.room) #0 is the room number (0 is the first kitchen, 301 is the first bedroom)


    if args.alltargets:
        #IMAGE_WIDTH = 300 #rendering
        #IMAGE_HEIGHT = 300
        #env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment
        env = sensing()

        room_range = [308,330]
        print("Creating all the training targets for graph convolution mapping training ")
        
        for r in range(room_range[0], room_range[1]):
            try:
                traj_file = get_file(rn = r, task_index = 0, trial_num = 0) #take default first task of each room
            except:
                print("File is not present ")
                continue
            env,event,traj_data = set_env(traj_file, env = env)
            #orient the agent to always facing north
            custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": 0,
                            "x": event.metadata['agent']['position']['x'],
                            "y": event.metadata['agent']['position']['y'],
                            "z": event.metadata['agent']['position']['z']}
            event = env.step(dict(custom_rot))

            fname = 'data/targets/'+repr(r)+'.npy'
            o_grids = gtm.gtmap(env,event) # Obtains ground truth occupancy grids using Ai2Thor functions / try-> Dresser|-01.33|+00.00|-00.75 for room 301 
            np.save(fname,o_grids)

    if args.allinputs:
        #IMAGE_WIDTH = 300 #rendering
        #IMAGE_HEIGHT = 300
        #env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment
        env = sensing()

        room_range = [327,331]
        print("Creating all the training inputs for graph convolution mapping training ")

        for r in range(room_range[0], room_range[1]):
            try:
                traj_file = get_file(rn = r, task_index = 0, trial_num = 0) #take default first task of each room
                print("Loaded room ",r)
            except:
                print("File is not present ")
                continue
            env,event,traj_data = set_env(traj_file, env = env)

            #load the ground truth BEV map obtained using other functions
            #from this map, we get to know the actual minimum coordinate positions in the room
            o_grids = np.load('data/targets/'+repr(r)+'.npy',allow_pickle = 'TRUE').item()
            #env,event = init_once(room = args.room, task = args.task)
            nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()

            for p in nav_pos:
                #command to position the agent in one of the navigable positions facing north
                custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": 0,
                                "x": p[0]*0.25+ o_grids['min_pos']['mx'],
                                "y": event.metadata['agent']['position']['y'],
                                "z": p[1]*0.25+ o_grids['min_pos']['mz']}

                event = env.step(dict(custom_rot))

                
                debug = False # True-load premade panorama image/ False- Dont
                #gridsize = 33
                gridsize = params.grid_size
                panorama = pan.rotation_image(env, objects_to_visit = [], debug = debug) #gets a panorama image of everything thats visible
                bev = proj.bevmap(panorama,grid_size = gridsize, debug = debug)
                #save the obtained bev projection
                np.save('data/inputs/bev_'+repr(r)+'_'+repr([p[1],p[0]])+'.npy',bev)
                print("saved panorama projection image for the room ",r," for the position ",[p[1],p[0]])
                


    #==========================================================================================
    #DEBUGGING PURPOSES 
    
    if args.checkpan: #check panorama images at a location
        env,event = init_once(room = args.room, task = args.task)
        debug = True # True-load premade panorama image/ False- Dont
        gridsize = 33
        panorama = pan.rotation_image(env, objects_to_visit = [], debug = debug) #gets a panorama image of everything thats visible

    if args.checkinput: #check whether the projection mapping was done properly
        room = 301
        pos = [0,2]
        room_obj = 'Bed'

        camera_proj = np.load('data/inputs/bev_'+repr(room)+'_'+repr(pos)+'.npy',allow_pickle = 'TRUE').item()
        nav_map = proj.input_navigation_map(camera_proj, room_obj, grid_size = params.grid_size, 
                                            unk_id = params.semantic_classes['unk'],
                                            flr_id = params.semantic_classes['flr'], 
                                            tar_id = params.semantic_classes['tar'], 
                                            obs_id = params.semantic_classes['obs'])
        #nav_map is an array of params.grid_size x params.grid_size x 4 (4 classes unk,flr,tar and obs)| contains values between 0 and 1
        #for example grid i,j may contain the value [0.1,0.2,0.3,0.5] - sum of values is 1.0, last index is the argmax meaning highest prob that the grid contains obstacle
        #because obstacle id=3 in params.py

        can_see_target = proj.prettyprint(nav_map,argmax = True) #it cant print without argmax because each element of nav_map is 4 dimensional
        print("Agent camera can see target ? ",can_see_target)

    if args.checktarget: #check whether the projection mapping was done properly
        room = args.room
        o_grids = np.load('data/targets/'+repr(room)+'.npy',allow_pickle = 'TRUE').item()
        print("All the objects in the room \n ")
        list_nicely(o_grids.keys())
        obidx = input("Enter the number from the object list above ")
        
        nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()
        print("All the navigable positions in the room are \n")
        list_nicely(nav_pos)
        pidx = input("Enter the number from the position list above ")
        #sys.exit(0)
        

        #pos = [2,0]
        pos = nav_pos[int(pidx)]
        #room_obj = 'Bed'
        room_obj = list(o_grids.keys())[int(obidx)]
        
        #nav_map is an array of params.grid_size x params.grid_size  -each grid [i,j] contains an integer between 0 to 4 denoting the type of object occupying that place
        #whether its a target/navgable space/unk/obstacle
        nav_map_t = gtm.target_navigation_map(  o_grids, room_obj, 
                                                {'x':pos[0]*0.25+ o_grids['min_pos']['mx'], 'y':0.0, 'z': pos[1]*0.25+ o_grids['min_pos']['mz']}, 
                                                grid_size = params.grid_size, 
                                                unk_id = params.semantic_classes['unk'],
                                                flr_id = params.semantic_classes['flr'], 
                                                tar_id = params.semantic_classes['tar'], 
                                                obs_id = params.semantic_classes['obs'])

        print("Showing ground truth map")
        #gtm.prettyprint(nav_map_t)
        proj.starviz(nav_map_t)


    if args.specific_input: #for debugging purposes
        #IMAGE_WIDTH = 300 #rendering
        #IMAGE_HEIGHT = 300
        #env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment
        env = sensing()

        room = 301
        pos = [9,4]

        print("Creating all the training inputs for graph convolution mapping training ")

        r = room
        try:
            traj_file = get_file(rn = r, task_index = 0, trial_num = 0) #take default first task of each room
            print("Loaded room ",r)
        except:
            print("File is not present ")
            
        env,event,traj_data = set_env(traj_file, env = env)

        #load the ground truth BEV map obtained using other functions
        #from this map, we get to know the actual minimum coordinate positions in the room
        o_grids = np.load('data/targets/'+repr(r)+'.npy',allow_pickle = 'TRUE').item()
        #env,event = init_once(room = args.room, task = args.task)
        nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()

        p = pos
        #command to position the agent in one of the navigable positions facing north
        custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": 0,
                        "x": p[0]*0.25+ o_grids['min_pos']['mx'],
                        "y": event.metadata['agent']['position']['y'],
                        "z": p[1]*0.25+ o_grids['min_pos']['mz']}

        event = env.step(dict(custom_rot))

        
        debug = False # True-load premade panorama image/ False- Dont
        #gridsize = 33
        gridsize = params.grid_size
        panorama = pan.rotation_image(env, objects_to_visit = [], debug = debug) #gets a panorama image of everything thats visible
        bev = proj.bevmap(panorama,grid_size = gridsize, debug = debug)
        #save the obtained bev projection
        np.save('data/inputs/bev_'+repr(r)+'_'+repr([p[1],p[0]])+'.npy',bev)
        print("saved panorama projection image for the room ",r," for the position ",[p[1],p[0]])

    if args.live_explore:
        env,event = init_once(room = args.room, task = 1) #every room should have atleast 1 task
        o_grids = np.load('data/targets/'+repr(args.room)+'.npy',allow_pickle = 'TRUE').item()
        
        def live_map(env):
            x,y,z = env.get_position()
            rot = env.get_rotation() 
            locate = [int((x-o_grids['min_pos']['mx'])/0.25), int((z-o_grids['min_pos']['mz'])/0.25), rot]
            nav_map_t = gtm.target_navigation_map(  o_grids, 'nav_space', 
                                                    {'x':x, 'y':0.0, 'z': z}, 
                                                    grid_size = params.grid_size, 
                                                    unk_id = params.semantic_classes['unk'],
                                                    flr_id = params.semantic_classes['flr'], 
                                                    tar_id = params.semantic_classes['tar'], 
                                                    obs_id = params.semantic_classes['obs'])

            print("Showing ground truth map")
            gtm.prettyprint(o_grids['nav_space'], locator = locate)
            #proj.starviz(o_grids['nav_space'])

        
        winname = "seg"
        s = env.get_segmented_image()
        cv2.imshow("seg",s)

        def process_mouse_events(event,x,y,flags,param): #allows for continuous key press !
            global s
            # to check if left mouse  
            # button was clicked 
            mouse_state = {'button':None, 'position':None}
            if event == cv2.EVENT_LBUTTONDOWN: 
                  
                # font for left click event 
                font = cv2.FONT_HERSHEY_TRIPLEX 
                LB = 'Left Button'
                print("left button pressed at ",x,y)
                print("colors ",s[y,x,:], env.identify_segmented_color(s[y,x,:]))
                mouse_state['button'] = 'left'
                mouse_state['position'] = (x,y)
                px,py = x,y
            else:
                mouse_state['button'] = None
                mouse_state['position'] = None
                px,py = -1,-1

        
        cv2.setMouseCallback(winname, process_mouse_events)

        while(True):
            s = env.get_segmented_image()
            cv2.imshow(winname,s)
            c = cv2.waitKey(0)


            #c = input("Enter command ")
            if c==ord('w'):
                event = env.step(dict({"action": "MoveAhead"}))
            if c==ord('a'):
                event = env.step(dict({"action": "MoveLeft"}))
            if c==ord('s'):
                event = env.step(dict({"action": "MoveBack"}))
            if c==ord('d'):
                event = env.step(dict({"action": "MoveRight"}))
            if c==ord('i'):
                event = env.step(dict({"action": "RotateRight"}))
            if c==ord('j'):
                event = env.step(dict({"action": "RotateLeft"}))
            if c==ord('o'):
                event = env.step(dict({"action": "LookUp"}))
            if c==ord('k'):
                event = env.step(dict({"action": "LookDown"}))
            if c==ord('b'):
                break
            if c==ord('v'):
                act = input("Enter the action ")
                objid = input("Enter the object_id ")
                event = env.step(dict({"action": act, "objectId": objid}))
                
            

            live_map(env)





'''
run examples (for any new room, first get the targets, then inputs, then live_explore for map corrections and then correct)
(live_explore will open a cv window you can click on the objects in segmented image to identify the name and also click w/a/s/d to move)

(debug the panorama image acquisition process)
python datagen.py --room 301 --task 0 --checkpan

(debug the projection map created  from the panorama image )
python3 datagen.py --room 301 --task 0 --checkinput

(get ground truth BEV data)
python3 datagen.py --room 301 --task 0 --checktarget


(explore the map and look for map corrections)
python datagen.py --room 301 --live_explore

(correct wrong ground truth BEV data)
python datagen.py --room 301 --correct

(get camera based approximate projection BEV data) (must run after ground truth BEV is extracted and corrected)
python datagen.py --room 301 --task 0 --inputs

python datagen.py --room 1 --aliasinput


(complete data preparation of inputs and outputs/ correction need to be done for each seperately later)

python datagen.py --alltargets

python datagen.py --allinputs
'''

    