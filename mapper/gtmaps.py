import numpy as np
import math
import sys
import glob
import os
import json
import random
import copy

from skimage.measure import regionprops, label



def get_file(rn = 302, task_index = 1, trial_num = 0):
    folders = sorted(glob.glob('/home/hom/alfred/data/json_2.1.0/train/*'+repr(rn))) #for home computer
    print("Number of demonstrated tasks for this room ",len(folders))
    trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
    print("Number of different trials (language instr) for the same task ",len(trials))
    traj = glob.glob(trials[trial_num]+'/*.json')
    print("got trajectory file ",traj)
    return traj

def touchmap(env,event):
    #sometimes in a room there are fixed objects which cannot be removed from scene using disable command
    #so need to go near them to check distance and then map them
    return



def gtmap(env,event):
    objs = event.metadata['objects']
    print("There are a total of ",len(objs)," objects in the scene")
    names = [o['objectId'] for o in objs]
    centers = [o['position'] for o in objs]

    print("Now disabling every object in the scene ")
    for n in names:
        event = env.step(dict({"action":"DisableObject", "objectId": n}))
    #getting reachable positions for the empty room
    event = env.step(dict(action = 'GetReachablePositions'))
    reach_pos = event.metadata['actionReturn'] #stores all reachable positions for the current scene
    #print("got reachable positions ",reach_pos)
    reach_x = [i['x'] for i in reach_pos]
    reach_z = [i['z'] for i in reach_pos]
    coords = [[i['x'],i['z']] for i in reach_pos]

    #getting navigable spaces in the empty room (only walls should be blocking now)
    c_x = int(math.fabs((max(reach_x)-min(reach_x))/0.25))+1 #0.25 is the grid movement size
    c_z = int(math.fabs((max(reach_z)-min(reach_z))/0.25))+1
    print("c_x ",c_x," c_z ",c_z)
    m_x = min(reach_x)
    m_z = min(reach_z)
    nav_grid = np.zeros((c_x,c_z))
    for i in range(nav_grid.shape[0]):
        for j in range(nav_grid.shape[1]):
            if [m_x + i*0.25, m_z + j*0.25] in coords:
                nav_grid[i,j] = 1
            else:
                nav_grid[i,j] = 0
    #print("nav_grid after disabling every object ")
    #print(nav_grid)
    #sys.exit(0)
    #print("Got nav_grid on empty room ",nav_grid)
    obj_grids = {}
    obj_grids['fixed_obstructions'] = nav_grid
    #flr_grid = np.zeros_like(nav_grid)
    for n in range(len(names)):
        obj_grid = copy.copy(nav_grid)
        #now enable just the object you want to map
        print("Now enabling ",names[n], " back ")
        event = env.step(dict({"action":"EnableObject", "objectId": names[n]}))

        #getting reachable positions again
        event = env.step(dict(action = 'GetReachablePositions'))
        reach_pos = event.metadata['actionReturn'] #stores all reachable positions for the current scene
        reach_x = [i['x'] for i in reach_pos]
        reach_z = [i['z'] for i in reach_pos]
        coords = [[i['x'],i['z']] for i in reach_pos]
        
        obj_center = [centers[n]['x'], centers[n]['z'] ] 

        for i in range(obj_grid.shape[0]):
            for j in range(obj_grid.shape[1]):
                if [m_x + i*0.25, m_z + j*0.25] in coords and obj_grid[i,j] == 1:
                    obj_grid[i,j] = 0
                '''
                if int(m_x + i*0.25) == int(obj_center[0]) and int(m_z + j*0.25) == int(obj_center[1]):
                    print("object center matched for object ",names[n])
                    obj_grid[i,j] == 1
                '''
        
        obj_grids[names[n]] = obj_grid
        #flr_grid = flr_grid + obj_grid
        print("Disabling the object")
        event = env.step(dict({"action":"DisableObject", "objectId": names[n]}))

    for n in names:
        print("Now enabling ",n, " back ")
        event = env.step(dict({"action":"EnableObject", "objectId": n}))
    
    event = env.step(dict(action = 'GetReachablePositions'))
    reach_pos = event.metadata['actionReturn'] #stores all reachable positions for the current scene
    reach_x = [i['x'] for i in reach_pos]
    reach_z = [i['z'] for i in reach_pos]
    coords = [[i['x'],i['z']] for i in reach_pos]
    flr_grid = np.zeros((c_x,c_z))

    for i in range(flr_grid.shape[0]):
        for j in range(flr_grid.shape[1]):
            if [m_x + i*0.25, m_z + j*0.25] in coords:
                flr_grid[i,j] = 1

    obj_grids['nav_space'] = flr_grid

    #x = event.metadata['agent']['position']['x']
    #y = event.metadata['agent']['position']['y']
    #z = event.metadata['agent']['position']['z']
    #obj_grids['agent_pos'] = {'x':x,'y':y,'z':z}
    obj_grids['min_pos'] = {'mx':m_x,'mz':m_z}

    return obj_grids


def prettyprint(mat,argmax = False, locator = [-1,-1,-1]):
    for j in range(mat.shape[1]):
        d = repr(j)
        if j<10:
            d = '0'+d

        print(d,end = '')
        print("  ",end = '')
    print(" ")
    print(" ")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            d = 0
            if argmax:
                d = np.argmax(mat[i,j,:])
                #d = np.max(mat[i,j,:])
            else:
                d = repr(int(mat[i,j]))
                if locator[0]==i and locator[1]==j:
                    if locator[2]==0:
                        d = '>' #"\u2192" #right arrow
                    if locator[2]==270:
                        d = '^' #"\u2191" #up arrow
                    if locator[2]==90:
                        d = 'v' #"\u2193" #down arrow
                    if locator[2]==180:
                        d = '<' #"\u2190" #left arrow


            print(d,end = '')
            print("   ",end = '')
        print(" --",repr(i))
        #print(" ")

def surrounding_patch(agentloc, labeled_grid, R = 16, unreach_value = -1): #returns a visibility patch centered around the agent with radius R
    #unreach_value = -1
    mat = labeled_grid

    position = agentloc

    r=copy.copy(R)
    init_shape = copy.copy(mat.shape)
    p = copy.copy(position)
    
    while position[0]-r<0: #append black columns to the left of agent position
        #print("Increasing columns to left ")
        mat = np.insert(mat,0, unreach_value,axis=1)
        r-=1
        p[0]+=1
    r=copy.copy(R)
    while position[0]+r>init_shape[1]-1: #append blank columns to the right of the agent position
        #print("Increasing columns to right")
        mat = np.insert(mat,mat.shape[1], unreach_value,axis=1)
        r-=1
    r=copy.copy(R)
    while position[1]-r<0:
        #print("Increasing rows above")
        mat = np.insert(mat,0, unreach_value,axis=0) #append blank rows to the top of the agent position
        r-=1
        p[1]+=1
    r=copy.copy(R)
    while position[1]+r>init_shape[0]-1:
        #print("Increasing rows below")
        mat = np.insert(mat,mat.shape[0], unreach_value,axis=0) #append blank columns to the bottom of the agent position
        r-=1
    #print("mat shape ",mat.shape) #outputs (33x33)
    return mat[p[1]-R:p[1]+R+1, p[0]-R:p[0]+R+1]

def target_navigation_map(o_grids, obj, agentloc, grid_size = 32, unk_id = 0,flr_id = 1, tar_id = 2, obs_id = 3, verbose = False):
    m = o_grids['nav_space']
    m = np.where(m==0,m,flr_id) #just to reinforce that the navigable spaces have the specified flr_id

    #==========================
    #if only asking about navigable space and not interested to navigate to a specific target object
    if obj=="nav_space":
        #print("Got nav_space in gtmaps line 200")
        '''
        for n in o_grids.keys():
            if n!="nav_space":
                m = np.where(o_grids[n]==0,m,obs_id)
        '''
        m = np.where(m!=0,m,obs_id)
        agentloc = [int((agentloc['z']-o_grids['min_pos']['mz'])/0.25), int((agentloc['x']-o_grids['min_pos']['mx'])/0.25)]
        if verbose:
            print("Got grid agent location from agentloc ",agentloc)
        m = surrounding_patch(agentloc, m, R=int(grid_size/2), unreach_value = unk_id)
        return m

    #two different modes of searching (if exact id is passed it is sometimes helpful if multiple objects of same type- ex- multiple chairs)
    if '|' not in obj:
        searchkey = obj+'|'
    else:
        searchkey = obj
    #==========================
    #if only asking about navigating to a specific target object
    for n in o_grids.keys():
        if searchkey in n:
            if verbose:
                print("Got exact objectid ",n)
            t = tar_id*o_grids[n]
            m = np.where(t==0,m,tar_id)
        '''
        else:
            o = obs_id*o_grids[n]
            m = np.where(o==0,m,obs_id)
        '''
    #identify obstacle locations
    m = np.where(m!=0,m,obs_id)
    #center the map according to agent location - agentloc
    #3d position supplied by simulator need to be swapped in grid order - z gets first position and x gets 2nd position
    agentloc = [int((agentloc['z']-o_grids['min_pos']['mz'])/0.25), int((agentloc['x']-o_grids['min_pos']['mx'])/0.25)]
    if verbose:
        print("Got grid agent location from agentloc ",agentloc)
    m = surrounding_patch(agentloc, m, R=int(grid_size/2), unreach_value = unk_id)

    return m

def manual_label(room): #function for manually correcting wrong maps (computed automatically)
    #fname = '/home/hom/Desktop/ai2thor/mapping/gcdata/'+repr(room)+'.npy'
    fname = '/ai2thor/mapper/data/targets/'+repr(room)+'.npy'
    o_grids = np.load(fname,allow_pickle = 'TRUE').item()
    print("The fixed obstructions map")
    prettyprint(o_grids['fixed_obstructions']) #grid with 0s and 1s showing navigable spaces with all objects in the room removed 
    
    def exists(o_grids,obj):
        for n in o_grids.keys():
            if obj+'|' in n:
                return True
        return False

    obj = ""
    while True:
        obj = input("Enter the name of the object you want to insert ")
        
        if obj=='space':
            p = input("Space on top(t),bottom(b),left(l) or right (r) ?")
            num = input("Number of tiles (eg-1,2,3) ? ")
            unreach_value = 0

            m_x,m_z = o_grids['min_pos']['mx'], o_grids['min_pos']['mz']

            for n in o_grids.keys():
                mat = o_grids[n]
                try:
                    isarray = mat.shape
                except:
                    #the final element in the dictionary is not a numpy array its stores the min and max grid position in the map 
                    #so skip this
                    continue
                for _ in range(int(num)):
                    if p=='t':
                        mat = np.insert(mat,0, unreach_value,axis=0) #append blank rows to the top of the agent position
                    if p=='b':
                        mat = np.insert(mat,mat.shape[0], unreach_value,axis=0) #append blank columns to the bottom of the agent position
                    if p=='l':
                        mat = np.insert(mat,0, unreach_value,axis=1) #append blank columns to left of agent position
                    if p=='r':
                        mat = np.insert(mat,mat.shape[1], unreach_value,axis=1) #append blank columns to the right of the agent position

                o_grids[n] = mat

            if p=='t':
                o_grids['min_pos'] = {'mx':m_x-int(num)*0.25,'mz':m_z}
            if p=='l':
                o_grids['min_pos'] = {'mx':m_x,'mz':m_z-int(num)*0.25}
            if p=='b':
                o_grids['min_pos'] = {'mx':m_x,'mz':m_z}
            if p=='r':
                o_grids['min_pos'] = {'mx':m_x,'mz':m_z}


            save = input("Save data ? (y/n)")
            if save=='y':
                np.save(fname,o_grids) #overwrites the existing one
            continue



        if obj=='bye':
            break
        
        if obj!='space' or obj!='bye':
            if exists(o_grids,obj):
                overwrite = input("This name is already taken want to overwrite ? (y/n) ")
                
                mat = np.zeros_like(o_grids['fixed_obstructions'])

                for n in o_grids.keys():
                    if obj+'|' in n:
                        print("Found ",n)
                        mat+=o_grids[n]
                prettyprint(mat)

                if overwrite=='n':
                    continue
                if overwrite=='y':
                    obj = input("In that case enter the exact objectid by searching from above ")
            else:
                o_grids[obj+'|'] = np.zeros_like(o_grids['fixed_obstructions'])

            print("You can enter the corners like this ...")
            print("<top left corner column number, top left corner row number _ bottom right corner column number, bottom right corner row number>")
            corners = input("Enter the corners (eg- 0,0_7,8) ")
            c1,c2 = corners.split('_')
            [c1x,c1y], [c2x,c2y] = c1.split(','), c2.split(',')
            print("Got coordinates ",c1x,c1y,c2x,c2y)
            
            try:
                if '|' in obj:
                    o_grids[obj][int(c1y):int(c2y)+1,int(c1x):int(c2x)+1] = 1.0
                else:
                    o_grids[obj+'|'][int(c1y):int(c2y)+1,int(c1x):int(c2x)+1] = 1.0
            except:
                print("Error occured with accessing key")
                if '|' in obj:
                    o_grids[obj] = np.zeros_like(o_grids['fixed_obstructions'])
                    o_grids[obj][int(c1y):int(c2y)+1,int(c1x):int(c2x)+1] = 1.0
                else:
                    o_grids[obj+'|'] = np.zeros_like(o_grids['fixed_obstructions'])
                    o_grids[obj+'|'][int(c1y):int(c2y)+1,int(c1x):int(c2x)+1] = 1.0

            print("Modified ",obj)
            if '|' in obj:
                prettyprint(o_grids[obj])
            else:
                prettyprint(o_grids[obj+'|'])

        save = input("Save data ? (y/n)")
        if save=='y':
            np.save(fname,o_grids) #overwrites the existing one


