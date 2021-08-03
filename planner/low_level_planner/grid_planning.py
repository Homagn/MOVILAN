import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from mapper import test_gcn
from planner.low_level_planner import move_camera


def occupancy_grid(target_object, localize_params, hallucinate = [0,0]): #displays the true map of the environment around the agent for a focussed object + floor
    print("(grid_planning.py -> occupancy_grid)")
    labeled_grid = test_gcn.estimate_map(target_object,localize_params = localize_params)
    
    #without map the agent spirals the camera upward to capture an image so reset the camera position
    if isinstance(localize_params['room'],int)==False:
        move_camera.set_default_tilt(localize_params['room'])
    
    if hallucinate[0]!=0 or hallucinate[1]!=0: 
        print("Target is visible, but too far away for depth sensor, halucinating the position")
        visualize = True
        p = int(labeled_grid.shape[0]/2)
        q = int(labeled_grid.shape[1]/2)
        labeled_grid[p+int(hallucinate[0]*p), q+int(hallucinate[1]*q)] = 4 # object_codes[target_object] = 4

    ######################   get the target bounding coordinates of the object to set graph search target     ###################
    search_grid = np.where(labeled_grid==2,labeled_grid,0) #only places where target object is mapped is nonzero
    props = regionprops(label(np.asarray(search_grid,dtype = np.int)))
    print("Number of regionprops found for ",target_object, " is = ",len(props))

    bbox = props[0].bbox #need to refine when multiple objects are present (for example drawers)

    if len(props)>1:
        dists = []
        print("Number of regionprops>1, picking closest region ")
        for p in props:
            bbox = p.bbox 
            corners = bbox-np.array([int(labeled_grid.shape[0]/2),int(labeled_grid.shape[1]/2)]*2) #top left and bottom right corners of the target obj
            av_dist = (corners[0]+corners[2])**2 + (corners[1]+corners[3])**2
            #av_dist = (corners[0]+corners[1]+corners[2]+corners[3])/4.0 #not a fool proof way, because sizes of similar objects can still be different
            dists.append(av_dist)
        min_d = np.argmin(dists)
        bbox = props[min_d].bbox


    
    #bbox = bbox - np.array([0,0,1,1])
    corners = bbox-np.array([int(labeled_grid.shape[0]/2),int(labeled_grid.shape[1]/2)]*2) #top left and bottom right corners of the target obj
    
    up,left,down,right = corners[0],corners[1],corners[2],corners[3] #-1 because region props gives open interval for the bottom corner

    n_facing_pos = [int((up+down)/2), left-1] #translating to i,j index of the the labeled_grid matrix
    w_facing_pos = [down, int((left+right)/2)]
    s_facing_pos = [int((up+down)/2), right]
    e_facing_pos = [up-1, int((left+right)/2)] #new modifications-> added -1 to left and up in lines 339,342

    #print("bounding box ",bbox-np.array([int(labeled_grid.shape[0]/2),int(labeled_grid.shape[1]/2)]*2))
    #print("four corners (agent rel) of ",target_object, " ",corners)
    print("Facing grid positions north, west, south, east ",n_facing_pos, w_facing_pos, s_facing_pos, e_facing_pos)
    print(" ")
    return labeled_grid, [n_facing_pos,w_facing_pos,s_facing_pos,e_facing_pos]
