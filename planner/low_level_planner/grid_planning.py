import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))
from language_understanding import equivalent_concepts as eqc

from mapper import test_gcn
from mapper import projection as proj
from mapper import panorama as pan
from mapper import params as mparams

from planner.low_level_planner import move_camera
import sys


GOTOOBJS = eqc.GOTOOBJS
#add a new parameter to the function called box_area/box_num
#if earlier occupancy_grid was called to search for an object
#and now we are simply trying to go to the other side of that object (already did target refinement for multiple targets)
#we can use the box area to directly refine the target
#WARNING - when using bounding box number it may not associate with the same previous bounding box number if the agent has drifted too far
def occupancy_grid(env, target_object, ref_object, localize_params, hallucinate = [0,0]): #displays the true map of the environment around the agent for a focussed object + floor
    print("(grid_planning.py -> occupancy_grid)")
    ref_object = ref_object.split(',')
    if isinstance(ref_object, list):
        ref_object = ref_object[0]
    #when using perfect map reupdate the agent location everytime this function gets called
    if isinstance(localize_params['room'],int):
        x,y,z = env.get_position()
        localize_params['position']= [x, z]

    labeled_grid = test_gcn.estimate_map(target_object,localize_params = localize_params)
    #correct the camera tilt everytime before starting navigation
    move_camera.set_default_tilt(env)
    #without map the agent spirals the camera upward to capture an image so reset the camera position
    if isinstance(localize_params['room'],int)==False:
        move_camera.set_default_tilt(localize_params['room'])
    
    if hallucinate[0]!=0 or hallucinate[1]!=0: 
        print("Target is visible, but too far away for depth sensor, halucinating the position")
        visualize = True
        p = int(labeled_grid.shape[0]/2)
        q = int(labeled_grid.shape[1]/2)
        #labeled_grid[p+int(hallucinate[0]*p), q+int(hallucinate[1]*q)] = 4 # object_codes[target_object] = 4 (earlier)
        labeled_grid[p+int(hallucinate[0]*p), q+int(hallucinate[1]*q)] = mparams.semantic_classes['tar'] # object_codes[target_object] = 4

    ######################   get the target bounding coordinates of the object to set graph search target     ###################
    #search_grid = np.where(labeled_grid==2,labeled_grid,0) #only places where target object is mapped is nonzero
    search_grid = np.where(labeled_grid==mparams.semantic_classes['tar'],labeled_grid,0) #only places where target object is mapped is nonzero
    props = regionprops(label(np.asarray(search_grid,dtype = np.int)))
    print("Number of regionprops found for ",target_object, " is = ",len(props))

    if len(props)!=0:
        env.memory['navigated'].append(target_object)

    bbox = props[0].bbox #need to refine when multiple objects are present (for example drawers)

    if len(props)>1:

        #if this issue has been encountered before and been registered to agent menory
        if  env.memory['tracked_targets'][-1]==target_object or env.memory['tracked_refinements'][-1]==ref_object:
            print("Recalling target refinement from earlier ")
            bbox_num = env.memory['tracked_target_nums'][-1]
            bbox = props[bbox_num].bbox

        else:
            #otherwise 
        
            dists = []
            ref_dists = []

            try:
                #Try to rotate camera in a spiral and get approximate map of the refinement object
                #after that compare the distances between the bounding box of the refinement object
                #and the various center to center distances to the bounding boxes of the multiple target objects
                #then pick the target object as the one with the minimum c2c distance
                print("Number of regionprops>1, trying to refine target based on refinement object ")
                
                if ref_object in GOTOOBJS: #means no need to use approximate technique
                    nav_map_ref = test_gcn.estimate_map(ref_object,localize_params = localize_params)
                    move_camera.set_default_tilt(env)

                else: #object was too small to map, need to use approximate technique
                    gridsize = mparams.grid_size
                    panorama = pan.rotation_image(env, objects_to_visit = [], debug = False) #gets a panorama image of everything thats visible
                    move_camera.set_default_tilt(env)
                    camera_proj = proj.bevmap(panorama,grid_size = gridsize, debug = False)

                    nav_map_r = proj.input_navigation_map(camera_proj, ref_object, grid_size = mparams.grid_size, 
                                                        unk_id = mparams.semantic_classes['unk'],
                                                        flr_id = mparams.semantic_classes['flr'], 
                                                        tar_id = mparams.semantic_classes['tar'], 
                                                        obs_id = mparams.semantic_classes['obs'])
                    #nav_map is an array of params.grid_size x params.grid_size x 4 (4 classes unk,flr,tar and obs)| contains values between 0 and 1
                    print("approximate map of the refinement object ",ref_object)
                    nav_map_ref = np.argmax(nav_map_r, axis=2)



                #proj.starviz(nav_map_ref)
                #sys.exit(0)
                search_grid_ref = np.where(nav_map_ref==mparams.semantic_classes['tar'],nav_map_ref,0) #only places where target object is mapped is nonzero
                props_ref = regionprops(label(np.asarray(search_grid_ref,dtype = np.int)))
                ref_box = props_ref[0]
                for p in props:
                    bbox = p.bbox 
                    bbox_r = ref_box.bbox

                    corners = bbox-np.array(bbox_r) #top left and bottom right corners of the target obj
                    av_dist = (corners[0]+corners[2])**2 + (corners[1]+corners[3])**2
                    #av_dist = (corners[0]+corners[1]+corners[2]+corners[3])/4.0 #not a fool proof way, because sizes of similar objects can still be different
                    ref_dists.append(av_dist)
                #proj.prettyprint(nav_map,argmax = True)
                min_d = np.argmin(ref_dists)
                bbox = props[min_d].bbox
                print("picking target region based on refinement object",bbox)
                #sys.exit(0)



            except:
                print("Number of regionprops>1, refining targets based on refinement objects failed, so picking closest region ")
                for p in props:
                    bbox = p.bbox 
                    corners = bbox-np.array([int(labeled_grid.shape[0]/2),int(labeled_grid.shape[1]/2)]*2) #top left and bottom right corners of the target obj
                    av_dist = (corners[0]+corners[2])**2 + (corners[1]+corners[3])**2
                    #av_dist = (corners[0]+corners[1]+corners[2]+corners[3])/4.0 #not a fool proof way, because sizes of similar objects can still be different
                    dists.append(av_dist)
                min_d = np.argmin(dists)
                bbox = props[min_d].bbox

            env.memory['tracked_targets'].append(target_object)
            env.memory['tracked_refinements'].append(ref_object)
            env.memory['tracked_target_nums'].append(min_d)
        


    
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
