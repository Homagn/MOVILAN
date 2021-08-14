import os
import sys
import copy

os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))
from language_understanding import equivalent_concepts as eqc

import numpy as np
import math

import planner.low_level_planner.resolve as resolve
from planner.low_level_planner import move_camera
field_of_view = resolve.field_of_view
RECEPS = eqc.RECEPS

def location_in_fov(env, mask_image, depth_image):
    m_i = copy.copy(mask_image)
    d_i = copy.copy(depth_image)

    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

    Areas = {}
    Cents = {}
    lf = {}
    mf = {}
    rf = {}

    for c in unique_obs.tolist():
        pixel_pos = np.argwhere(m_i==c)
        pixel_area = len(pixel_pos.flatten())/3
        pixel_cent = np.mean(pixel_pos,axis=0)

        geom_dist = np.mean(d_i[pixel_pos]) #get the mean depth of the segmented object over all segmented pixels in the depth normals

        object_id = env.identify_segmented_color(c)

        if object_id!="Nothing":
            Areas[object_id] = pixel_area
            Cents[object_id] = pixel_cent

            if pixel_cent[1]>=0 and pixel_cent[1]<=100: #the pixel centroid of the segmented object is withing image columns 0 and 100
                lf[object_id] = geom_dist
            if pixel_cent[1]>100 and pixel_cent[1]<=200: #the pixel centroid of the segmented object is withing image columns 0 and 100
                mf[object_id] = geom_dist
            if pixel_cent[1]>200 and pixel_cent[1]<=300: #the pixel centroid of the segmented object is withing image columns 0 and 100
                lf[object_id] = geom_dist
        
    return lf,mf,rf, Areas, Cents

def object_in_fov(env,obj):
    #obj should be atleast one string in a list - eg - ["Pot"]
    all_visibles = []
    receptacle = ""
    #finding the receptacle object
    env.step(dict({"action": "LookUp"}))
    cnt = 0
    for i in range(3): # was 3 earlier 
        field = field_of_view(env)
        for k in field:
            all_visibles.append(k)
            try:
                k1 = obj[0]
            except:
                k1 = "Nothing"
            if k1+'|' in k:
                print("Exact object iD for ",k1," is ",k)
                receptacle = k
                break
        if receptacle!="":
            break
        env.step(dict({"action": "LookDown"}))
        if env.actuator_success():
            cnt+=1
    
    for j in range(cnt-1):
        env.step(dict({"action": "LookUp"}))

    move_camera.set_default_tilt(env)
    return env, receptacle,all_visibles

def common_connected_component(mask_image,env,ref_objs): 
    #used for answering questions like- what is the exact objectid on top of which there is a remote control ?
    #ref_objs generic names (not exact)
    #take the mask image and find out the component that is connected to most of the objects in the ref_objs list
    #eg - ref objs is creditcard and television and both of them are on top of Dresser, then it should return 
    #the exact object id of dresser

    #Algo
    '''
    Pass very small rectangle patches in striding fashion over the segmented image
    count the number of rectangles that have 50% of one color (on of the refinement objs) and 50% of another color (probably the object they are sitting on)
    If the number of such special rectangles exceed threshold then isolate the corresponding object
    '''
    print("manipulation_signatures.py -> common_connected_component ")
    print("Got refinement objects ",ref_objs)
    #make sure using instance segmentation, then only d['name'] will provide unique objectid
    m_i = copy.copy(mask_image)
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

    suspect_obj = []
    refinement_obj = []

    suspect_col = []
    refinement_col = []



    for c in unique_obs.tolist():
        obj = env.identify_segmented_color(c)
        if obj!="Nothing":
            suspect_obj.append(obj)
            suspect_col.append(c)

            for r in ref_objs:
                if r+'|' in obj:
                    #refinements[d['name']] = d['color']
                    refinement_obj.append(obj)
                    refinement_col.append(c)



    #some adjustable parameters
    patch_size = 20
    area_ratio = 0.4
    thresh_num = 5 #atleast 10 boxes should point to the commmon connected object (suspect)

    suspects = {}

    for i in range(0,m_i.shape[0],patch_size):
        for j in range(0,m_i.shape[1],patch_size): #small square patches of size 20x20
            patch = m_i[i:i+patch_size,j:j+patch_size,:]
            unique_cols = np.unique(patch.reshape(-1,patch.shape[2]),axis=0)

            ref = []
            sus = []

            for c in unique_cols.tolist():
                pos = np.argwhere(m_i==c)
                area = len(pos.flatten())/3
                if float(area/(patch_size*patch_size))>=area_ratio: #this color occupies 50% of the small patch
                    if c in refinement_col: #the color that occupies 50%, is that of a refinement obj?
                        ref.append(c)
                    else:
                        sus.append(c)
            
            if ref!=[] and sus!=[]: #means atleast one refinement obj and atleast one suspect object co occupy >=80% area of the small patch
                for s in sus:
                    so = suspect_obj[suspect_col.index(s)]
                    if so not in suspects.keys():
                        suspects[so] = 0
                    else:
                        suspects[so] +=1

    possib = []
    for s in suspects.keys():
        if suspects[s]>=thresh_num and s[:s.index('|')] in RECEPS:
            possib.append(s)

    print("Possible connected components ",possib)
    return possib[0]

