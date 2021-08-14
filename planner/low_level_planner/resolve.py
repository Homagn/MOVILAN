import numpy as np
from skimage.measure import regionprops, label

import sys
import os
import copy

os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from language_understanding import equivalent_concepts as eqc

GOTOOBJS = eqc.GOTOOBJS
CONFUSIONS = eqc.CONFUSIONS

def resolve_refinement(ref_objs,tar_objs):
    print("navigation_signatures.py -> resolve_refinement)")
    print("got ref_objs ",ref_objs)
    print("got tar objs ",tar_objs)
    if ref_objs!='':
        for r in ref_objs.split(','):
            if r in GOTOOBJS:
                return r
    #else:
    print("Failed to capture refinement objects, thus looking in target objects ")
    for r in tar_objs.split(','):
        if r in GOTOOBJS:
            return r


def field_of_view(env, verbose = False):
    #whatever unique object ids are visible based on entire panorama segmented image
    mask_image = env.get_segmented_image()
    m_i = copy.copy(mask_image)
    d_i = env.get_depth_image()
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)
    
    fov_dists = {}

    for c in unique_obs.tolist():
        pixel_pos = np.argwhere(m_i==c)
        pixel_area = len(pixel_pos.flatten())/3
        pixel_cent = np.mean(pixel_pos,axis=0)

        geom_dist = np.mean(d_i[pixel_pos]) #get the mean depth of the segmented object over all segmented pixels in the depth normals

        object_id = env.identify_segmented_color(c)

        if object_id!="Nothing":
            fov_dists[object_id] = geom_dist


    return fov_dists

def resolve_confusions(obj, env, ref_obj = ""):
    print("(navigation_signatures.py -> resolve_confusions)")
    #Resolve sequence :
    #1. based on whether the object is present in the room event meta (was earlier- removed this check now)
    #2. based on whether the object is visible directly in the agents view/ supplied refinement object criteria
    #3. if still multiple candidates, 
    #   if 
    #   refinement object is provided filter the target object to be the one closest to it
    #   else
    #    which object is closer distance to the agent

    conf = CONFUSIONS[obj]
    
    conf_copy = copy.copy(conf)

    if len(conf)>1: #still need to resolve (step 2 now)
        rot_step = 36
        visibles = {}
        for _ in range(10):
            #x,y,z = env.get_position()
            rot = env.get_rotation()
            env.custom_rotation(30,rot_step)
            
            
            # if two identical objects are present in the scene then their object ids must be different
            visibles.update(field_of_view(env)) #merge the two dictionaries

        
        for c in conf:
            if c not in visibles.keys():
                conf_copy.pop(conf_copy.index(c))
        print("Modified confusion objects based on panorama visibility ",conf_copy)
    
    #print("conf_copy ",conf_copy)
    if len(conf_copy)==1:
        conf = [conf_copy[0]]
    else:
        conf = conf_copy
    #print("conf after ",conf)
    
    
    if len(conf)>1: #still need to resolve (step 3 now)
        dists = []
        names = []
        rf = ""

        print("In len(conf)>1, ref_obj ",ref_obj)
        ref_obj = ref_obj.split(',') #get a list out of comma seperated objects

        for o in list(visibles.keys()):
            if isinstance(ref_obj, list): #multiple refinement objects have been provided
                for r in ref_obj:
                    if r!="" and r+'|' in o:
                        rf = o
                        print("Matched refinement object ",rf)

            else:
                if ref_obj!="" and ref_obj+'|' in o:
                    rf = o
                    print("Matched refinement object ",rf)

        if rf!="":
            for c in conf:
                d_rf = visibles[rf]
                d_c = visibles[c]



                #a,b = localize(event,m_x,m_z) #agent position
                if c!=rf:
                    print("object ",c," depth normal dist ",visibles[c])
                    print("refinement ",rf," depth normal dist ",visibles[rf])
                    #print("p,q,x,y ",p,q,x,y)
                    dists.append((d_rf-d_c)**2)
                    print("Distance between ",rf," and ",c," is ",dists[-1])
                    names.append(c)
        else: 
            for c in conf:
                d_c = visibles[c]

                print("object ",c," depth normal distance ",d_c)
                #print("a,b,x,y ",a,b,x,y)
                dists.append(d_c) #depth normals shouldnt be negative
                names.append(c)
        
        dists = np.array(dists)
        conf = [names[np.argmin(dists)]]
    


    print("resolved confusion object to be ",conf[0])
    print(" ")
    #sys.exit(0)
    return conf[0] #will return the actual objectid (the object to navigate to) in the event meta this time 