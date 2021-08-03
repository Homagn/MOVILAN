import os
import sys
import cv2
import numpy as np
import copy
from os import path



def normalize_segment1(vis_meta, mask_image): #make sure all segmentations follow the same color map
        seg_colors = []
        seg_names = []
        masks = []
        
        m_i = copy.copy(mask_image)
        unique_obs = np.unique(m_i.reshape(-1,m_i.shape[2]),axis=0)
        event_colors = vis_meta['colors']
        for d in event_colors:
            if d['color'] in unique_obs.tolist(): #and d['name'] in objects_to_visit: #sometimes selected objects are way outside of field of view
            #if d['color'] in unique_obs.tolist() and d['name'] in visibles:
                seg_colors.append(d['color'])
                seg_names.append(d['name'])
                rcolor = d['color']
                e = 255*np.ones((mask_image.shape[0],mask_image.shape[1]))
                d = np.where((mask_image[:,:,0]==rcolor[0])&(mask_image[:,:,1]==rcolor[1])&(mask_image[:,:,2]==rcolor[2]),e,e*0.0)
                masks.append(d/255.0)

        return masks,seg_colors,seg_names


def rotation_image(env, objects_to_visit = [], debug = False):

    event = env.step(dict(action = 'GetReachablePositions'))

    starting_position = {"x": event.metadata['agent']['position']['x'],
                        "y": event.metadata['agent']['position']['y'],
                        "z": event.metadata['agent']['position']['z']}

    print("Got starting position accroding to task ",starting_position)

    if path.exists('test.npz')==True and debug:
        return []
    if path.exists('test.npz')==False and debug:
        print("In debug mode but test file does not exist creating one")


    if objects_to_visit==[]:
        objects_to_visit = [d['name'] for d in event.metadata['colors']] #get access to all possible segmentations in the room

    
    rot_init = 0
    rot_step = 50
    pix = 300
    #numpix = int(360/rot_step) #should be actually
    numpix = 6 #observed empirically
    #panorama to be resized later to 768*128
    panorama_rgb = np.zeros((900,300+pix*numpix,3)) #empirically estimated that each 10 degree rotation adds 50 new (viewing) width pixels (columns in the data)
    panorama_seg = np.zeros((900,300+pix*numpix,3))
    panorama_seg_instance = np.zeros((900,300+pix*numpix,3))
    panorama_depth = np.zeros((900,300+pix*numpix))
    #event = env.step(dict({"action": "LookDown"}))
    custom_rot = {"action": "TeleportFull","horizon": 60,"rotateOnTeleport": True,"rotation": 0,
                    "x": event.metadata['agent']['position']['x'],
                    "y": event.metadata['agent']['position']['y'],
                    "z": event.metadata['agent']['position']['z']}
    event = env.step(dict(custom_rot))
    m=0
    for look in range(3):
        rot_init = 0

        rgb = event.frame[:, :, ::-1]
        seg = event.class_segmentation_frame
        seg_instance = event.instance_segmentation_frame
        depth = event.depth_frame
        depth = depth * (255 / 10000)
        depth = depth.astype(np.uint8) #2D 1 channel array


        if m==0:
            panorama_rgb[(-300-300*m):,-300:,:] = rgb
            panorama_seg[(-300-300*m):,-300:,:] = seg
            panorama_seg_instance[(-300-300*m):,-300:,:] = seg_instance
            panorama_depth[(-300-300*m):,-300:] = depth
        else:
            panorama_rgb[(-300-300*m):(-300-300*(m-1)),-300:,:] = rgb
            panorama_seg[(-300-300*m):(-300-300*(m-1)),-300:,:] = seg
            panorama_seg_instance[(-300-300*m):(-300-300*(m-1)),-300:,:] = seg_instance
            panorama_depth[(-300-300*m):(-300-300*(m-1)),-300:] = depth

        n = 0
        for i in range(numpix):
            n+=1
            #save images seperately
            #cv2.imwrite('panorama_data/DFSdatanew/rgb_'+repr(counter)+'.png', rgb)
            #counter+=1

            custom_rot = {"action": "TeleportFull","horizon": 60-30*m,"rotateOnTeleport": True,"rotation": rot_init-rot_step,
                            "x": event.metadata['agent']['position']['x'],
                            "y": event.metadata['agent']['position']['y'],
                            "z": event.metadata['agent']['position']['z']}
            event = env.step(dict(custom_rot))
            rot_init-=rot_step
            
            rgb = event.frame[:, :, ::-1][:,0:pix,:]
            seg = event.class_segmentation_frame[:,0:pix,:]
            seg_instance = event.instance_segmentation_frame[:,0:pix,:]
            depth = event.depth_frame[:,0:pix]
            depth = depth * (255 / 10000)
            depth = depth.astype(np.uint8) #2D 1 channel array

            if m==0:
                panorama_rgb[(-300-300*m):,(-300-pix*n):(-300-pix*(n-1)),:] = rgb
                panorama_seg[(-300-300*m):,(-300-pix*n):(-300-pix*(n-1)),:] = seg
                panorama_seg_instance[(-300-300*m):,(-300-pix*n):(-300-pix*(n-1)),:] = seg_instance
                panorama_depth[(-300-300*m):,(-300-pix*n):(-300-pix*(n-1))] = depth
            else:
                panorama_rgb[(-300-300*m):(-300-300*(m-1)),(-300-pix*n):(-300-pix*(n-1)),:] = rgb
                panorama_seg[(-300-300*m):(-300-300*(m-1)),(-300-pix*n):(-300-pix*(n-1)),:] = seg
                panorama_seg_instance[(-300-300*m):(-300-300*(m-1)),(-300-pix*n):(-300-pix*(n-1)),:] = seg_instance
                panorama_depth[(-300-300*m):(-300-300*(m-1)),(-300-pix*n):(-300-pix*(n-1))] = depth
        #event = env.step(dict({"action": "LookUp"}))
        custom_rot = {"action": "TeleportFull","horizon": 60-30*(m+1),"rotateOnTeleport": True,"rotation": 0,
                        "x": event.metadata['agent']['position']['x'],
                        "y": event.metadata['agent']['position']['y'],
                        "z": event.metadata['agent']['position']['z']}
        event = env.step(dict(custom_rot))
        m+=1

    scene_state = {}

    scene_state["rgb"]= cv2.resize(panorama_rgb,(700,300)) #actual size - 2100x900, earlier 1800x900
    scene_state["seg"]= cv2.resize(panorama_seg,(700,300))
    scene_state["seg_instance"]= cv2.resize(panorama_seg_instance,(700,300))
    scene_state["depth"]= cv2.resize(panorama_depth,(700,300))
    #done obtaining spiral
    
    
    #Debugging 
    if debug:
        cv2.imshow("spiral panorama", scene_state["rgb"]/255.0) #show the spiral panorama rgb image
        cv2.waitKey(0)
        cv2.imshow("spiral panorama", scene_state["depth"]/255.0) #show the spiral panorama rgb image
        cv2.waitKey(0)
        cv2.imshow("spiral panorama", scene_state["seg"]/255.0) #show the spiral panorama rgb image
        cv2.waitKey(0)
    


    seg_names = []
    seg_colors = []
    masks = []
    #start = 0

    masks,sc,sn = normalize_segment1(event.metadata, scene_state["seg_instance"].astype(np.uint8))
    panorama = {"rgb":scene_state["rgb"],"seg": scene_state["seg"], "seg_instance": scene_state["seg_instance"],
                "depth": scene_state["depth"], "masks": masks, "seg_names":sn, "seg_colors":sc}

    '''
    if debug:
        np.savez('test.npz',
                rgb = scene_state["rgb"],seg = scene_state["seg"], seg_instance = scene_state["seg_instance"],
                depth = scene_state["depth"], masks = masks, seg_names = sn, seg_colors = sc)
    '''
    
    return panorama
        


