from disambiguation.image_vector import *
from disambiguation.im_download import *
import numpy as np
import cv2

absolute_loc = "/home/hom/Desktop/ai2thor/disambiguation/"

def google_confusion(event, resolve = "event_image"):
    #event is passed from ai2thor simulator
    if resolve=="event_image":
        print("Asked to resolve by event images ")
        print("We will see pairwise similarity with each segmented object in image and google downloaded image for the query")
        mask_image = event.instance_segmentation_frame
        rgb_image = np.array(event.frame[:, :, ::-1])
        query = input("Enter your query to associate ")
        

        download_images(rel_loc = absolute_loc, data=query,n_images = 5)

        m_i = copy.copy(mask_image)
        unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

        seg_colors = []
        seg_names = []

        for d in event.metadata['colors']:
            if d['color'] in unique_obs.tolist(): #and d['name'] in objects_to_visit: #sometimes selected objects are way outside of field of view
            #if d['color'] in unique_obs.tolist() and d['name'] in visibles:
                seg_colors.append(d['color'])
                seg_names.append(d['name'])
                rcolor = d['color']
                e = np.ones((mask_image.shape[0],mask_image.shape[1]))
                mask = np.where((mask_image[:,:,0]==rcolor[0])&(mask_image[:,:,1]==rcolor[1])&(mask_image[:,:,2]==rcolor[2]),e,e*0.0)

                mask = np.array(mask,dtype = np.uint8)
                mask = np.stack([mask,mask,mask],axis=2)
                
                print(d['name'])
                res = rgb_image*mask

                file = absolute_loc+"/downloads/test.jpg"
                #cv2.imwrite("cross_modal/downloads/test.jpg", res*255.0)
                cv2.imwrite(file, res*255.0)
                #time.sleep(0.2)
                s = similarity(rel_loc = absolute_loc, query = file, support = query)
                print("Average similarity with segmented object ",s)

    if resolve=="event_word":
        print("Asked to resolve by event words ")
        print("We will see pairwise similarity with google images downloaded for each object name in the scene and google images downloaded for your query")
        mask_image = event.instance_segmentation_frame
        rgb_image = np.array(event.frame[:, :, ::-1])
        unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)
        query = input("Enter your query to associate ")
        
        download_images(rel_loc = absolute_loc, data=query,n_images = 5)


        for d in event.metadata['colors']:
            if d['color'] in unique_obs.tolist(): #and d['name'] in objects_to_visit: #sometimes selected objects are way outside of field of view

                seg_name = d['name'][:d['name'].index('|')]
                download_images(rel_loc = absolute_loc, data=seg_name,n_images = 1)

                #res = cv2.imread("cross_modal/downloads/"+seg_name+repr(1)+".jpg")
                res = cv2.imread(absolute_loc+"downloads/"+seg_name+repr(1)+".jpg")
                file = absolute_loc+"/downloads/test.jpg"
                cv2.imwrite("cross_modal/downloads/test.jpg", res)
                cv2.imwrite(file, res)
                #time.sleep(0.2)
                s = similarity(rel_loc = absolute_loc, query = file, support = query)
                print("Average similarity of segmented object ",seg_name, " with google downloaded image for your query is ",s)