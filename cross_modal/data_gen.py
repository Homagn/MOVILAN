import numpy as np
import cv2
import math
import copy 
import os
from os import path
import sys
import glob
import json
import random
import traceback

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

from env.thor_env import ThorEnv
import ai2thor.controller
import random
from datetime import datetime as dt
import time as t

import webcolors

INVEN_OBJS = ["CellPhone","Pen","Pencil","TissueBox","Statue","Watch", "Bowl", "Mug", "CD", "Laptop", "BaseballBat", "AlarmClock",
                'Box', 'Cloth','Pillow', 'TennisRacket','Vase','Cup',"CreditCard","BasketBall","Book","Plate", "KeyChain", "RemoteControl"]

GOTOOBJS = ["SideTable","Dresser","Desk","table","Shelf","DeskLamp","Bed","table","TVStand","Television","Sofa", "CounterTop", "CoffeeTable"]

STACKS = ["Drawer","Shelf"] #These are objects generally present as a bunch 

def desc1(event,f,fov):
    #This function describes whether an object is contained (or sitting on top of) in another object
    #for objects like shelf that can contain objects
    if f[:f.index('|')] in STACKS: #seperate special function for stackable objects like shelf and drawer
        for ob in fov:
            p = parent_receptacle(event,ob)
            if p==f:
                sent = f+" containing "+ob
                return [sent]
    #for other objects
    sent = f
    p = parent_receptacle(event,f) #straight away uses event meta data for flawless and fast analysis
    if f[:f.index('|')] in INVEN_OBJS and p!=[]:
            #print(f," on ",p)
            sent = f+" on "+p
    return [sent]

def desc2(tc,f):
    #This function describes the general position of the object up/down/left/right wrt the agent
    
    #Describe the relative location of the chosen object f with respect to the agent
    sent = []
    if tc[0]>200:
        #print(f, "is to your right side ")
        sent.append(f+" your right")
    if tc[0]<50:
        #print(f, "is to your leftside ")
        sent.append(f+" your left")

    if f[:f.index('|')] in INVEN_OBJS: #bed is upwards or desk is upwards doesnt have any meaning
        if tc[1]>200:
            #print(f, "is below")
            sent.append(f+" is below")
        if tc[1]<50:
            #print(f, "is upwards ")
            sent.append(f+" is upwards")
    
    if sent!=[]:
        return [random.choice(sent)]
    else:
        return [f]

def desc3(oc,f):
    #For an object f, this function describes positions of all visible objects g wrt it

    #Dont describe if the object f is something like a shelf (will generate lots of useless descriptions)
    if f[:f.index('|')] in STACKS: #seperate special function for stackable objects like shelf and drawer
        return [f]
    
    #Describe pairs of relative location informations for f with every other visible object (o) 
    sent = []
    for o in oc.keys():
        _,_,d,x,y = oc[o][0], oc[o][1], oc[o][2], oc[o][3], oc[o][4]

        if math.fabs(d)<0.25: #0.25 (or 1 agent step size) is the threshold
            d = 0
        

        if f[:f.index('|')] in INVEN_OBJS and o[:o.index('|')] in INVEN_OBJS:
            if math.fabs(x)>=math.fabs(y):
                if math.fabs(x)<=150: #less than 150 pixels apart
                    if x>0:
                        #print(f, "is to the left of ",o)
                        sent.append(f+ " left "+o)
                    if x<0:
                        #print(f, "is to the right of ",o)
                        sent.append(f+ " right "+o)
            if math.fabs(y)>math.fabs(x):
                if math.fabs(y)<=150: #less than 150 pixels apart
                    if y>0:
                        #print(f, "is below ",o)
                        sent.append(f+ " below "+o)
                    if y<0:
                        #print(f, "is above ",o)
                        sent.append(f+ " above "+o)
        #if d!=0:
        if f[:f.index('|')] in GOTOOBJS and o[:o.index('|')] in GOTOOBJS:
            if math.fabs(x)>=math.fabs(y):
                if math.fabs(x)<=150: #less than 150 pixels apart
                    if x>0:
                        #print(f, "is to the left of ",o)
                        sent.append(f+ " left "+o)
                    if x<0:
                        #print(f, "is to the right of ",o)
                        sent.append(f+ " right "+o)
            if math.fabs(y)>math.fabs(x): #only this part is different
                if math.fabs(y)<=150: #less than 150 pixels apart
                    if y>0:
                        #print(f, "is just before ",o)
                        sent.append(f+ " before "+o)
                    if y<0:
                        #print(f, "is just after ",o)
                        sent.append(f+ " after "+o)
    if sent!=[]:
        return [random.choice(sent)]
    else:
        return [f]

def desc4(oc,f):
    #If multiple objects of the same type (eg 2 cups on a table) are present describe group relative positions
    #Describe left most,rightmost, furthermost, topmost, etc properties among objects of same group


    objtype = f[:f.index('|')]
    X,Y,D = [],[],[]
    for o in oc.keys():
        if objtype+'|' in o:
            _,_,d,x,y = oc[o][0], oc[o][1], oc[o][2], oc[o][3], oc[o][4]
            if math.fabs(y)>50:
                Y.append(y)
            if math.fabs(d)>0.1:
                D.append(d)
            X.append(x)

    def group_relative(X,f,rel = ["left","right"]):
        sent = ""
        if X!=[]:
            X = [1 if a>0 else -1 for a in X]
            if all(a<0 for a in X):
                if len(X)>2:
                    #print(f, "is the ",rel[1],"most ",f[:f.index('|')])
                    sent = f+ " "+rel[1]+" most "+f[:f.index('|')]
                else:
                    #print(f, "is the ",rel[1]," ",f[:f.index('|')])
                    sent = f+ " "+rel[1]+" "+f[:f.index('|')]
            
            if all(a>0 for a in X):
                if len(X)>2:
                    #print(f, "is the ",rel[0],"most ",f[:f.index('|')])
                    sent = f+ " "+rel[0]+" most "+f[:f.index('|')]
                else:
                    #print(f, "is the ",rel[0]," ",f[:f.index('|')])
                    sent = f+ " "+rel[0]+" "+f[:f.index('|')]

            if sum(X)==0 and len(X)>=2:
                #print(f, "is the middle ",f[:f.index('|')])
                sent = f+ " middle "+f[:f.index('|')]
        return sent

    s1 = group_relative(X,f,rel = ["left","right"])
    s2 = group_relative(Y,f,rel = ["bottom","top"])
    s3 = group_relative(D,f,rel = ["closer","farther"])

    if s1=="" and s2=="" and s3=="":
        return [f]
    else:
        S=[]
        if s1!="":
            S.append(s1)
        if s2!="":
            S.append(s2)
        if s3!="":
            S.append(s3)

        return [random.choice(S)]

def desc5(event,oc,f):
    #If multiple objects of the same type (eg 2 cups on a table) are present describe group relative positions
    #Eg- If multiple cups are present on a table and target cup is closest to a pencil, say that the cup is the cup that is closest to the pencil
    
    if f[:f.index('|')] in STACKS: #again neglecting this function for objects liek shelf because it will create too many useless descriptions
        return [f]

    p = parent_receptacle(event,f) #straight away uses event meta data for flawless and fast analysis
    sent = []

    objtype = f[:f.index('|')]
    group_objs = [f]
    for o in oc.keys():
        if objtype+'|' in o:
            group_objs.append(o)

    group_target = {}
    
    for o in event.metadata['objects']:
        if o['objectId'] in oc.keys() and o['objectId'] in group_objs:
            x1,y1,z1 = o['position']['x'], o['position']['y'], o['position']['z']
            group_target[o['objectId']] = [x1,y1,z1]
        if o['objectId'] == f:
            x1,y1,z1 = o['position']['x'], o['position']['y'], o['position']['z']
            group_target[o['objectId']] = [x1,y1,z1]

    #print("Got group targets ",group_target)

    for o in event.metadata['objects']:
        if o['objectId'] in oc.keys() and o['objectId'] not in group_objs:
            x,y,z = o['position']['x'], o['position']['y'], o['position']['z']
            dists = {} #stores how far each object is from the target object
            for gt in group_target.keys():
                x1,y1,z1 = group_target[gt][0],group_target[gt][1],group_target[gt][2]

                dists[gt] = (x1-x)**2 + (y1-y)**2 + (z1-z)**2

            sorted_order = sorted(dists.items(), key = lambda x:x[1])
            #print("object ",o['objectId'])
            #print("sorted_order ",sorted_order)
            #sorted_order = list(sorted_order.keys())
            if o['objectId'][:o['objectId'].index('|')] not in STACKS and o['objectId']!=p:
                if sorted_order[0][0]==f and len(sorted_order)>1:
                    #print(f," is the ",f[:f.index('|')]," that is closest to ",o['objectId'])
                    #sent.append(f+" is the "+f[:f.index('|')]+" that is closest to "+o['objectId'])
                    sent.append(f+" closest "+o['objectId'])
                if sorted_order[-1][0]==f and len(sorted_order)>1:
                    #print(f," is the ",f[:f.index('|')]," that is farthest from ",o['objectId'])
                    #sent.append(f+" is the "+f[:f.index('|')]+" that is farthest from "+o['objectId'])
                    sent.append(f+" farthest "+o['objectId'])
    if sent!=[]:
        return [random.choice(sent)]
    else:
        return [f]

def desc6(event,f):
    #WARNING ! does not work for all objects gives wildly different color names sometimes
    #tries to get the commonly know color name for the average rgb tuple extracted for the regions from the target object f
    sent = ""
    mask_image = event.instance_segmentation_frame
    rgb_image = event.frame[:, :, ::-1]

    fov = field_of_view(mask_image,event)
    
    m_i = copy.copy(mask_image)
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

    new_seg = np.zeros_like(m_i)

    for d in event.metadata['colors']:
        if d['color'] in unique_obs.tolist() and d['name']==f:

            #pos = np.argwhere(m_i==d['color'])
            pos = np.argwhere((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]))
            print("got pos ",pos)
            av_r,av_g,av_b  = 0,0,0
            for p in pos:
                rgb_pix = rgb_image[p[0],p[1],:]
                #print("rgb_pix ",rgb_pix)
                av_r +=rgb_pix[0]
                av_g +=rgb_pix[1]
                av_b +=rgb_pix[2]
            av_r/=len(pos)
            av_g/=len(pos)
            av_b/=len(pos)
            #print("Average colors ",(int(av_r),int(av_g),int(av_b)))
            
            def closest_colour(requested_colour):
                min_colours = {}
                for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                    r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                    rd = (r_c - requested_colour[0]) ** 2
                    gd = (g_c - requested_colour[1]) ** 2
                    bd = (b_c - requested_colour[2]) ** 2
                    min_colours[(rd + gd + bd)] = name
                return min_colours[min(min_colours.keys())]

            def get_colour_name(requested_colour):
                try:
                    closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
                except ValueError:
                    closest_name = closest_colour(requested_colour)
                    actual_name = None
                return actual_name, closest_name

            requested_colour = (int(av_r),int(av_g),int(av_b))
            actual_name, closest_name = get_colour_name(requested_colour)

            #print ("Actual colour name:", actual_name, ", closest colour name:", closest_name)
            sent = f+" has the color "+closest_name
    
    return [sent]

def process_text(sent):
    final = ""
    numbers = ['1','2','3','4','5','6','7','8','9','0']
    for s in sent:
        if s=='|' or s in numbers or s=='.' or s=='+' or s=='-':
            pass
        else:
            final= final+s
    return final

def localize(event,m_x,m_z):
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']
    a = int(math.fabs((x - m_x)/0.25))
    b = int(math.fabs((z - m_z)/0.25))
    return a,b
def segment(event,obj):
    #obj = input("Enter object you want to segment ")
    mask_image = event.instance_segmentation_frame
    rgb_image = event.frame[:, :, ::-1]

    fov = field_of_view(mask_image,event)

    m_i = copy.copy(mask_image)
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

    new_seg = np.zeros_like(m_i)

    for f in fov:
        if obj==f:
            #print("You mean ",f)
            for d in event.metadata['colors']:
                if d['color'] in unique_obs.tolist() and d['name']==f:
                    pos = np.argwhere((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]))

                    area = len(pos.flatten())/3
                    cent = np.mean(pos,axis=0)
                    #print("Got color centroid of object ",d['name']," as (x,y) ",cent[1],cent[0])
                    e = 255*np.ones((mask_image.shape[0],mask_image.shape[1]))
                    new_seg = np.where((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]), e, e*0.0)
                    #cv2.imshow("segmentation", new_seg) 
                    #cv2.waitKey(0)
                    return new_seg


def roam(event,env, traj_data, m_x, m_z):
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']

    a = int(math.fabs((x - m_x)/0.25))
    b = int(math.fabs((z - m_z)/0.25))
    rot_step = 36

    for _ in range(200):
        inp = input("enter command ")
        #print(type(x))
        #action["x"] = float(x)
        #event = env.step(action)
        if inp=='l':
            event = env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
        if inp=='r':
            event = env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
        if inp=='u':
            event = env.step(dict({"action": "MoveAhead", "moveMagnitude" : 0.25}))
        if inp=='d':
            event = env.step(dict({"action": "MoveBack", "moveMagnitude" : 0.25}))
        if inp=='rl':
            event = env.step(dict({"action": "RotateLeft"}))
        if inp=='rr':
            event = env.step(dict({"action": "RotateRight"}))
        if inp=='srl':
            rot = event.metadata['agent']['rotation']['y']
            custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": rot-rot_step,"x": x,"y": y,"z": z}
            event = env.step(dict(custom_rot))
        if inp=='srr':
            rot = event.metadata['agent']['rotation']['y']
            custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": rot+rot_step,"x": x,"y": y,"z": z}
            event = env.step(dict(custom_rot))
        if inp=='lu':
            event = env.step(dict({"action": "LookUp"}))
        if inp=='ld':
            event = env.step(dict({"action": "LookDown"}))
        if inp=='oo':
            event = env.step(dict({'action': 'OpenObject', 'objectId': 'Drawer|-01.29|+00.44|-00.58'}))  
        if inp=='init':
            traj_data['scene']["init_action"] = {
                "action": "TeleportFull",
                "horizon": 30,
                "rotateOnTeleport": True,
                "rotation": 0, #hopefully 0 means always facing agents relative north
                "x": x,
                "y": y,
                "z": z
            }
            event = env.step(dict(traj_data['scene']['init_action']))
        if inp=="segment":
            obj = input("Enter object you want to segment ")
            mask_image = event.instance_segmentation_frame
            rgb_image = event.frame[:, :, ::-1]

            fov = field_of_view(mask_image,event)
            
            m_i = copy.copy(mask_image)
            unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

            new_seg = np.zeros_like(m_i)

            for f in fov:
                if obj+'|' in f:
                    print("You mean ",f)

                    for d in event.metadata['colors']:
                        if d['color'] in unique_obs.tolist() and d['name']==f:

                            #pos = np.argwhere(m_i==d['color'])
                            pos = np.argwhere((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]))

                            area = len(pos.flatten())/3
                            cent = np.mean(pos,axis=0)
                            print("Got color centroid of object ",d['name']," as (x,y) ",cent[1],cent[0])
                            e = 255*np.ones((mask_image.shape[0],mask_image.shape[1]))
                            new_seg = np.where((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]), e, e*0.0)
                            cv2.imshow("segmentation", new_seg) 
                            cv2.waitKey(0)




        if inp=="describe":
            obj = input("Enter target object  ")
            mask_image = event.instance_segmentation_frame
            fov = field_of_view(mask_image,event)

            exact_object = ""
            for f in fov:
                if obj+'|' in f:
                    print("You mean ",f)
                    yn = input("Is this the object you are looking for ? (y/n)")
                    if yn=="y":
                        exact_object = f
                        break


            f = exact_object
            if f!="":

                sent = []

                sent.extend(desc1(event,f,fov))


                #Extract pairs of relative location informations for f with every other visible object (o) 
                oc, tc = relative_location(mask_image, event, f, area_thresh = 0)
                

                sent.extend(desc2(tc,f))

                sent.extend(desc3(oc,f))
                
                sent.extend(desc4(oc,f))
                
                sent.extend(desc5(event,oc,f))

                #sent.extend(desc6(event,f)) #desc6 gives improper color names sometimes.. need to check more

                for s in sent:
                    s = process_text(s)
                    print(s)

                #remaining examples:
                #1. this cup is the cup that is on the topmost shelf (if multiple receps -ie shelfs are present)
                #2. This cup is of the color white (color of random objects)
                #3. Put the pencil on the shelf with the golden statue (referring receptacles like shelfs with the objects in them)
                

        if inp=='datacollect': #test the automated data collection process
            mask_image = event.instance_segmentation_frame
            fov = field_of_view(mask_image,event)
            FOV = []
            for f in fov:
                if 'Floor' in f or 'Wall' in f or 'Door' in f:
                    pass
                else:
                    FOV.append(f)
            fov = FOV
            f = random.choice(fov)
            print("Randomly chose ",f, " as the target object")
            s = segment(event,f)
            rgb_image = event.frame[:, :, ::-1]
            store_id = '301_10_(10,10)_90' #room number, task number, grid position, agent rotation

            cv2.imwrite('data/seg/'+store_id+'.jpg', s)
            cv2.imwrite('data/rgb/'+store_id+'.jpg', rgb_image)

            #Extract automated descriptions of the object in the rgb image
            sent = []
            sent.extend(desc1(event,f,fov))
            #Extract pairs of relative location informations for f with every other visible object (o) 
            oc, tc = relative_location(mask_image, event, f, area_thresh = 0)
            sent.extend(desc2(tc,f))
            sent.extend(desc3(oc,f))
            sent.extend(desc4(oc,f))
            sent.extend(desc5(event,oc,f))
            descriptions = {}
            
            descriptions[store_id] = [] 
            c = 0
            for s in sent:
                s = process_text(s)
                print(s)
                descriptions[store_id].append(s)
                c+=1
            with open('data/descriptions.json', 'w') as fp:
                json.dump(descriptions, fp, indent = 4)





        if inp=='bye':
            return
        x = event.metadata['agent']['position']['x']
        y = event.metadata['agent']['position']['y']
        z = event.metadata['agent']['position']['z']

        a = int(math.fabs((x - m_x)/0.25))
        b = int(math.fabs((z - m_z)/0.25))
        print("got agents position as ",a,b)
        print("agents 3D position ",x,y,z)
        print("got agents rotation as ",event.metadata['agent']['rotation'])




def parent_receptacle(event,obj):
    for o in event.metadata['objects']:
        if o['objectId']==obj:
            if o['parentReceptacles']!=[] and o['parentReceptacles']!= None: #means the object is placed on something
                return o['parentReceptacles'][0] #the first object that is the receptacle in the parent receptacle list
    return []


def field_of_view(mask_image, event, verbose = False):
    m_i = copy.copy(mask_image)
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)
    
    fovs = []

    for c in event.metadata['colors']:
        if c['color'] in unique_obs.tolist():
            if verbose:
                print("color match -> ",c['color']," --> ",c['name'])
            fovs.append(c['name'])
    return fovs



def relative_location(mask_image, event, obj, area_thresh = 100, center = "self"): 
    #area_thresh = 100

    #print("Got object ",obj)
    #make sure using instance segmentation, then only d['name'] will provide unique objectid
    m_i = copy.copy(mask_image)
    unique_obs = np.unique(mask_image.reshape(-1,mask_image.shape[2]),axis=0)

    target_obj = ""
    visible = []

    if center =="self":
        x = event.metadata['agent']['position']['x']
        y = event.metadata['agent']['position']['y']
        z = event.metadata['agent']['position']['z']
    else:
        for o in event.metadata['objects']:
            if o['objectId']==center:
                x = o['position']['x']
                y = o['position']['y']
                z = o['position']['z']




    centroids = {}
    tar_centroid = []
    areas = {}


    for d in event.metadata['colors']:
        if d['color'] in unique_obs.tolist():
            visible.append(d['name'])

            #pos = np.argwhere(m_i==d['color'])
            pos = np.argwhere((m_i[:,:,0]==d['color'][0]) & (m_i[:,:,1]==d['color'][1]) & (m_i[:,:,2]==d['color'][2]))

            area = len(pos.flatten())/3
            areas[d['name']] = area #store the segmented area of each visible object

            cent = np.mean(pos,axis=0)
            #print("Got color centroid of object ",d['name']," as ",cent)
            centroids[d['name']] = cent


            #if obj+'|' in d['name']:
            if obj == d['name']:
                target_obj = d['name']
                tar_centroid = [cent[1],cent[0]]

            
    
    x1 = 0
    y1 = 0
    z1 = 0

    for o in event.metadata['objects']:
        if o['objectId']==target_obj:
            x1 = o['position']['x']
            y1 = o['position']['y']
            z1 = o['position']['z']
            #print("x1 ",x1)
            #print("y1 ",y1)
            #print("z1 ",z1)


    object_centers = {}
    for o in event.metadata['objects']:
        if o['objectId'] in visible and o['objectId']!=target_obj:
            if areas[o['objectId']]>area_thresh: #area is greater than 100 pixels 
                #object_centers[o['objectId']] = [,o['position']['y'],o['position']['z']] 
                dx = (o['position']['x']-x) - (x1-x)
                dy = (o['position']['y']-y1)
                #is the object farther from the agent than from the target object?
                #dz = (o['position']['z']-z) - (z1-z) 
                dz = ( (o['position']['z']-z)**2 + (o['position']['x']-x)**2 ) - ( (z-z1)**2 + (x-x1)**2 ) 
                if x1==0 and y1==0 and z1==0: #object doesnt have an associated center (maybe a texture)
                    object_centers[o['objectId']] = [0,0,0,centroids[o['objectId']][1]-centroids[target_obj][1], centroids[target_obj][0]-centroids[o['objectId']][0]]
                else:
                    object_centers[o['objectId']] = [dx,dy,dz,centroids[o['objectId']][1]-centroids[target_obj][1], centroids[target_obj][0]-centroids[o['objectId']][0]] 

    #print("Visible objects ",object_centers)
    #object_centers dict stores info --
    # position 1 is the true dx from event meta (not used)
    # position 2 is the true dy from event meta (not used)
    # position 3 is the difference in distances between agent-object and target-object (used for determining whether the object is farther out front or nearer as compared to a target object)
    # position 4 is the estimated dx from color segmentation (used to determine whether an object is to left or right of a target object)
    # position 5 is the estimated dy from color segmentation (used to determing whether an object is above or below a target object)

    return object_centers, tar_centroid

def get_file(rn = 302, task_index = 1, trial_num = 0):
    folders = sorted(glob.glob('/home/hom/alfred/data/json_2.1.0/train/*'+repr(rn))) #for home computer
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

def set_env(env,json_file, debug = True):

    with open(json_file) as f:
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

    env.step(dict(traj_data['scene']['init_action']))
    event = env.step(dict(action = 'GetReachablePositions'))
    reach_pos = event.metadata['actionReturn'] #stores all reachable positions for the current scene
    rand_init = random.choice(reach_pos)

    reach_x = [i['x'] for i in reach_pos]
    reach_z = [i['z'] for i in reach_pos]

    m_x = min(reach_x)
    m_z = min(reach_z)

    if debug:
        roam(event,env,traj_data,m_x,m_z)
    else:
        return  env, event, m_x, m_z, reach_pos




def random_gen(env, event, room, task, m_x, m_z):
    
    x = event.metadata['agent']['position']['x']
    y = event.metadata['agent']['position']['y']
    z = event.metadata['agent']['position']['z']
    rot = event.metadata['agent']['rotation']['y']
    a = int(math.fabs((x - m_x)/0.25))
    b = int(math.fabs((z - m_z)/0.25))
    rot_step = 72
    print("In grid location ",a,b)

    descriptions = {}

    for _ in range(5):
        
        mask_image = event.instance_segmentation_frame
        fov = field_of_view(mask_image,event)
        FOV = []
        for f in fov:
            if 'Floor' in f or 'Wall' in f or 'Door' in f or 'FP' in f or '|' not in f:
                pass
            else:
                FOV.append(f)
        fov = FOV

        #f = random.choice(fov)
        
        store_id = repr(room)+'_'+repr(task)+'_'+repr(a)+','+repr(b)+'_'+repr(rot) #room number, task number, grid position, agent rotation
        descriptions[repr(rot)] = {}
        for f in fov:
            descriptions[repr(rot)][f] = []

            #print("Chose ",f, " as the target object")
            s = segment(event,f)
            rgb_image = event.frame[:, :, ::-1]
            #store_id = '301_10_(10,10)_90' #room number, task number, grid position, agent rotation
            

            cv2.imwrite('data/seg/'+store_id+'_'+f+'.jpg', s)
            cv2.imwrite('data/rgb/'+store_id+'.jpg', rgb_image)

            #Extract automated descriptions of the object in the rgb image
            sent = []
            sent.extend(desc1(event,f,fov))
            #Extract pairs of relative location informations for f with every other visible object (o) 
            oc, tc = relative_location(mask_image, event, f, area_thresh = 0)
            sent.extend(desc2(tc,f))
            sent.extend(desc3(oc,f))
            sent.extend(desc4(oc,f))
            sent.extend(desc5(event,oc,f))

            for s in sent:
                s = process_text(s)
                #print(s)
                descriptions[repr(rot)][f].append(s)

        '''
        with open('data/descriptions.json', 'w') as fp:
            json.dump(descriptions, fp, indent = 4)
        '''

        #before end of the loop rotate left by 36 degrees
        rot = event.metadata['agent']['rotation']['y']
        custom_rot = {"action": "TeleportFull","horizon": 30,"rotateOnTeleport": True,"rotation": rot-rot_step,"x": x,"y": y,"z": z}
        event = env.step(dict(custom_rot))

    return descriptions


if __name__ == '__main__':
    print("############ Time now ##########")
    print(dt.now())
    debug = False


    IMAGE_WIDTH = 300 #rendering
    IMAGE_HEIGHT = 300
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,player_screen_height=IMAGE_HEIGHT) #blank ai2thor environment


    if debug:
        json_file = get_file(rn = 301, task_index = 10, trial_num = 0)
        set_env(env,json_file[0], debug = debug)

    else:
        descriptions = {}
        rooms = list(range(301,320))+list(range(201,220))#+list(range(401,420))

        for i in rooms: #room range
            descriptions[repr(i)] = {}
            for j in range(0,30): #task range
                try:
                    json_file = get_file(rn = i, task_index = j, trial_num = 0)
                    env, event, m_x, m_z, reach_pos = set_env(env,json_file[0], debug = debug)
                    a,b = localize(event,m_x,m_z)
                    #Generate descriptions for all objects visible in the current grid location at each 36 degree rotations
                    
                    descriptions[repr(i)][repr(j)] = {}
                    descriptions[repr(i)][repr(j)][repr(a)+'_'+repr(b)] = random_gen(env,event,i,j,m_x,m_z)

                    print("Now going to 10 random locations")
                    #Generate descriptions for 30 random locations too
                    positions = random.sample(reach_pos,10)
                    print("Got positions ",positions)
                    traj_data = {'scene':{}}

                    for rand_init in positions:
                        traj_data['scene']["init_action"] = {
                            "action": "TeleportFull",
                            "horizon": 30,
                            "rotateOnTeleport": True,
                            "rotation": 0, #hopefully 0 means always facing agents relative north
                            "x": rand_init['x'],
                            "y": rand_init['y'],
                            "z": rand_init['z']
                        }
                        print("position ","x ", rand_init['x'],
                                "y ", rand_init['y'],
                                "z ", rand_init['z'])
                        
                        event = env.step(dict(traj_data['scene']['init_action']))
                        a,b = localize(event,m_x,m_z)
                        descriptions[repr(i)][repr(j)][repr(a)+'_'+repr(b)] = random_gen(env,event,i,j,m_x,m_z)

                except:
                    print("This room and task is not present ")
                    traceback.print_exc()

        with open('data/descriptions.json', 'w') as fp:
            json.dump(descriptions, fp, indent = 4)




    
'''
Modify desc functions to return atmost 1 sentence and if no sentence can be returned , return the name of the target object

Write a description function for shelves -> eg- this is the shelf that contains pencil
                                            eg- This is the top right shelf

'''