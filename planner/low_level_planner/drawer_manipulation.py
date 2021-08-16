import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

import planner.low_level_planner.object_type as object_type
import planner.low_level_planner.reverse_actions as reverse_actions
import planner.low_level_planner.object_localization as object_localization


openables = object_type.openables
find_reverse = reverse_actions.find_reverse
location_in_fov = object_localization.location_in_fov

def drawer_manipulation_remove(manip_action,targ_obj,relative,ref_objs,env): 
    #pseudocode
    '''
    1. Extract the different language cues, disambiguate what to open (drawer) and what to remove (eg pen) and set the simulator actions
        for picking up objects and open/close drawer
    2. When multiple drawers might be present, the user might have said open the top-left drawer or the bottom drawer
        decide a preferential order of search based on that
    3. Associate the exact drawer object iD with each element in preferential order in facings dictionary
    4. Repeatedly try to open each of the drawer in the preferential order and remove the (said) object (if found) 
        and also close back the drawer (if object successfully picked up)
    '''
    print("(manipulation_signatures.py -> drawer_manipulation_remove)")

    #1
    targ_obj = targ_obj.split(',') #split entry into a list by commas
    manip_action = manip_action.split(',')
    relative = relative.split(',')
    ref_objs = ref_objs.split(',')
    things2pick, things2open = openables(targ_obj+ref_objs) 

    act1 = manip_action[0]
    if act1=='open':
        act1 = "OpenObject"
    if act1=='close':
        act1 = "CloseObject"

    act1r = find_reverse(act1)
    act2 = manip_action[1]
    if act2=="pick":
        act2 = 'PickupObject'

    act3 = ""
    try:
        act3 = manip_action[2]
        if act3=='close':
            act3 = "CloseObject"
    except:
        pass

    
    #2
    env.step(dict({"action": "LookUp"}))
    facings = {}
    pitch = ['up','mid','bottom']
    yaw = ['left','mid','right']
    for p in pitch:
        for y in yaw:
            facings[p+'_'+y] = []
    look_cost = [1.5,1.5]
    for r in relative:
        if r =='up':
            look_cost[0] = 0
        if r =='mid':
            look_cost[0] = 1
        if r =='bottom':
            look_cost[0] = 2
        
        if r=='left':
            look_cost[1] = 0
        if r=='mid':
            look_cost[1] = 1
        if r=='right':
            look_cost[1] = 2
    dists = []
    dist_keys = []

    for fk in facings.keys():
        a = pitch.index(fk.split('_')[0])
        b = yaw.index(fk.split('_')[1])
        d = (a-look_cost[0])**2+(b-look_cost[1])**2
        dists.append(d)
        dist_keys.append(fk)

    sorted_dists = (np.argsort(np.array(dists))).tolist()
    order = [dist_keys[i] for i in sorted_dists]
    print("Preferrential order of directions while searching ",order)



    #3
    for i in range(3): #perform a basic 3 grid search for visible objects around the agent
        env.step(dict({"action": "LookDown"}))
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,_,_ = location_in_fov(env, mask_image,depth_image)
        
        for k in lf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[0]].append(k)

        for k in mf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[1]].append(k)

        for k in rf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[2]].append(k)
    env.step(dict({"action": "LookUp"}))
    #print("facings ",facings)

    #4
    print("Performing actions to view the target object ")
    for o in order:
        objects = facings[o]
 
        for objs in objects:
            print("in obj ",objs)
            env.step(dict({'action': act1, 'objectId': objs}))
            mask_image = env.get_segmented_image()
            depth_image = env.get_depth_image()
            lf,mf,rf,_,_ = location_in_fov(env, mask_image,depth_image)
            for k in lf.keys():
                if things2pick[0]+'|' in k:
                    print("found ",k)

                    env.step(dict({'action': act2, 'objectId': k}))
                    if act3!="":
                        env.step(dict({'action': act3, 'objectId': objs}))
                    return env

            for k in mf.keys():
                if things2pick[0]+'|' in k:
                    print("found ",k)

                    env.step(dict({'action': act2, 'objectId': k}))
                    if act3!="":
                        env.step(dict({'action': act3, 'objectId': objs}))
                    return env

            for k in rf.keys():
                if things2pick[0]+'|' in k:
                    print("found ",k)

                    env.step(dict({'action': act2, 'objectId': k}))
                    if act3!="":
                        env.step(dict({'action': act3, 'objectId': objs}))
                    return env

                env.step(dict({'action': act1r, 'objectId': objs}))
    return env


def drawer_manipulation_place(manip_action,targ_obj,relative,ref_objs,env, numtries_r=0, numtries_s = 0): 
    #pseudocode
    '''
    1. Kepp track of original variables to backtrack later using recursion
    2. Extract cues from language, disambiguate which to place and what to open, fill incomplete action sequences such as 
        place and close generally has an open before to open the drawer
        finally decide the different actions from the simulator to open,close, pick and place objects 
    3. Determine the prefered search order for opening drawers if multiple are present based on language cues (look cost)
    4. Associate each of the element in prefered search order with the exact drawer object ids (facings dictionary)
    5. If no drawers were visible, then facings will be empty dictionary, so go back recursively to step 1, 
        keep rotating the agent clockwise so that drawers might become visible
    6.
        6-1. Repeated open each drawer according to preferential order in facings dictionary and try to place the object in hand
        6-2. Place operation may not succeed due to positioning of agent, although drawer is close enough and visible
                In that case, follow a knight's trajectory (as in chess) with repeated backtracking to step 1 and try to place the object
        6-3. Finally if place was successful report it and close the drawer
    '''

    print("(manipulation_signatures.py -> drawer_manipulation_place)")
    print("Trying this for ",numtries_r,numtries_s," time")

    #1
    o_manip_action = manip_action
    o_targ_obj = targ_obj
    o_relative = relative
    o_ref_objs = ref_objs


    #2
    targ_obj = targ_obj.split(',') #split entry into a list by commas
    manip_action = manip_action.split(',')
    relative = relative.split(',')
    ref_objs = ref_objs.split(',')

    things2pick, things2open = openables(targ_obj+ref_objs) 

    try:
        if manip_action[0]=='place' and manip_action[1]=='close':
            #cannot possibly place it inside without opening unless already opened, 
            #which does not give problems with trying to open again
            manip_action = ['open','place','close'] 
    except:
        pass
    try: 
        if manip_action[0]=='place' and len(manip_action)==1: #only place action is specified
            #cannot possibly place it inside without opening unless already opened, 
            #which does not give problems with trying to open again
            manip_action = ['open','place'] 
    except:
        pass
    act1 = manip_action[0]
    if act1=='open':
        act1 = "OpenObject"
    if act1=='close':
        act1 = "CloseObject"

    act1r = find_reverse(act1)
    act2 = manip_action[1]
    if act2=="pick":
        act2 = 'PickupObject'
    if act2=="place":
        act2 = 'PutObject'

    act3 = ""
    try:
        act3 = manip_action[2]
        if act3=='close':
            act3 = "CloseObject"
    except:
        pass

    
    #3
    env.step(dict({"action": "LookUp"}))

    facings = {}
    pitch = ['up','mid','bottom']
    yaw = ['left','mid','right']
    for p in pitch:
        for y in yaw:
            facings[p+'_'+y] = []

    look_cost = [1.5,1.5]

    for r in relative:
        if r =='up':
            look_cost[0] = 0
        if r =='mid':
            look_cost[0] = 1
        if r =='bottom':
            look_cost[0] = 2
        
        if r=='left':
            look_cost[1] = 0
        if r=='mid':
            look_cost[1] = 1
        if r=='right':
            look_cost[1] = 2

    dists = []
    dist_keys = []

    for fk in facings.keys():
        a = pitch.index(fk.split('_')[0])
        b = yaw.index(fk.split('_')[1])
        d = (a-look_cost[0])**2+(b-look_cost[1])**2
        dists.append(d)
        dist_keys.append(fk)

    sorted_dists = (np.argsort(np.array(dists))).tolist()
    order = [dist_keys[i] for i in sorted_dists]
    print("Preferred order of search ",order)

    

    #4
    for i in range(3): #perform a basic 3 grid search for visible objects around the agent
        env.step(dict({"action": "LookDown"}))
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,_,_ = location_in_fov(env,mask_image,depth_image)

        for k in lf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[0]].append(k)

        for k in mf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[1]].append(k)

        for k in rf.keys():
            if things2open[0]+'|' in k: #Assuming atmost 1 thing is told to open in 1 instruction
                #print("found ",k1," in ",k)
                facings[pitch[i]+'_'+yaw[2]].append(k)
    env.step(dict({"action": "LookUp"}))
    env.step(dict({"action": "LookUp"}))
    print("Facings ",facings)

    #5
    if all([facings[x]==[] for x in facings.keys()]):
        print("Object is not visible, trying to rotate and find it ")
        if numtries_r<4:
            env.step(dict({"action": "RotateLeft",'forceAction': True}))
            return drawer_manipulation_place(o_manip_action,o_targ_obj,o_relative,o_ref_objs,env,numtries_r=numtries_r+1,numtries_s = numtries_s)
        else:
            return env


    #6
    print("Performing actions to view the target object ")
    for o in order:
        objects = facings[o]
        #6-1
        for objs in objects:
            print("Opening ",objs)
            env.step(dict({'action': act1, 'objectId': objs}))
            mask_image = env.get_segmented_image()
            depth_image = env.get_depth_image()

            lf,mf,rf,_,_ = location_in_fov(env,mask_image,depth_image)
            toplace = ""
            for k in lf.keys():
                if things2pick[0]+'|' in k:
                    #print("found ",k, " in lf")
                    toplace = k
            for k in mf.keys():
                if things2pick[0]+'|' in k:
                    #print("found ",k, " in mf")
                    toplace = k
            for k in rf.keys():
                if things2pick[0]+'|' in k:
                    #print("found ",k, " in rf")
                    toplace = k

            #event = env.step(dict({'action': act2, 'objectId': k}))
            print("Trying to place ",toplace," in ",objs, "with action ",act2)
            env.step(dict({"action": "LookDown"}))
            env.step(dict({'action': act2, 'objectId': toplace, 'receptacleObjectId': objs, 'placeStationary': True, 'forceAction': True}))

            #6-2

            #(4 nested ifs)
            #1. Go forward, rotate right and try to place the object, if that fails
            #2. Go back to initial pos and rotation, now try go forward and rotate left to place the object
            #3. Try again with backtrack and rotate right and move back this time
            #4. Try again with backtrack and rotate left and move back this time
            #This place operation is very glitchy in the simulator
            nx,ny = 0,0
            if not env.actuator_success(): #moveahead, rotate right and try to place
                
                env.step(dict({"action": "MoveAhead",'forceAction': True}))
                if env.actuator_success():
                    ny = 1
                env.step(dict({"action": "RotateRight",'forceAction': True}))

                env.step(dict({'action': act1, 'objectId': objs}))
                env.step(dict({'action': act2, 'objectId': toplace, 'receptacleObjectId': objs, 'placeStationary': True, 'forceAction': True}))

                if not env.actuator_success(): # backtrack, moveahead, rotate left and try to place

                    env.step(dict({"action": "RotateLeft",'forceAction': True}))
                    if ny==1:
                        env.step(dict({"action": "MoveBack",'forceAction': True}))
                        ny = 0
                    
                    env.step(dict({"action": "MoveAhead",'forceAction': True}))
                    if env.actuator_success():
                        ny = 1
                    env.step(dict({"action": "RotateLeft",'forceAction': True}))
                    env.step(dict({'action': act1, 'objectId': objs}))
                    env.step(dict({'action': act2, 'objectId': toplace, 'receptacleObjectId': objs, 'placeStationary': True, 'forceAction': True}))

                    if not env.actuator_success(): # backtrack, moveright, moveback and try to place
                        env.step(dict({"action": "RotateRight",'forceAction': True}))
                        if ny==1:
                            env.step(dict({"action": "MoveBack",'forceAction': True}))
                            ny = 0

                        env.step(dict({"action": "MoveRight",'forceAction': True}))
                        env.step(dict({"action": "MoveBack",'forceAction': True}))
                        if env.actuator_success():
                            ny = -1

                        env.step(dict({'action': act1, 'objectId': objs}))
                        env.step(dict({'action': act2, 'objectId': toplace, 'receptacleObjectId': objs, 'placeStationary': True, 'forceAction': True}))

                        if not env.actuator_success(): # backtrack, moveleft, moveback and try to place
                            env.step(dict({"action": "MoveLeft",'forceAction': True}))
                            if ny==-1:
                                env.step(dict({"action": "MoveAhead",'forceAction': True}))
                                ny = 0

                            env.step(dict({"action": "MoveLeft",'forceAction': True}))
                            env.step(dict({"action": "MoveBack",'forceAction': True}))
                            if env.actuator_success():
                                ny = -1
                            env.step(dict({'action': act1, 'objectId': objs}))
                            env.step(dict({'action': act2, 'objectId': toplace, 'receptacleObjectId': objs, 'placeStationary': True, 'forceAction': True}))
            
            
            #6-3
            if env.actuator_success():
                print("Place success !")
                if act3!="":
                    print("Closing the ",objs)
                    env.step(dict({'action': act3, 'objectId': objs}))
                    env.step(dict({"action": "LookUp"}))
                    return env
                else:
                    env.step(dict({"action": "LookUp"}))
                    return env
            
            env.step(dict({"action": "LookUp"}))
            print("Closing ",objs)
            env.step(dict({'action': act1r, 'objectId': objs}))


    #7
    if check_place(env)==False:
        print("Trying to reposition agent ")
        if numtries_s<3:
            if numtries_s==0:
                env.step(dict({"action": "MoveLeft",'forceAction': True}))
                if env.actuator_success()==False: #action was unsuccessful
                    numtries_s = numtries_s+1
            if numtries_s==1:
                env.step(dict({"action": "MoveRight",'forceAction': True}))
                env.step(dict({"action": "MoveRight",'forceAction': True}))
            if numtries_s==2:
                env.step(dict({"action": "MoveRight",'forceAction': True}))
            return drawer_manipulation_place(o_manip_action,o_targ_obj,o_relative,o_ref_objs,env,numtries_r = numtries_r, numtries_s=numtries_s+1)
        else:
            return env


    return env