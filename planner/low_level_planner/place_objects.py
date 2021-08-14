import numpy as np
import sys
import os
import copy

os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

from language_understanding import equivalent_concepts as eqc
import planner.low_level_planner.object_localization as object_localization
import planner.low_level_planner.object_type as object_type
import planner.low_level_planner.move_camera as move_camera
import planner.low_level_planner.resolve as resolve
import planner.low_level_planner.refinement as refinement

location_in_fov = object_localization.location_in_fov
object_in_fov = object_localization.object_in_fov
openables = object_type.openables
set_default_tilt = move_camera.set_default_tilt
field_of_view = resolve.field_of_view
swivel_search = refinement.swivel_search

CONFUSIONS = eqc.CONFUSIONS_M
CONFLICT = eqc.CONFLICT
CONTEXTUALS = eqc.CONTEXTUALS

bigNsmallobs = object_type.bigNsmallobs
possible_receptacle = object_type.possible_receptacle
common_connected_component = object_localization.common_connected_component


def check_place(env):
    if env.check_inventory()!=[]:
        print("Place was unsuccessful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        return False
    elif env.check_inventory()==[]:
        print("Place was successful !")
        return True

def resolve_place(manip_action,targ_obj,ref_rel,ref_obj):
    print("(manipulation_signatures.py -> resolve_place)")
    targ_obj = targ_obj.split(',') #split entry into a list by commas
    manip_action = manip_action.split(',')
    relative = ref_rel.split(',')
    ref_objs = ref_obj.split(',')

    things2pick, things2open = openables(targ_obj+ref_objs) 
    if things2open==[]:
        return False
    print("Relative ",relative)
    #print("Ref_obj ",ref_obj)
    print("Found things2open ",things2open)

    #preposition = relative[ref_objs.index(things2open[0])]
    #if preposition=="in":
    if 'in' in relative:
        print("Must open ",things2open[0], " before putting ",targ_obj[0]," inside")
        return True
    else:
        return False


def refined_place(manip_action,targ_obj,refinement_rel,refinement_obj,env,tries=0):
    print("(manipulation_signatures.py -> refined_place)")
    print("Trying this for ",tries," time")

    o_manip_action = manip_action
    o_targ_obj = targ_obj
    o_refinement_rel = refinement_rel
    o_refinement_obj = refinement_obj

    targ_obj = targ_obj.split(',')[0] #split entry into a list by commas
    manip_action = manip_action.split(',')
    relative = refinement_rel.split(',')
    ref_obj = refinement_obj.split(',')

    

    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    lf,mf,rf,areas,_ = location_in_fov(env, mask_image,depth_image)
    receptacle = ""
    targets = []
    #finding the object you want to place (even if you are holding it in your hand !)
    if env.check_inventory()!=[]:
        for inv in range(len(env.check_inventory())):
            print("Current object in hand ",env.check_inventory()[inv]['objectId'])
            #targets.append(event.metadata['inventoryObjects'][inv]['objectId'])
            targets.append(env.check_inventory()[inv]['objectId'])



    small,big = bigNsmallobs(ref_obj)

    all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
    


    act1 = manip_action[0]
    if act1=="pick":
        act1 = 'PickupObject'
    if act1=="place":
        act1 = 'PutObject'

    if small!=[]:
        n_small = copy.copy(small)
        #sometimes user may confuse Mug as a cup and order to pick up the cup which is not visible in scene
        for sm in small:
            if sm in CONFUSIONS.keys():
                print("Possible confused object, all visibles are ", all_vis)
                csma = [CONFUSIONS[sm]+'|' in a for a in all_vis]
                sma = [sm+'|' not in a for a in all_vis]
                if any(csma) and all(sma):
                    print(CONFUSIONS[sm]," is visible but ",sm," is not visible so replacing as a related object")
                    n_small[n_small.index(sm)] = CONFUSIONS[sm]
                    ref_obj[ref_obj.index(sm)] = CONFUSIONS[sm]

        small = n_small
        #"facing": (centering around the small object first) (eg-DeskLamp)
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
        for k in lf.keys():
            k1 = small[0]
            if k1+'|' in k:
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
                break
        for k in rf.keys():
            k1 = small[0]
            if k1+'|' in k:
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
                break
    
        print("Small object ",small)
        print("Big object ",big)
        print("Refinement object ",ref_obj)

        #aaplying the relative movement (move left if it says place watch to the left of DeskLamp, cause you are already standing centered to the DeskLamp by now)
        
        ref_rel = ""
        try:
            ref_rel = relative[ref_obj.index(small[0])]
        except:
            ref_rel = relative[-1]

        d = ref_rel
        if d=="left":
            env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
        if d=="right":
            env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
        if d=="facing" or d=="with":
            mask_image = env.get_segmented_image()
            depth_image = env.get_segmented_image()
            lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
            for k in lf.keys():
                k1 = small[0]
                if k1+'|' in k:
                    print("Exact object iD for ",k1," is ",k)
                    env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
                    break
            for k in rf.keys():
                k1 = small[0]
                if k1+'|' in k:
                    print("Exact object iD for ",k1," is ",k)
                    env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
                    break
        if d=="in" or d=="on": #(task 305-17, place the pencil on the bowl)

            if targ_obj.capitalize() in CONFLICT.keys():
                s = small[0]
                #s = s[:s.index("|")].capitalize()
                s = s.capitalize()
                if s in CONFLICT[targ_obj]:
                    pass
                else:
                    if big!=[]:
                        big[0] = copy.copy(small[0]) #for cases like trying to place a pen in a bowl on top of dresser
                    else:
                        big = [copy.copy(small[0])]
            else:
                if big!=[]:
                    big[0] = copy.copy(small[0]) #for cases like trying to place a pen in a bowl on top of dresser
                else:
                    big = [copy.copy(small[0])]
                #big[0] = copy.copy(small[0]) #for cases like trying to place a pen in a bowl on top of dresser







    
    
    receptacle = ""
    receptacle_addendum = ""
    print("Big object (receps) ",big)



    #sometimes user may use general terms such as place the book on the table whereas it should be Desk or SideTable
    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()

    lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
    all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
    #big = copy.copy(things2pick)
    n_big = copy.copy(big)
    n_big_replace = [] #here all the possible replaceable objects are added

    for b in big:
        if b in CONFUSIONS.keys():
            print("Possible confused object, all visibles are ", all_vis)
            for bb in CONFUSIONS[b]:
                
                csma = [bb+'|' in a for a in all_vis]
                sma = [b+'|' not in a for a in all_vis]
                if any(csma) and all(sma):
                    print(bb," is visible but ",b," is not visible so replacing as a possibility")
                    n_big_replace.append(bb)
            if n_big_replace!=[]:
                n_big[n_big.index(b)] = n_big_replace[0]
    if big!=n_big:
        print("Changed big object (receps) ",big, " to ",n_big)
    big = n_big

    if big[0] in list(CONTEXTUALS["place"].keys()):
        receptacle_addendum = '|'+CONTEXTUALS["place"][big[0]]
        #if its Sink|+00.00|+00.89|-01.44, you need to place the object in Sink|+00.00|+00.89|-01.44|SinkBasin (room 2,task1,trial0)
        print("Contextual place operation encountered, adding extra context ",receptacle_addendum)



    env, receptacle, all_visibles = object_in_fov(env,big)
    
    '''
    all_visibles = []
    #finding the receptacle object
    env.step(dict({"action": "LookUp"}))
    cnt = 0
    for i in range(3): # was 3 earlier 
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)

        #field = list(lf.keys())+list(mf.keys())+list(rf.keys())
        field = field_of_view(env)
        for k in field:
            all_visibles.append(k)
            try:
                k1 = big[0]
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
    '''
    




    if receptacle=="":
        print("Receptacle not found, trying to swivel and search ")
        swivel_search(env,big[0])
        env, receptacle, all_visibles = object_in_fov(env,big)

    #nudge left right and try to find the receptacle
    if receptacle=="":
        print("Receptacle still not found, nudging left right ")
        #env, event = unit_refinement(env,event, big[0])




    if receptacle=="":
        print("WARNING ! receptacle may not match with language ")
        try:
            receptacle = possible_receptacle(all_visibles,big[0], bias = ref_obj)
        except:
            receptacle = "NOTHING"
        
        if receptacle=="NOTHING": #this time try without a bias from the language
            try:
                mask_image = env.get_segmented_image()
                #remove the target object from the list of refinement objects
                r_ref_obj = []
                for r in ref_obj:
                    if r in targ_obj:
                        pass
                    else:
                        r_ref_obj.append(r)
                #receptacle = common_connected_component(mask_image,event,r_ref_obj)
                receptacle = common_connected_component(mask_image,env,r_ref_obj)
            except:
                receptacle = "NOTHING"
    #applying place operation for detected target objects 
    #for tries in range(6): #3 tries to place the object
    t = targets[0]
    if receptacle_addendum not in receptacle:
        receptacle = receptacle+receptacle_addendum #based on additional infered context
    print("Trying to place the object ",t, " on the receptacle ",receptacle," (refined_place)")

    #print("objects in agents inventory ", event.metadata['inventoryObjects'])
    
    env.step(dict({"action": "LookDown"}))
    env.step(dict({'action': 'PutObject', 'objectId': t, 'receptacleObjectId': receptacle, 'placeStationary': True, 'forceAction': True}))

    #print("Environment actuator success ",env.actuator_success())
    if not env.actuator_success():
        print("Place was not successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        print("Trying again by looking down ") #for low lying receptacles like couch
        env.step(dict({"action": "LookDown"}))
        env.step(dict({'action': 'PutObject', 'objectId': t, 'receptacleObjectId': receptacle, 'placeStationary': True, 'forceAction': True}))

    
    if env.check_inventory()!=[]:
        print("Place was not successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        
        if tries==0 or tries==3:
            print("Tilting head all the way up and scanning top to bottom")
            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))
        #event = env.step(dict({'action': 'PutObject', 'objectId': t, 'receptacleObjectId': k}))
        if tries==1:
            env.step(dict({"action": "RotateLeft", 'forceAction': True}))
        if tries==2:
            env.step(dict({"action": "RotateRight", 'forceAction': True}))
            env.step(dict({"action": "RotateRight", 'forceAction': True}))
        if tries==3:
            env.step(dict({"action": "RotateLeft", 'forceAction': True}))
            print("Probably the object is too far, trying to move the agent closer ")
            env.step(dict({"action": "MoveAhead", "moveMagnitude" : 0.25}))

        if tries>=6:
            #return env,event
            return env

        return  refined_place(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,tries=tries+1)


    return env
        


#try to modify it
def refined_place2(manip_action,targ_obj,refinement_rel,refinement_obj,env,tries = 0):
    print("(manipulation_signatures.py -> refined_place2)")
    print("Trying this for ",tries," time")
    o_manip_action = manip_action
    o_targ_obj = targ_obj
    o_refinement_rel = refinement_rel
    o_refinement_obj = refinement_obj

    targ_obj = targ_obj.split(',')[0] #split entry into a list by commas
    manip_action = manip_action.split(',')
    relative = refinement_rel.split(',')
    ref_obj = refinement_obj.split(',')

    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()

    lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
    receptacle = ""
    ref_rel = ""
    targets = []
    #finding the object you want to place (even if you are holding it in your hand !)
    for k in lf.keys():
        k1 = targ_obj
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            targets.append(k)
            break
    for k in mf.keys():
        k1 = targ_obj
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            targets.append(k)
            break
    for k in rf.keys():
        k1 = targ_obj
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            targets.append(k)
            break

    small,big = bigNsmallobs(ref_obj)

    act1 = manip_action[0]
    if act1=="pick":
        act1 = 'PickupObject'
    if act1=="place":
        act1 = 'PutObject'


    #pop one of the smaller objects which is at the rightmost
    if "between" in relative or "with" in relative:
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        _,_,_,_,cent = location_in_fov(env,mask_image,depth_image)
        
        minkey = ""
        minv = 100000
        for k in cent.keys():
            for s in small:
                if s+'|' in k:
                    dist = cent[k][0]**2+cent[k][1]**2
                    if dist<minv:
                        minv = dist
                        minkey = k

        ref_rel = "right"
        small = [minkey]
        print("Modified small objects ",small)
    if "in" in relative:
        ref_rel = "in"
    if "on" in relative:
        ref_rel = "on"


    if small!=[]:
        #"facing": (centering around the small object first) (eg-DeskLamp)
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
        for k in lf.keys():
            print("In small!=[], key ",k)
            k1 = small[0]
            if k1==k:
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
                break
        for k in rf.keys():
            k1 = small[0]
            if k1==k:
                print("In small!=[], key ",k)
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
                break
    
    print("Small object ",small)
    print("Big object ",big)




    #sometimes user may use general terms such as place the book on the table whereas it should be Desk or SideTable
    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
    all_vis = list(lf.keys()) + list(mf.keys()) + list(rf.keys())
    #big = copy.copy(things2pick)
    n_big = copy.copy(big)
    n_big_replace = [] #here all the possible replaceable objects are added

    for b in big:
        if b in CONFUSIONS.keys():
            print("Possible confused object, all visibles are ", all_vis)
            for bb in CONFUSIONS[b]:
                
                csma = [bb+'|' in a for a in all_vis]
                sma = [b+'|' not in a for a in all_vis]
                if any(csma) and all(sma):
                    print(bb," is visible but ",b," is not visible so replacing as a possibility")
                    n_big_replace.append(bb)
            if n_big_replace!=[]:
                n_big[n_big.index(b)] = n_big_replace[0]
    if big!=n_big:
        print("Changed big object (receps) ",big, " to ",n_big)
    big = n_big




    #aaplying the relative movement (move left if it says place watch to the left of DeskLamp, cause you are already standing centered to the DeskLamp by now)
    
    d = ref_rel
    if d=="left":
        env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
    if d=="right":
        env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
    if d=="facing":
        mask_image = env.get_segmented_image()
        depth_image = env.get_depth_image()
        lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
        for k in lf.keys():
            k1 = small[0]
            if k1==k:
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
                break
        for k in rf.keys():
            k1 = small[0]
            if k1==k:
                print("Exact object iD for ",k1," is ",k)
                env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
                break
    if d=="in" or d=="on": #(305-17, place the pencil on the bowl)
        if targ_obj.capitalize() in CONFLICT.keys():
            s = small[0]
            s = s[:s.index("|")].capitalize()
            if s in CONFLICT[targ_obj]:
                pass
            else:
                big[0] = copy.copy(small[0])
        else:
            big[0] = copy.copy(small[0]) #for cases like trying to place a pen in a bowl on top of dresser
    
    #finding the receptacle object
    mask_image = env.get_segmented_image()
    depth_image = env.get_depth_image()
    lf,mf,rf,areas,_ = location_in_fov(env,mask_image,depth_image)
    receptacle = ""

    env, receptacle, _ = object_in_fov(env,big)

    '''
    #print("lf keys ",lf.keys())
    #print("mf keys ",mf.keys())
    #print("rf keys ",rf.keys())
    print("Searching exact id for receptacle ",big[0])
    for k in lf.keys():
        k1 = big[0]
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            receptacle = k
            break
    for k in mf.keys():
        k1 = big[0]
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            receptacle = k
            break
    for k in rf.keys():
        k1 = big[0]
        if k1+'|' in k:
            print("Exact object iD for ",k1," is ",k)
            receptacle = k
            break
    '''




    if receptacle=="":
        print("Receptacle not found, trying to swivel and search ")
        swivel_search(env,big[0])
        env, receptacle, _ = object_in_fov(env,big)
    
    if receptacle=="":
        print("Receptacle still not found, looking for connected components ")
        mask_image = env.get_segmented_image()
        try:
            #first remove the target object from refinement objects if present
            r_ref_obj = []
            for r in ref_obj:
                if r in targ_obj:
                    pass
                else:
                    r_ref_obj.append(r)
            #receptacle = common_connected_component(mask_image,event,r_ref_obj)
            receptacle = common_connected_component(mask_image,env,r_ref_obj)
        except:
            receptacle = "NOTHING"

    t = targets[0]

    print("Trying to place the object ",t, " on the receptacle ",receptacle, "(refined_place2)")
    #print("objects in agents inventory ", event.metadata['inventoryObjects'])
    
    env.step(dict({"action": "LookDown"}))
    env.step(dict({'action': 'PutObject', 'objectId': t, 'receptacleObjectId': receptacle, 'placeStationary': True, 'forceAction': True}))
    
    if env.check_inventory()!=[] and tries<6: #something is still in hand
        #print("Place was not successful ! ,object in hand ",event.metadata['inventoryObjects'][0]['objectId'])
        print("Place was not successful ! ,object in hand ",env.check_inventory()[0]['objectId'])
        
        if tries==0 or tries==3:
            print("Tilting head all the way up and scanning top to bottom")
            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))
            env.step(dict({"action": "LookUp"}))
        #event = env.step(dict({'action': 'PutObject', 'objectId': t, 'receptacleObjectId': k}))
        if tries==3:
            print("Probably the object is too far, trying to move the agent closer ")
            env.step(dict({"action": "MoveAhead", "moveMagnitude" : 0.25}))
        return  refined_place2(o_manip_action,o_targ_obj,o_refinement_rel,o_refinement_obj,env,tries=tries+1)
        
        #if tries>=6:
            #return env,event

    return env
