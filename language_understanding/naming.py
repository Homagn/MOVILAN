import numpy as np
import copy
import math
from word2number import w2n
import traceback


interaction_objects = ["Pencil","Desk","Drawer","Dresser","Drawer","Watch","Cabinet","Pen","Bowl","Bed",
                        "Book","Boots","Bottle","CD","Desktop","DiningTable","FloorLamp","Footstool",
                        "Fork","GarbageCan","HousePlant","KeyChain","LaundryHamper","LightSwitch",
                        "Mirror","Ottoman","Pillow","Poster","Pot","Safe","Shelf","Sofa","Statue",
                        "Stool","Television","TennisRacket","TissueBox","Window","Vase","Towel","Cloth",
                        "Chair","ArmChair","CounterTop","CreditCard","Cup","BasketBall","BaseballBat","AlarmClock",
                        "Wall","Room", "Mug",'Laptop','Beanbag','table','Box', 'Cloth', "Plate", "RemoteControl",
                        "Sofa","CoffeeTable","Newspaper","Knife","Sink","Microwave","Apple","Stove","CoffeeMachine"]

toggleable = ["lamp","Lamp",'light','Light','tablelamp']

#bean bag should not be assigned to nothing, but the damned authors hardcoded the beanbag to be invisible (room 301), 
#like literaly o['visble'] is false for beanbag
#chest of drawers is most probably a dresser or a cabinet
'''
group_terms = {("cell","phone"):"cellphone",("box","tissues"):"TissueBox",
                ("trash","can"):"GarbageCan", ("bean","bag"):" ",
                ("chest","drawers"):"Dresser"}  
'''

group_terms = {("cell","phone"):"cellphone",("box","tissues"):"TissueBox",
                ("trash","can"):"GarbageCan",("bean","bag"):"Beanbag",
                ("chest","drawers"):"Dresser",("shelving","unit"):"Shelf", ("desk","shelf"):"Shelf",
                ("garbage","can"):"GarbageCan", ("counter","window"):"Window", ("dining","table"):"DiningTable"} 

def infer_action(obj):
    if obj in toggleable:
        return "on"
    else:
        return []

def fnl(obj_list):
    o_list = []
    def flatten_nested_list(obj_list):
        if isinstance(obj_list,list):
            for o in obj_list:
                a = flatten_nested_list(o)
                if a!=None:
                    o_list.append(a)
        else:
            o_list.append(obj_list)
    flatten_nested_list(obj_list)
    return o_list

def name_objects(obj_list,sentence): #make sense of user provided object names according to event meta, also remove senseless objects like 'wood'
    print("(naming.py-> name_objects)")
    if obj_list=='':
        return obj_list
    o_list = []


    def condense(obj_list,sentence): #use place if occurence in the sentence to group together fragmented words pointing to the same object
        l = copy.copy(obj_list)
        sentence = sentence.replace(',','')
        sentence = sentence.replace('.','')

        sentence = sentence.split(' ')
        #print("input obj_list ",obj_list)
        #print("sentence ",sentence)
        for o1 in obj_list:
            for o2 in obj_list:
                try:
                    if math.fabs(sentence.index(o2)-sentence.index(o1)<=2):
                        if (o1,o2) in group_terms.keys() or (o2,o1) in group_terms.keys():
                            try:
                                l[l.index(o1)]  = group_terms[(o1,o2)]
                            except:
                                l[l.index(o1)]  = group_terms[(o2,o1)]
                except:
                    #print("terms in the original sentence might have already been grouped together")
                    pass

        uniq = []
        for i in l:
            if i not in uniq:
                uniq.append(i)
        return uniq



    o_list = fnl(obj_list)
    o_list = condense(o_list,sentence)

    objs = ""
    #Manually "lemmatizing" words below, can easily use embeddings similarity from spacy (word_similarity.py)
    for o in o_list:

        if o=="tissues":
            objs = objs+"TissueBox"+','
        if o=="counter" or o=="island" or o=="islandcounter" or o=="kitchenisland":
            objs = objs +"CounterTop"+','
        if o=="coffeemaker" or o=="coffeemachine" or o=="coffee":
            objs = objs +"CoffeeMachine"
        if o=="stove" or o=="oven" or o=="range" or o=="Oven":
            objs = objs +"Stove"+','
        if o=="stool" or o=="footstool":
            objs = objs+"Ottoman"+','
        if o=="bottle" or o=="Bottle" or o=="container":
            objs = objs+"Vase"+','
        if o=="tray":
            objs = objs+"Plate"+','
        if o=="rag":
            objs = objs+"Cloth"+','
        if o=="computer" or o=="compute":
            objs = objs+"Laptop"+','
        if o=="bureau":
            objs = objs+"Bureau"+','
        if o=="clock":
            objs = objs+"AlarmClock"+','
        if o=="bookshelves" or o=="shelving" or o=="deskshelf" or o=="bookcase":
            objs = objs+"Shelf"+','
        if o=="disc" or o=="discs" or o=="cd":
            objs = objs+"CD"+','
        if o=="card":
            objs = objs+"CreditCard"+','
        if o=="counter":
            objs = objs+"CounterTop"+','
        if o=="keys":
            objs = objs+"KeyChain"+','
        if o=="bat":
            objs = objs+"BaseballBat"+','
        if o=="bean":
            objs = objs+"Beanbag"+','
        if o=="drawers":
            objs = objs+"Dresser"+','
        if o=="dresserdrawer": #happens in 306-8
            objs = objs+"Drawer"+','
        
        if o=="Lamp" or o=="lamp" or o=="light" or o=="tablelamp":
            objs = objs+"DeskLamp"+','
            #objs = objs+"FloorLamp"+','

        if o=="nightstand" or o=="Nightstand" or o=="side-table" or o=="sidetable" or o=="bedsidetable" or o=="nighttable":
            objs = objs+"SideTable"+','
        
        if o=="trashcan" or o=="trash" or o=="Garbage" or o=="garbage" or o=="garbagebin" or o=="trashbin" or o=="trashcanister" or o=="recycle" or o=="recycling" or o=="recyclingbin":
            objs = objs+"GarbageCan"+','

        if o=="cellphone" or o=="Cellphone" or o=="cell" or o=="Cell" or o=="phone" or o=="Phone":
            objs = objs+"CellPhone"+','
        if o=="plant" or o=="Plant":
            objs = objs+"HousePlant"+','
        if o=="basket" or o=="Basket":
            objs = objs+"LaundryHamper"+','
        if o=="racket" or o=="Racket" or o=="tennisracket" or o=="paddle":
            objs = objs+"TennisRacket"+','
        if o=="tv" or o=="Television" or o=="television" or o=="TV" or o=="tvstand":
            objs = objs+"Television"+','
        if o=="tvstand" or o=="TVstand" or o=="TVStand":
            objs = objs+"TVStand"+','
        if o=="dish":
            objs = objs+"Bowl"+','
        if o=="couch" or o=="Couch":
            objs = objs+"Sofa"+','

        if o=="door" or o=="Door":
            objs = objs+"StandardDoor"+','
        if o=="coffeetable":
            objs = objs+"CoffeeTable"+','


            

        

        else:
            for io in interaction_objects:
                if o.upper()==io.upper():
                    objs = objs+io+','
                    break
            #print("WARNING ! cannot find appropriate map in name_objects")
    if objs == "":
        #print("WARNING ! could not find any relations ")
        return ""

    #remove trailing commas
    if objs[-1]==',':
        objs = objs[:-1]
    #print("got named objects ",objs)
    return objs

def name_actions(act_list):
    print("(naming.py -> name_actions)")
    o_list = []
    #take care of nested lists
    #print("act_list ",act_list)
    '''
    if isinstance(act_list,list):
        for o in act_list:
            if isinstance(o,list):
                o_list.extend(o)
            else:
                o_list.append(o)
    else:
        o_list = [act_list]
    '''
    o_list = fnl(act_list)

    #print("o_list ",o_list)
    objs = ""
    for o in o_list:
        if o=="Open" or o=="open":
            objs = objs+"open"+','
        if o=="remove" or o=="Remove" or o=="Pick" or o=="pick" or o=="take" or o=="Take" or o=="Grab" or o=="grab" or o=="Take" or o=="take" or o=="Get" or o=="get":
            objs = objs+"pick"+','
        if o=="Place" or o=="place" or o=="put" or o=="drop" or o=="set":
            objs = objs+"place"+','
        if o=="close" or o=="Close" or o=="Shut" or o=="shut": 
            objs = objs+"close"+','
        if o=="on" or o=="On": 
            objs = objs+"turnon"+','
        if o=="Carry" or o=="carry" or o=="bring": 
            objs = objs+"carry"+','
        if o=="wash" or o=="Wash" or o=="clean": 
            objs = objs+"clean"+','
        if o=="Look" or o=="look": 
            objs = objs+"look"+','
        if o=="slice" or o=="cut": 
            objs = objs+"slice"+','
        if o=="cook" or o=="heat": 
            objs = objs+"cook"+','
        else:
            #print("WARNING ! cannot find appropriate map in name_actions")
            pass
    if objs == "":
        #print("WARNING ! could not find any relations ")
        return ""
    #remove trailing commas
    if objs[-1]==',':
        objs = objs[:-1]
    #print("named actions ",objs)
    return objs

def name_directions(dir_list):
    print("(naming.py -> named_directions)")
    o_list = []
    #take care of nested lists
    if isinstance(dir_list,list):
        for o in dir_list:
            if isinstance(o,list):
                o_list.extend(o)
            else:
                o_list.append(o)        
    else:
        o_list = [dir_list]

    objs = ""
    for o in o_list:
        if o=="Bottom" or o=="bottom":
            objs = objs+"bottom"+','
        if o=="between" or o=="between":
            objs = objs+"between"+','
        if o=="Bottom" or o=="bottom":
            objs = objs+"bottom"+','
        if o=="Top" or o=="top" or o=="above" or o=="up" or o=="Up":
            objs = objs+"up"+','
        if o=="Right" or o=="right": 
            objs = objs+"right"+','
        if o=="Left" or o=="left": 
            objs = objs+"left"+','
        if o=="Face" or o=="face": 
            objs = objs+"face"+','
        if o=="With" or o=="with": 
            objs = objs+"with"+','
        if o=="On" or o=="on": 
            objs = objs+"on"+','
        if o=="opposite" or o=="Opposite": 
            objs = objs+"opposite"+','
        if o=="At" or o=="at": 
            objs = objs+"at"+','
        if o=="Next" or o=="next": 
            objs = objs+"next"+','
        if o=="In" or o=="in" or o=="inside" or o=="Inside": 
            objs = objs+"in"+','
        else:
            #print("WARNING ! cannot find appropriate map in name_directions")
            pass
    if objs == "":
        #print("WARNING ! could not find any relations ")
        return ""
    #remove trailing commas
    if objs[-1]==',':
        objs = objs[:-1]
    #print("named directions ",objs)
    return objs

def intensity2digits(desc):
    print("(naming.py -> intensity2digits)")
    #print("got desc ",desc)
    l = []
    if desc=='':
        return [-1,-1,-1,-1]
    for d in desc:
        try:
            l.append(w2n.word_to_num(d))
        except:
            #print("WARNING ! valid number description not found in action intensity desc")
            print("Exception caught !")
            #traceback.print_exc()

            '''
            #probably just missing one step wound not be that important
            if d=='step':
                l.append(1)
            else:
                l.append(d)
            '''
            l.append(d)
    return l


def name_movements(move_list):
    print("(naming.py -> name_movements)")
    o_list = []
    '''
    #take care of nested lists
    if isinstance(move_list,list):
        for o in move_list:
            if isinstance(o,list):
                o_list.extend(o)
            else:
                o_list.append(o)        
    else:
        o_list = [move_list]
    '''
    o_list = fnl(move_list)

    objs = ""
    for o in o_list:
        if o=="turn" or o=="Turn" or o=="turning" or o=="Turning":
            objs = objs+"turn"+','
        if o=="go" or o=="Go":
            objs = objs+"go"+','
        if o=="forward" or o=="Forward" or o=='straight' or o=='Straight':
            objs = objs+"forward"+','
        if o=="Walk" or o=="walk" or o=="walking" or o=="Walking":
            objs = objs+"walk"+','
        if o=="Take" or o=="take":
            objs = objs+"take"+','
        if o=="through":
            objs = objs+"through"+','
        if o=="Move" or o=="move" or o=="veer":
            objs = objs+"move"+','

        if o=="Around" or o=="around":
            objs = objs+"around"+','
        if o=="Left" or o=="left":
            objs = objs+"left"+','
        if o=="Right" or o=="right":
            objs = objs+"right"+','
        if o=="upwards" or o=="up":
            objs = objs+"up"+','
        if o=="downwards" or o=="down":
            objs = objs+"down"+','
        else:
            #print("WARNING ! cannot find appropriate map in name_movements")
            pass
    if objs == "":
        #print("WARNING ! could not find any relations ")
        return ""
    #remove trailing commas
    if objs[-1]==',':
        objs = objs[:-1]
    #print("named movements ",objs)
    return objs