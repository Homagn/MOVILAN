import numpy as np
from skimage.measure import regionprops, label
from language_understanding import equivalent_concepts as eqc
import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

OPENS = eqc.OPENS
INVEN_OBJS = eqc.INVEN_OBJS
TOGGLES = eqc.TOGGLES
CONFUSIONS = eqc.CONFUSIONS
RECEPS = eqc.RECEPS
SLICES = eqc.SLICES

def bigNsmallobs(objs):
    small = []
    big = []
    for o in objs:
        if o=="DeskLamp" or o in INVEN_OBJS:
            small.append(o)
        if o in RECEPS:
            big.append(o)
    return small, big

def openables(objs):
    to_open = []
    to_put = []
    #convert to list if only 1 object is passed
    if isinstance(objs, list)==False:
        objs = [objs]
    
    for o in objs:
        if o in OPENS:
            to_open.append(o)
        if o in INVEN_OBJS:
            to_put.append(o)
    return to_put, to_open

def opens_toggles(objs):
    opto = []
    #convert to list if only 1 object is passed
    if isinstance(objs, list)==False:
        objs = [objs]
    
    for o in objs:
        if o in OPENS and o in TOGGLES:
            opto.append(o)
    return opto

def receps_toggles(objs):
    receps = []
    toggles = []
    #convert to list if only 1 object is passed
    if isinstance(objs, list)==False:
        objs = [objs]
    
    for o in objs:
        if o in RECEPS:
            receps.append(o)
        if o in TOGGLES:
            toggles.append(o)
    return receps,toggles

def sliceables(objs):
    to_slice = []

    for o in objs:
        if o in SLICES:
            to_slice.append(o)

    return to_slice

def toggleables(objs):
    to_on = []
    #convert to list if only 1 object is passed
    if isinstance(objs, list)==False:
        objs = [objs]
    for o in objs:
        if o in TOGGLES:
            to_on.append(o)
    return to_on

def possible_receptacle(objs,said_object,bias = []):
    possib = []
    if bias==[]:
        bias = objs
    else:
        new_objs = []
        for o in objs:
            for b in bias:
                if b+'|' in o:
                    new_objs.append(o)
        objs = new_objs
    print("manipulation_signatures.py -> possible_receptacle ",objs)
    print("Said object ",said_object)

    for o in objs:
        for r in RECEPS:
            if r+'|' in o:
                possib.append(o)
    
    if said_object in CONFUSIONS.keys():
        if isinstance(CONFUSIONS[said_object],list):
            for c in CONFUSIONS[said_object]:
                if c +'|' in possib:
                    print("Resolved confusion to be ",c)
                    return c #return whatever you saw that might be close to the said object
        elif CONFUSIONS[said_object]+'|' in possib:
            print("Resolved confusion to be ",c)
            return c #return whatever you saw that might be close to the said object


    return random.choice(possib)