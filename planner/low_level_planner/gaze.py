import os
import sys
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))


def gaze(action_desc,env):
    #sample input->['up']
    print("(manipulation_signatures.py -> gaze)")
    #print("Trying this for ",numtries," time")
    if action_desc[0] == "up":
        env.step(dict({"action": "LookUp"}))
    if action_desc[0]=="down":
        env.step(dict({"action": "LookDown"}))
    return env

