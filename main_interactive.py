#This code designed to fetch all the trajectory data from the alfred folder in an organized fashion
import os
from os import path
import sys
from argparse import Namespace

os.environ['LU'] = 'language_understanding'
sys.path.append(os.path.join(os.environ['LU']))

os.environ['M'] = 'mapper'
sys.path.append(os.path.join(os.environ['M']))

os.environ['P'] = 'planner'
sys.path.append(os.path.join(os.environ['P']))



from robot.sensing import sensing

from language_understanding import equivalent_concepts as eqc
from language_understanding import parse_funcs as pf
from language_understanding import instruction_parser as ip

from planner import high_level_planner as hlp

print("------------- Welcome to the MOVILAN execution environment -------------")
print("This code uses the Ai2thor execution environment with the ALFRED dataset ")
print("A demonstration is shown is here to start you need to enter room number, task and trial number ")
print("Example : room number (enter 301), task number (enter 1), trial number (enter 0) ")
print("If you chose ground truth grid occupancy data for the room (y) then a prestroed map of the room will be used ")


i1 = input("Enter the room number to start the instance (ex- 301) ")
i2 = input("Enter the task index (ex- 1) ")
i3 = input("Enter the trial number (ex- 0) ")

args = {'data_path':'robot/data', 'num_threads':1, 'reward_config':'robot/data/config/rewards.json', 'shuffle':False, 'smooth_nav':True, 'time_delays':False}
task_args = Namespace(**args)
env = sensing()
env.prepare_navigation_environment(room = int(i1), task_index = int(i2), trial_num = int(i3))
env.set_task(task_completion_arguments = task_args)

sent_list = env.get_instructions()

p = ip.parse() #takes around 0.05s for parsing a single sentence
sentences = env.traj_data['turk_annotations']['anns'][0]["high_descs"]
list_intent, list_dic_parsing = p.predict(sent_list)

for s in range(len(list_intent)):
    print("SENTENCE ->", env.traj_data['turk_annotations']['anns'][0]["high_descs"][s])
    print("INTENT ->", list_intent[s])
    print("SLOTS ->", list_dic_parsing[s])
    print(" ")

inp = input("Use ground truth grid occupancy data for the mapping module ? (y/n) ")
if inp=='y':
    hlp.run(env,sentences,list_intent,list_dic_parsing,int(i1), interactive = False)
else:
    hlp.run(env,sentences,list_intent,list_dic_parsing, interactive = False)
