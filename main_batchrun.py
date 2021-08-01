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

#================= Parameters ==========================
rooms = range(301,305)
task_idcs = range(0,10)
trial_nums = range(0,1) #the same room and same task can be executed in upto 3 ways by different crowd workers
use_gt_map = True
if use_gt_map:
    inp = 'y'
else:
    inp = 'n'
#================= Parameters ==========================



args = {'data_path':'robot/data', 'num_threads':1, 'reward_config':'robot/data/config/rewards.json', 'shuffle':False, 'smooth_nav':True, 'time_delays':False}
task_args = Namespace(**args)
env = sensing()

for i1 in rooms:
    for i2 in task_idcs:
        for i3 in trial_nums:
            print("*******************************************************")
            print("*******************************************************")
            print("------------- ROOM ---------------------------------",i1)
            print("------------- task index ---------------------------",i2)
            print("------------- trial number -------------------------",i3)
            print("*******************************************************")
            print("*******************************************************")
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

            if inp=='y':
                hlp.run(env,sentences,list_intent,list_dic_parsing,int(i1))
            else:
                hlp.run(env,sentences,list_intent,list_dic_parsing)

            print("\n\n\n\n")






