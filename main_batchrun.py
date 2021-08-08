#This code designed to fetch all the trajectory data from the alfred folder in an organized fashion
import os
from os import path
import sys
from argparse import Namespace
import json
import csv

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


#function for writing csv files 
def writelog(fname, values):
    file1 = open(fname, 'a')
    writer = csv.writer(file1)
    fields1=values #is a list
    writer.writerow(fields1)
    file1.close()



print("------------- Welcome to the MOVILAN execution environment -------------")

#================= Parameters ==========================
rooms = range(301,315)
task_idcs = range(0,30)
trial_nums = range(0,3) #the same room and same task can be executed in upto 3 ways by different crowd workers
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

            try:
                env.prepare_navigation_environment(room = int(i1), task_index = int(i2), trial_num = int(i3))
                env.set_task(task_completion_arguments = task_args)

                sent_list = env.get_instructions()

            except:
                continue #some rooms do not have 30 demonstrated tasks

            print("*******************************************************")
            print("*******************************************************")
            print("------------- ROOM ---------------------------------",i1)
            print("------------- task index ---------------------------",i2)
            print("------------- trial number -------------------------",i3)
            print("*******************************************************")
            print("*******************************************************")

            p = ip.parse() #takes around 0.05s for parsing a single sentence
            sentences = env.traj_data['turk_annotations']['anns'][0]["high_descs"]
            list_intent, list_dic_parsing = p.predict(sent_list)

            for s in range(len(list_intent)):
                print("SENTENCE ->", env.traj_data['turk_annotations']['anns'][0]["high_descs"][s])
                print("INTENT ->", list_intent[s])
                print("SLOTS ->", list_dic_parsing[s])
                print(" ")

            if inp=='y':
                task_tracker = hlp.run(env,sentences,list_intent,list_dic_parsing,int(i1))
            else:
                task_tracker = hlp.run(env,sentences,list_intent,list_dic_parsing)

            print("\n\n\n\n")
            


            #logging csv file for benchmark
            
            values = ["Room number ",i1,"Task number ",i2, "Trial number ", i3, "Goals satisfied (0/1) "]
            if task_tracker["goal_satisfied"]== True:
                values.append(1)
            else:
                values.append(0)
            values.append("subgoal idx")
            values.append(task_tracker["subgoal_idx"])
            values.append(task_tracker["post_conditions"][0])
            values.append("out of")
            values.append(task_tracker["post_conditions"][1])
            values.append("number of instructions ")
            values.append(len(task_tracker["task"]))

            values.append("Algo path length ")
            values.append(task_tracker["trajectory_length"])


            exp_length = task_tracker["exp_length"]

            values.append("Expert path length ")
            values.append(exp_length)

            values.append("ratio")
            #r = float(task_tracker["trajectory_length"])/float(max(task_tracker["trajectory_length"], exp_length))
            r = float(exp_length)/float(max(task_tracker["trajectory_length"], exp_length))
            values.append(r)

            writelog('benchmark_results.csv',values)









