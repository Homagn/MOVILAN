#Goes through the entire alfred dataset and logs all the natural language instructions in a json file
import json
import traceback
import glob

trajectory_data_location = '/alfred/data/json_2.1.0/train/*-'

lang_data = {}

for rn in range (0,30):
    lang_data[str(rn)] = {}
    for task_index in range(0,100):
        lang_data[str(rn)][str(task_index)] = {}
        for trial_num in range(0,2):

            try:
                folders = sorted(glob.glob(trajectory_data_location+repr(rn)))
                print("Number of demonstrated tasks for this room ",len(folders))
                trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
                print("Number of different trials (language instr) for the same task ",len(trials))
                traj = glob.glob(trials[trial_num]+'/*.json')
                print("got trajectory file ",traj)
                with open(traj[0]) as f:
                    traj_data = json.load(f)

                
            except:
                traceback.print_exc()
                #print("folder does not exist")
                continue

            #print(traj_data['turk_annotations'])#['anns'][0]["high_descs"][0])

            
            
            lang_data[str(rn)][str(task_index)][str(trial_num)] = traj_data['turk_annotations']['anns'][0]["high_descs"]

            j = json.dumps(lang_data, indent =4)
            f = open('language_instructions.json', 'w')
            print(j, file=f)
            f.close()