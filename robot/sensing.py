import os
import sys
os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
import robot.params as params
import json
import glob

#serves as a wrapper class for all different kinds of vision language navigation environment
#be it ai2thor, custom gazebo, R2R or any other unity based stuff, or even real robot
#currently serves as wrapper for only ai2thor simulator

class sensing(object):
    def __init__(self, abstraction = 'ai2thor'):
        self.abstraction = abstraction
        self.perception = {} #when using any abstraction should store all inputs gathered from the environment
        self.cur_traj_len = 0
        camera = params.camera

        if abstraction=='ai2thor':
            os.environ['ALFRED_ROOT'] = '/alfred'
            sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
            sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
            from env.thor_env import ThorEnv
            self.env = ThorEnv(player_screen_width=camera['width'],player_screen_height=camera['height']) #blank ai2thor environment

        self.init_memory()

    def init_memory(self):
        self.memory = {}
        self.memory['tracked_targets'] = ["NOTHING"]
        self.memory['tracked_refinements'] = ["NOTHING"]
        self.memory['tracked_target_nums'] = []
        self.memory['navigated'] = ["NOTHING"]


    def reset(self, scene_name):
        if self.abstraction=='ai2thor':
            return self.env.reset(scene_name)
        else:
            raise NotImplementedError
            return

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        if self.abstraction=='ai2thor':
            return self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        else:
            raise NotImplementedError
        return


    

    def get_goal_satisfied(self):
        if self.abstraction=='ai2thor':
            return self.env.get_goal_satisfied()
        else:
            raise NotImplementedError
        return

    def get_subgoal_idx(self):
        if self.abstraction=='ai2thor':
            return self.env.get_subgoal_idx()
        else:
            raise NotImplementedError
        return

    def get_postconditions_met(self):
        if self.abstraction=='ai2thor':
            #return self.env.get_postconditions_met()
            return self.env.get_goal_conditions_met()
        else:
            raise NotImplementedError
        return


    def get_instructions(self):
        if self.abstraction=='ai2thor':
            print("Task: %s" % (self.traj_data['turk_annotations']['anns'][0]["high_descs"]))
            return self.traj_data['turk_annotations']['anns'][0]["high_descs"]
        else:
            raise NotImplementedError
        return

    def set_task(self, **kwargs):
        if self.abstraction=='ai2thor':

            task_completion_arguments = kwargs['task_completion_arguments']
            self.env.set_task(self.traj_data, task_completion_arguments, reward_type='dense')
        else:
            raise NotImplementedError
        return
        

    def prepare_navigation_environment(self, **kwargs):
        if self.abstraction=='ai2thor':
            rn = kwargs['room']
            task_index = kwargs['task_index']
            trial_num = kwargs['trial_num']

            self.rn = rn
            self.task_index = task_index
            self.trial_num = trial_num

            #obtain the trajectory folder that describes the entire scene settings and also stores the task instruction
            #folders = sorted(glob.glob('/alfred/data/json_2.1.0/train/*-'+repr(rn))) #for home computer
            folders = sorted(glob.glob(params.trajectory_data_location+repr(rn)))
            print("Number of demonstrated tasks for this room ",len(folders))
            trials = glob.glob(folders[task_index]+'/*') #there would be len(folders) number of different tasks 
            print("Number of different trials (language instr) for the same task ",len(trials))
            traj = glob.glob(trials[trial_num]+'/*.json')
            print("got trajectory file ",traj)

            with open(traj[0]) as f:
                traj_data = json.load(f)


            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            object_toggles = traj_data['scene']['object_toggles']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            # reset
            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            print("setting orientation of the agent to facing north ")
            traj_data['scene']['rotation'] = 0
            self.step(dict(traj_data['scene']['init_action']))
            self.traj_data = traj_data
            self.cur_traj_len = 0



        else:
            raise NotImplementedError
            return

    #========================   Sensing functions return some sensed data from the abstraction environment  ===========================
    def get_position(self): #equivalent to indoor GPS use only when perfect map is available 
        x,y,z = self.perception.metadata['agent']['position']['x'], self.perception.metadata['agent']['position']['y'], self.perception.metadata['agent']['position']['z']
        return x,y,z
    def get_rotation(self): #equivalent to a compass
        return self.perception.metadata['agent']['rotation']['y']
    def get_segmented_image(self):
        return self.perception.instance_segmentation_frame
    def get_depth_image(self):
        return self.perception.depth_frame
    def identify_segmented_color(self,c):
        for color in self.perception.metadata['colors']:
            if color['color'][0] == c[0] and color['color'][1] == c[1] and color['color'][2] == c[2]:
                return color['name']
        return "Nothing"
    def actuator_success(self):
        return self.perception.metadata['lastActionSuccess']
    def check_inventory(self):
        return self.perception.metadata['inventoryObjects']



    #========================   Action functions that make the abstraction perform some actions ===========================
    #while benchmarking panorama assumptions have been made so sometimes some agent actions performed in order to gather the panorama are not counted
    #count_step is a flag used for that purpose
    def step(self, trajectory, count_step  = True): 
        #NOTE - step can do a lot of actions in ai2thor environment based on the keyword supplied in the trajectory
        #to get a list of all different things step can do visit their website
        #if another environment is used, their step function should have similar functionalities as ai2thor
        if self.abstraction=='ai2thor':
            self.perception = self.env.step(trajectory)
            if count_step:
                self.cur_traj_len+=1
            return self.perception
        else:
            raise NotImplementedError
        return

    def check_collision(self,direction):
        if self.abstraction=='ai2thor':
            #https://ai2thor.allenai.org/ithor/documentation/navigation/
            #event = self.step(dict({"action": "Done"})) #provides a cleaned up metedata
            #provides a cleaned up metedata
            event = self.step(dict({"action": "LookUp"})) 
            event = self.step(dict({"action": "LookDown"})) 
            x1,y1,z1 = self.get_position()

            event = self.step(dict({"action": direction}))
            x2,y2,z2 = self.get_position()
            col = (x1==x2 and z1==z2)
            return event, col
        
        else:
            raise NotImplementedError
        return

    def custom_rotation(self, horizon, lateral):
        if self.abstraction=='ai2thor':
            x,y,z = self.get_position()
            rot = self.get_rotation()

            custom_rot = {"action": "TeleportFull","horizon": horizon,"rotateOnTeleport": True,"rotation": rot-lateral,"x": x,"y": y,"z": z}
            self.step(dict(custom_rot))
        
        else:
            raise NotImplementedError
        return

    def set_rotation(self, horizon, lateral):
        if self.abstraction=='ai2thor':
            x,y,z = self.get_position()
            custom_rot = {"action": "TeleportFull","horizon": horizon,"rotateOnTeleport": True,"rotation": lateral,"x": x,"y": y,"z": z}
            self.step(dict(custom_rot))
        
        else:
            raise NotImplementedError
        return