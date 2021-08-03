import os
import sys
os.environ['MAIN'] = '../'
sys.path.append(os.path.join(os.environ['MAIN']))
from robot.sensing import sensing


import numpy as np
import graph_convnet as gcn

import random

import gtmaps as gtm
import projection as proj
import panorama as pan
import mapper.params as params

import math
import copy

TRAIN_LABELS = '/ai2thor/mapper/data/targets/'


class graph_conv_mapper(object):
    def __init__(self):
        ##############################################################################
        #Initialize
        self.rooms = list(range(301,307))
        self.grid_size = params.grid_size
        self.neighborhood = params.node_neighborhood
        self.concat_gridxy = params.concat_gridxy #concat the grid xy values also to the node labels?

        xy = 0
        if self.concat_gridxy:
            xy = 2

        nvd = (2*self.neighborhood+1)**2


        nhd = params.node_hidden_dim
        sem_class = params.semantic_classes
        layer_type = params.graph_conv_layer_type
        self.n_classes = len(sem_class)

        nvd = nvd*(self.n_classes+xy)
        print("nvd ",nvd)

        #This time load the weights of the GCN while initializing
        self.ng = gcn.nav_graph( grid_size = self.grid_size, 
                                embed_dim = params.node_embed_dim, 
                                node_value_dim = nvd, 
                                hop_connection = params.hop_connection,
                                num_classes = len(sem_class), 
                                node_hidden_dim = nhd, 
                                train_parts = ["spatial"],
                                layer_type = layer_type,
                                load_weights = True)

        print("Initialized graph convolution network ")
        #! need to load trained GCN weights


    def test_instance(self, room, pos):
        #! need checks to see if the position is premapped otherwise load the input bev maps bypassing panorama step
        try:
            #o_grids = np.load(TRAIN_LABELS+repr(room)+'.npy',allow_pickle = 'TRUE').item()
            #pos = [int((pos[0]-o_grids['min_pos']['mx'])/0.25), int((pos[1]-o_grids['min_pos']['mz'])/0.25)]
            print("loading previous projection from position ",pos)
            camera_proj = np.load('/ai2thor/mapper/data/inputs/bev_'+repr(room)+'_'+repr(pos)+'.npy',allow_pickle = 'TRUE').item()
            return True
        except:
            print("Either this position is not reachable ...")
            print("Or this room and position is not premapped")
            return False

    def account_visibilities(self):
        #(needs to be called after sample_data function)
        #takes in the self.nav_map_t matrix and with respect to the current position of agent (the center grid of the matrix),
        #finds out other grid coordinates that may not be visible to the agent because of blockage by walls/objects
        #using a simple ray shooting and intersection technique
        #these node that are found out to be invisible are not set as training targets to the GCN and thus 
        #the GCN is not penelized unnecessarily for nodes that are not visible 

        mat = self.nav_map_t #(available after sample_data function)
        agent_loc = [int(self.grid_size/2),int(self.grid_size/2)] #its the center coordinate (because ego map)

        blocked_nodes = []
        step_res = 0.05

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i==agent_loc[0] and j==agent_loc[1]:
                    #print("i and j are same")
                    continue
                if mat[i,j]==params.semantic_classes['tar']:
                    # the blocked nodes can be the target themselves - in that case training would be hampered so prevent it
                    # however we still dont want to predict a completely invisible target behind the wall
                    # later another condition that checks input panorama projection for at least 1 projected target element catches this 
                    continue

                t = [i,j]

                if i==agent_loc[0]: #means the center row
                    slope = 1.0
                else:
                    slope = math.fabs((t[1]-agent_loc[1])/(t[0]-agent_loc[0]))

                if t[0]-agent_loc[0]>0.0:
                    x_dir = -1.0
                if t[0]-agent_loc[0]<=0.0:
                    x_dir = 1.0

                if t[1]-agent_loc[1]>0.0:
                    y_dir = -1.0
                if t[1]-agent_loc[1]<=0.0:
                    y_dir = 1.0



                #print("starting ray casting slope = ",slope)
                #print("agent center ",agent_loc)
                x = t[0]
                y = t[1]
                x1 = copy.copy(x)
                y1 = copy.copy(y)

                step = step_res
                #print("caster x ",x," y ",y, " i ",i," j ",j)
                
                while(x1!=agent_loc[0] or math.fabs(y1-agent_loc[1])>=1): #rays are coming from each point to the agent location
                    if x1!=agent_loc[0]:
                        x1 = round(x+ x_dir*step)
                    if y1!=agent_loc[1]:
                        y1 = round(y + y_dir*step*slope)
                    step+=step_res
                    '''
                    if x==16:
                        print("x ",x1," y ",y1)
                    '''
                    if x1>=self.grid_size:
                        break

                    if mat[x1,y1]==params.semantic_classes['unk'] or mat[x1,y1]==params.semantic_classes['obs']:
                        if x1==x and y1==y: #dont account the starting node from which rays are cast
                            #ray has been interrupted by obstacle
                            pass
                        else:
                            #print("Ray has been blocked")
                            blocked_nodes.append((i,j))
                            break
        return blocked_nodes

    def visualize(self):
        #=================================
        #to test whether the correct ground truth labels are being sampled 
        nav_map_t = gtm.target_navigation_map(  self.o_grids, self.target_object, 
                                                {'x':self.a*0.25+ self.o_grids['min_pos']['mx'], 'y':0.0, 'z': self.b*0.25+ self.o_grids['min_pos']['mz']}, 
                                                grid_size = self.grid_size, 
                                                unk_id = params.semantic_classes['unk'],
                                                flr_id = params.semantic_classes['flr'], 
                                                tar_id = params.semantic_classes['tar'], 
                                                obs_id = params.semantic_classes['obs'])
        self.nav_map_t = nav_map_t
        blocked_nodes = self.account_visibilities() #done based on current self.nav_map_t

        print("Showing ground truth map")
        #gtm.prettyprint(nav_map_t)
        proj.starviz(nav_map_t, blocked_nodes = blocked_nodes)
        #=================================

        print("Showing inputs to GCN (argmaxes of each element in approximate projection) ")
        proj.prettyprint(self.nav_map,argmax = True) #it cant print without argmax because each element of nav_map is 4 dimensional


        print("Showing predicted labeled grid ")
        proj.starviz(self.lg, blocked_nodes = blocked_nodes)

    def check_error(self):
        nav_map_t = gtm.target_navigation_map(  self.o_grids, self.target_object, 
                                                {'x':self.a*0.25+ self.o_grids['min_pos']['mx'], 'y':0.0, 'z': self.b*0.25+ self.o_grids['min_pos']['mz']}, 
                                                grid_size = self.grid_size, 
                                                unk_id = params.semantic_classes['unk'],
                                                flr_id = params.semantic_classes['flr'], 
                                                tar_id = params.semantic_classes['tar'], 
                                                obs_id = params.semantic_classes['obs'])
        self.nav_map_t = nav_map_t
        #nav_map_t ->ground truth map
        #self.nav_map ->input approx projection
        #self.lg ->predicted map using GCN correction
        num_correct_approx = 0
        num_correct_pred = 0
        accounted_nodes = 0

        blocked_nodes = self.account_visibilities() #done based on current self.nav_map_t

        for i in range(nav_map_t.shape[0]):
            for j in range(nav_map_t.shape[1]):
                if (i,j) in blocked_nodes:
                    pass

                else:
                    accounted_nodes+=1

                    element_gt = nav_map_t[i,j]
                    element_approx = np.argmax(self.nav_map[i,j,:])
                    element_pred = self.lg[i,j]
                    #print("Element gt -> ",element_gt, " element approx -> ",element_approx, " element pred -> ",element_pred)
                    
                    if element_approx == params.semantic_classes['tar'] and element_gt == params.semantic_classes['tar']:
                        num_correct_approx+=1
                    if element_approx == params.semantic_classes['obs'] and element_gt == params.semantic_classes['obs']:
                        num_correct_approx+=1
                    if element_approx == params.semantic_classes['flr'] and element_gt == params.semantic_classes['flr']:
                        num_correct_approx+=1
                    
                    if element_approx == params.semantic_classes['unk'] and element_gt == params.semantic_classes['unk']:
                        num_correct_approx+=1
                    


                    if element_pred == params.semantic_classes['tar'] and element_gt == params.semantic_classes['tar']:
                        num_correct_pred+=1
                    if element_pred == params.semantic_classes['flr'] and element_gt == params.semantic_classes['flr']:
                        num_correct_pred+=1
                    if element_pred == params.semantic_classes['obs'] and element_gt == params.semantic_classes['obs']:
                        num_correct_pred+=1
                    
                    if element_pred == params.semantic_classes['unk'] and element_gt == params.semantic_classes['unk']:
                        num_correct_pred+=1
                

                '''
                if element_approx==element_gt:
                    num_correct_approx+=1
                if element_pred==element_gt:
                    num_correct_pred+=1
                '''
        print("Percentage correct approx ",float(num_correct_approx/accounted_nodes))
        print("Percentage correct pred ",float(num_correct_pred/accounted_nodes))


    def add_inputs(self, room, pos, target_object):
        #pos will be passed in form of dictionary {x: , y: , z:}
        ##############################################################################

        if isinstance(room,int): #ground truth accessible
            #prepare input for GCN
            #o_grids = np.load(TRAIN_LABELS+repr(room)+'.npy',allow_pickle = 'TRUE').item()
            print("loading previous projection from position ",pos)
            camera_proj = np.load('/ai2thor/mapper/data/inputs/bev_'+repr(room)+'_'+repr([pos[1],pos[0]])+'.npy',allow_pickle = 'TRUE').item()
        else:
            #the variable room also doubles as the environment variable when ground truth inaccessible
            print("Ground truth room occupancy grid is not available")
            gridsize = params.grid_size
            panorama = pan.rotation_image(room, objects_to_visit = [], debug = False) #gets a panorama image of everything thats visible
            camera_proj = proj.bevmap(panorama,grid_size = gridsize, debug = False)



        nav_map = proj.input_navigation_map(camera_proj, target_object, grid_size = params.grid_size, 
                                            unk_id = params.semantic_classes['unk'],
                                            flr_id = params.semantic_classes['flr'], 
                                            tar_id = params.semantic_classes['tar'], 
                                            obs_id = params.semantic_classes['obs'])
        #nav_map is an array of params.grid_size x params.grid_size x 4 (4 classes unk,flr,tar and obs)| contains values between 0 and 1
        
        self.nav_map = nav_map
        if params.debug_viz:
            print("mapper/test_gcn.py line 268 visualizing input to GCN")
            proj.prettyprint(self.nav_map,argmax = True) #it cant print without argmax because each element of nav_map is 4 dimensional



        ##############################################################################
        #pass input data to the graph convolutional neural network class
        self.ng.clear_input()

        for i in range(self.grid_size): 
            for j in range(self.grid_size):
                #l = []
                n_v = []
                for m in range(i-self.neighborhood,i+self.neighborhood+1):
                    for n in range(j-self.neighborhood,j+self.neighborhood+1):
                        try:
                            #labl = labls[m,n,:]
                            node_inp = nav_map[m,n,:]
                        except: #means outside map border, so unknown
                            u = [0]*self.n_classes
                            u[params.semantic_classes['unk']] = 1.0
                            node_inp = np.array(u, dtype = np.float32)
                        
                        if self.concat_gridxy:
                            #l.extend(Labl+[(m-(grid_size/2.0))/float(grid_size), (n-(grid_size/2.0))/float(grid_size)])
                            n_v.extend(node_inp.tolist()+[(m+self.neighborhood)/float(self.grid_size+self.neighborhood), 
                                                          (n+self.neighborhood)/float(self.grid_size+self.neighborhood)])
                        else:
                            n_v.extend(node_inp.tolist())

                self.ng.add_input(self.grid_size*(i)+(j), np.array(n_v)) 
        print("successfully added inputs to GCN")

    def estimate_map(self, room, pos, target_object, motion):
        #fuction for  G.ng.vis_matrix 
        #hutils/navigation_signatures.py chunk of code from line 470 to 493
        self.add_inputs(room, pos, target_object)
        self.ng.test_once(motion) #should be called only after add_inputs function
        infer_proj = self.ng.vis_matrix() #32x32 numpy matrix contains 10 where target, 1 where floor, and 0 where obstacle/unmap
        if params.debug_viz:
            print("mapper/test_gcn.py line 306, visualizing GCN prediction map")
            proj.starviz(infer_proj)
        return infer_proj

    def aggregate_belief(self,room,pos, target_object, motion, var = 0, thresh = 1.0):
        if isinstance(room,int): #-1 is supplied for unknwon room pure testing
            o_grids = np.load(TRAIN_LABELS+repr(room)+'.npy',allow_pickle = 'TRUE').item()
        else:
            o_grids = {}
        
        if isinstance(room,int):
        #if pos[0]>0 and pos[1]>0:
            a,b = int((pos[0]-o_grids['min_pos']['mx'])/0.25), int((pos[1]-o_grids['min_pos']['mz'])/0.25)
            print("Ground truth accessible- Obtained agent standing grid pos ",a,b)
        else:
            a,b = -1, -1
        
        self.o_grids = o_grids
        self.target_object = target_object
        self.b = b
        self.a = a
        
        
        lg = self.estimate_map(room, [a,b], target_object, motion)
        self.lg = lg

        return lg

    def benchmark(self):
        #set parameters for testing
        
        #check_rooms = [301,302,304,305,306,313,314,321,322,328,329]
        #check_targets = ["Bed","Desk","Dresser","SideTable","Chair","ArmChair","DiningTable","Shelf","Safe","Sofa","nav_space"]

        check_rooms = [301]
        check_targets = ["Bed"]

        for rn in check_rooms:
            print("Checking room ",rn)
            o_grids = np.load(TRAIN_LABELS+repr(rn)+'.npy',allow_pickle = 'TRUE').item()
            nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()
            nav_pos_real = [ [n[0]*0.25 + o_grids['min_pos']['mx'], n[1]*0.25 + o_grids['min_pos']['mz']] for n in nav_pos ]

            nav_pos_real = [nav_pos_real[0]] #for debugging purposes only

            for n in nav_pos_real:
                print("Checking position ",n)
                a, c = n[0], n[1]
                for target_object in check_targets:
                    motion = [0,0]
                    labeled_grid = self.aggregate_belief(rn, [a,c], target_object, [0,0])
                    self.visualize()
                    self.check_error()
            

def estimate_map(target_object, localize_params = {'room':304,'position':[0.75, 2.0] } ):
    gcm = graph_conv_mapper()
    #in localize_params dict room can be an object or a room number (when ground truth map is there)
    if isinstance(localize_params['room'],int)==False: #pure testing unknwon room/ room variable is now environment object
        rn = localize_params['room']
        a,c = -1,-1
    else:
        rn = localize_params['room']
        pos = [ localize_params['position'][0], localize_params['position'][1] ]
        o_grids = np.load(TRAIN_LABELS+repr(rn)+'.npy',allow_pickle = 'TRUE').item()
        a,b = int((pos[0]-o_grids['min_pos']['mx'])/0.25), int((pos[1]-o_grids['min_pos']['mz'])/0.25)
        print("Ground truth accessible- Obtained agent standing grid pos ",a,b)
        
        #print("available keys ",o_grids.keys())
        nav_map_t = gtm.target_navigation_map(  o_grids, target_object, 
                                                {'x':a*0.25+ o_grids['min_pos']['mx'], 'y':0.0, 'z': b*0.25+ o_grids['min_pos']['mz']}, 
                                                grid_size = gcm.grid_size, 
                                                unk_id = params.semantic_classes['unk'],
                                                flr_id = params.semantic_classes['flr'], 
                                                tar_id = params.semantic_classes['tar'], 
                                                obs_id = params.semantic_classes['obs'])
        #a,c = localize_params['position'][0], localize_params['position'][1] #task 2 / agents 3D position can be obtained by running datagen.py --room <> --task <> --checkpan 
        if params.debug_viz:
            print("mapper/test_gcn.py line 384, visualizing ground truth map")
            proj.starviz(nav_map_t)
        return nav_map_t

    motion = [0,0]
    labeled_grid = gcm.aggregate_belief(rn, [a,c], target_object, [0,0])
    #gcm.visualize()

    return labeled_grid


if __name__ == '__main__':
    '''
    Note on inferencing
    (first read the notes in train_gcn.py main function to get idea how the temporal aggregation works)
    each temporal aggregation consists of supplying a unit agent step direction to update the previous hiddens in the LSTM
    consider- labeled_grid = gcm.aggregate_belief(rn, [a,c], target_object, [0,0])
    so passing motion = [0,0] (the last argument) means that the agent did not move this time step from the grid position [a,c]
    however say next time the agent moved 1 step forward, then to aggregate temporally the new prediction would be obtained as
    labeled_grid = gcm.aggregate_belief(rn, [a+1,c], target_object, [1,0]) 
    a+1 to account for the new spatial panorama data, and the motion is now [1,0] to account that the agent moved 1 step forward from [a,c]
    everytime time step in the agent trajectory following are the options to pass for motion- [0,0],[-1,0],[1,0],[0,-1],[0,1]
    if you want to do a trjectory reset for the temporal memory (sy agent entered new room or given a new task)
    do-> gcm.reset_memory()
    the maximum length of the trajectory is affected by the size of the persistent map in self,map_size in TAN class in graph_convnet.py
    currently around 15-20 steps from the agent starting point should work
    '''

    gcm = graph_conv_mapper()

    
    #===========================================================================================
    #(specific testing)
    #set parameters for testing
    
    """
    #================
    rn = 301
    a,c = -0.75, -1.25 #task 0 / agents 3D position can be obtained by running datagen.py --room <> --task <> --checkpan 
    target_object = 'Bed'
    #================
    """

    """
    #================
    rn = 302
    a,c = 0.25, 0.25 #task 1 / agents 3D position can be obtained by running datagen.py --room <> --task <> --checkpan 
    target_object = 'Desk'
    #================
    """

    """
    #================
    rn = 303
    a,c = 1.5, 0.25 #task 2 / agents 3D position can be obtained by running datagen.py --room <> --task <> --checkpan 
    target_object = 'Shelf'
    #================
    """

    #================
    rn = 304
    a,c = 0.75, 2.0 #task 2 / agents 3D position can be obtained by running datagen.py --room <> --task <> --checkpan 
    target_object = 'Desk'
    #================


    

    motion = [0,0]
    labeled_grid = gcm.aggregate_belief(rn, [a,c], target_object, [0,0])

    
    #visualization for debugging purposes
    gcm.visualize()
    #============================================================================================
    


    #===========================================================================================
    #(for benchmarking with respect to approx proj map)
    #gcm.benchmark()



    


        