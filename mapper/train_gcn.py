import numpy as np
import graph_convnet as gcn
import params
import random
import traceback

import gtmaps as gtm
import projection as proj

import AStarSearch as ast

from os import path
import sys
import math
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--check', dest='check', action='store_true') 
args = parser.parse_args()

TRAIN_LABELS = '/ai2thor/mapper/data/targets/'


class graph_conv_trainer(object):
    def __init__(self, trainpart = "s"):
        ##############################################################################
        #Initialize
        #self.rooms = list(range(301,330))
        self.trainpart = trainpart
        #self.rooms = [301,302,304,305,306,313,314,321,322,328,329] #these are good bedrooms
        self.rooms = params.trainrooms
        #311,318,319,323,324,330 are especially bad because they have places where agent can get shut off from other parts by walls
        #308,315,325,326 are not able to load

        
        #self.good_objects = ["Bed","Desk","Dresser","SideTable","Chair","ArmChair","DiningTable","Shelf","Safe","Sofa","nav_space"]
        self.good_objects = params.trainobjects


        self.grid_size = params.grid_size
        self.neighborhood = params.node_neighborhood
        self.concat_gridxy = params.concat_gridxy #concat the grid xy values also to the node labels?

        self.max_wander = 20 #agent can wander around the same room for atmost 10 time steps
        self.motions = [] #first time step agent does not move
        self.past_positions = []
        self.prev_blocked_nodes = []


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

        if trainpart=="s":
            self.ng = gcn.nav_graph( grid_size = self.grid_size, 
                                    embed_dim = params.node_embed_dim, 
                                    node_value_dim = nvd, 
                                    hop_connection = params.hop_connection,
                                    num_classes = len(sem_class), 
                                    node_hidden_dim = nhd,  
                                    train_parts = ["spatial"],
                                    layer_type = layer_type,
                                    load_weights = False)
        if trainpart=="t":
            self.ng = gcn.nav_graph( grid_size = self.grid_size, 
                                    embed_dim = params.node_embed_dim, 
                                    node_value_dim = nvd, 
                                    hop_connection = params.hop_connection,
                                    num_classes = len(sem_class), 
                                    node_hidden_dim = nhd,  
                                    train_parts = ["temporal"],
                                    layer_type = layer_type,
                                    load_weights = True) #needs to load previous trained spatial weights

        print("Initialized graph convolution network ")


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

    def data_consistency_check(self):
        #for debugging puposes
        #checks whether all saved data in inputs/ folder have corresponding labels in target/ folder
        self.room = random.choice(self.rooms)
        o_grids = np.load(TRAIN_LABELS+repr(self.room)+'.npy',allow_pickle = 'TRUE').item()
        self.o_grids = o_grids
        nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()

        gtm.prettyprint(o_grids['nav_space']) #navigable space in the map considering all obstructions

        for i in nav_pos:
            print("checking position ",i)
            p = [i[1],i[0]]
            #p = i #this is wrong ordering
            camera_proj = np.load('data/inputs/bev_'+repr(self.room)+'_'+repr(p)+'.npy',allow_pickle = 'TRUE').item()

        sys.exit(0)

    def sample_trajectory(self, nav, cur_pos, tar_pos):
        
        g = ast.Graph()
        for i in range (nav.shape[0]):
            for j in range (nav.shape[1]):
                try:
                    if nav[i-1,j]==1 and nav[i,j]==1:
                        g.add_edge(repr(i-1)+'_'+repr(j),repr(i)+'_'+repr(j),1.0) 
                except:
                    #print(i-1)
                    pass
                try:
                    if nav[i+1,j]==1 and nav[i,j]==1:
                        g.add_edge(repr(i+1)+'_'+repr(j),repr(i)+'_'+repr(j),1.0)
                except:
                    #print(i+1)
                    pass
                try:
                    if nav[i,j+1]==1 and nav[i,j]==1:
                        g.add_edge(repr(i)+'_'+repr(j+1),repr(i)+'_'+repr(j),1.0)
                except:
                    #print(j+1)
                    pass
                try:
                    if nav[i,j-1]==1 and nav[i,j]==1:
                        g.add_edge(repr(i)+'_'+repr(j-1),repr(i)+'_'+repr(j),1.0)
                except:
                    #print(j-1)
                    pass
        source = repr(cur_pos[0])+'_'+repr(cur_pos[1])
        target = repr(tar_pos[0])+'_'+repr(tar_pos[1])
        path ,total_cost = g.dijsktra(source,target)

        path = [ [ int( p[:p.index('_')] ), int( p[p.index('_')+1:] ) ]  for p in path]

        deltas = []
        for p in range(1,len(path)):
            deltas.append( [ path[p][0]-path[p-1][0], path[p][1]-path[p-1][1] ] )


        #print("Got paths ",path)
        #print("length of path ",len(path))
        #print("Got deltas ",deltas)
        #sys.exit(0)
        return deltas
            




    def sample_data(self):
        bad_target = False

        ##############################################################################
        #chose a random room, obtain a navigable position, chose a random object in room to map and load the ground truth data
        #if starting a new trajectory
        if len(self.motions)<1 or len(self.motions)>=self.max_wander or self.trainpart=="s": #if only spatial training always do this
            self.motions.append([0,0])
            self.prev_blocked_nodes = []
            self.blocked_nodes = []

            self.ng.reset_memory() #resets the persistent map encoding of the graph convolution

            self.room = random.choice(self.rooms)
            bad_target = False
            #print("randomly chosen room ",self.room)

            o_grids = np.load(TRAIN_LABELS+repr(self.room)+'.npy',allow_pickle = 'TRUE').item()
            self.o_grids = o_grids

            nav_pos = np.argwhere(o_grids['nav_space']==1).tolist()
            self.nav_poses = nav_pos
            #print("list of all navigable grid locations ",nav_pos)
            #self.pos = [0,2] #random.choice(nav_pos)
            self.pos = random.choice(nav_pos)



            if self.trainpart=="t": #for temporal training agent needs to walk a trajectory
                #===================================================================================
                #(sample a position far away from initial chosen position and get the path to it)
                #(later that path will be used to sample ego-maps as a time sequence to the lstm part of GCN)
                samp_pos = copy.copy(self.pos)
                tries = 1000
                #wander_len = random.choice([self.max_wander,0]) #randomly also chose 0 len trajectories to better train just the spatial aggregatio part
                #print("randomly chosen wander length ",wander_len)
                wander_len = self.max_wander
                #try to find a target navigable position roughly 15 manhattan distance away
                while(math.fabs(samp_pos[0]-self.pos[0]) + math.fabs(samp_pos[1]-self.pos[1]) < wander_len+1):
                    tries = tries-1
                    samp_pos = random.choice(nav_pos)
                    if tries<0:
                        break

                if self.pos==samp_pos:
                    bad_target = True
                    print("Failed to find a nice trajectory due to initial choice of position")

                print("cur_pos ",self.pos, " samp_pos ",samp_pos)
                self.traj_deltas = self.sample_trajectory(self.o_grids['nav_space'], self.pos, samp_pos)
                #===================================================================================



            #===================================================================================
            #(chose a target object to focus on for the entire trajectory - it could be something as well as nothing- just navigable space)
            self.past_positions = [self.pos]
            room_objects = list(set(o_grids.keys())- set(['nav_space','fixed_obstructions','min_pos']))
            #remove object position identifiers
            room_objects = [o[:o.index('|')] for o in room_objects] + ["nav_space"]

            #ensure only good objects are sampled
            room_objects = list(set(room_objects).intersection(set(self.good_objects)))
            print("Loaded room ",self.room," possible room objects ",room_objects)

            if room_objects==[]:
                print("WARNING ! set intersection is empty")
                bad_target = True

            self.room_obj = random.choice(room_objects)
            print("Chosen object ",self.room_obj)
            #print("all objects in the room ",room_objects)
            #print("randomly chosen room object ",self.room_obj)
            #===================================================================================
            


        else:
            if len(self.traj_deltas)>0:
                random_motion = self.traj_deltas.pop(0) #self.traj_deltas should have the first element removed now
                new_pos = [self.pos[0]+random_motion[0],self.pos[1]+random_motion[1]]
                self.motions.append(random_motion)
                self.past_positions.append(new_pos)
                self.pos = copy.copy(new_pos)
                print("all past positions ",self.past_positions)
            else:
                print("trajectory exhausted ")
                return False



        #major bug- position os self.pos[0] and self.pos[1] was swapped earlier
        self.nav_map_t = gtm.target_navigation_map(  self.o_grids, self.room_obj, 
                                                {'x':self.pos[0]*0.25+ self.o_grids['min_pos']['mx'], 'y':0.0, 'z': self.pos[1]*0.25+ self.o_grids['min_pos']['mz']}, 
                                                grid_size = self.grid_size, 
                                                unk_id = params.semantic_classes['unk'],
                                                flr_id = params.semantic_classes['flr'], 
                                                tar_id = params.semantic_classes['tar'], 
                                                obs_id = params.semantic_classes['obs'])

        contains_target = params.semantic_classes['tar'] in self.nav_map_t.flatten().tolist()
        
        blocked_nodes = self.account_visibilities() #done based on current self.nav_map_t
        
        #=================================================================================================
        #account for previous blocked nodes from agent trajectory (what was not visible may now be visible)
        transformed_blocked = []
        m = self.motions[-1]
        for i in self.prev_blocked_nodes:
            transformed_blocked.append((i[0]-m[0], i[1]-m[1])) #lists of lists cannot go into set needs a list of tuples
        common_blocked_nodes = copy.copy(blocked_nodes)
        if self.prev_blocked_nodes!=[]: #otherwise intersection would be empty
            #nodes blocked now and also previously
            #print("blocked nodes ",blocked_nodes)
            #print("set blocked nodes ",set(blocked_nodes))
            #print("intersection ",set(blocked_nodes).intersection(set(transformed_blocked)))
            common_blocked_nodes = list(set(blocked_nodes).intersection(set(transformed_blocked)))
        self.blocked_nodes = copy.copy(common_blocked_nodes)
        self.prev_blocked_nodes = copy.copy(self.blocked_nodes)
        #=================================================================================================


        #print(nav_map_t[16,16]) #nav_map_t is a numpy array (16,16) is the ego-center where the agent is standing
        good_sample = True
        if bad_target:
            good_sample = False
        if contains_target==False:
            if self.room_obj!="nav_space":
                good_sample = False

        return good_sample


    def add_labels(self):
        ##############################################################################
        #pass target labels to the graph convolutional neural network class
        target_nodes = []
        labls = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                labl = np.zeros((self.n_classes,))
                #for p in possib:
                v = self.nav_map_t[i,j]
                
                if (i,j) in self.blocked_nodes: #dont need to penalize GCN for grids which it cannot see because of blockage by walls/obstacles
                    pass
                else:
                    targ_node_idx = (i)*self.grid_size+ (j) # earlier instead of i and j was (i-1) and (j-1) / why?
                    target_nodes.append(targ_node_idx)
                    labls.append(int(v))
                

                '''
                #without the blocked nodes condition
                targ_node_idx = (i)*self.grid_size+ (j) # earlier instead of i and j was (i-1) and (j-1) / why?
                target_nodes.append(targ_node_idx)
                labls.append(int(v))
                '''
                #print("node targets ",labl)

        #gets the grid graph ready with the target values (ground truth for training)
        self.ng.add_node_targets(node_list = target_nodes,labels = labls) 
        #print("Added graph convolution targets ")

    def add_inputs(self):
        ##############################################################################
        #prepare input for GCN
        try:
            #camera_proj = np.load('data/inputs/bev_'+repr(self.room)+'_'+repr(self.pos)+'.npy',allow_pickle = 'TRUE').item()
            camera_proj = np.load('data/inputs/bev_'+repr(self.room)+'_'+repr([self.pos[1],self.pos[0]])+'.npy',allow_pickle = 'TRUE').item()
        except:
            print("WARNING ! Unable to load approximate camera projection at position ",self.pos)
            print("you can run the data data_consistency_check function to make sure ")
            print("possible x,z ",(self.pos[1]*0.25+ self.o_grids['min_pos']['mx']), (self.pos[0]*0.25+ self.o_grids['min_pos']['mz']))
            return False

        #camera_proj = np.load('data/inputs/bev_'+repr(self.room)+'_'+repr(self.pos)+'.npy',allow_pickle = 'TRUE').item()
        camera_proj = np.load('data/inputs/bev_'+repr(self.room)+'_'+repr([self.pos[1],self.pos[0]])+'.npy',allow_pickle = 'TRUE').item()
        nav_map = proj.input_navigation_map(camera_proj, self.room_obj, grid_size = params.grid_size, 
                                            unk_id = params.semantic_classes['unk'],
                                            flr_id = params.semantic_classes['flr'], 
                                            tar_id = params.semantic_classes['tar'], 
                                            obs_id = params.semantic_classes['obs'])
        #nav_map is an array of params.grid_size x params.grid_size x 4 (4 classes unk,flr,tar and obs)| contains values between 0 and 1

        #proj.prettyprint(nav_map,argmax = True) #it cant print without argmax because each element of nav_map is 4 dimensional
        self.nav_map = nav_map #used later for visualization

        can_see_target = proj.prettyprint(nav_map,argmax = True, show = False) #no need to print the projection map everytime 
        
        if can_see_target:
            pass
        else:
            if self.room_obj=="nav_space":
                pass #no target is projected if agent is just interested to map the navigable space
            else:
                print("returning false")
                return False
        ##############################################################################
        #pass input data to the graph convolutional neural network class
        self.ng.clear_input()

        for i in range(self.grid_size): #16 is added if originally proj maps were made using grid size 64, but now calling grid size 32
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
        #print("successfully added inputs to GCN")
        return True #is always going to be true if it made this far

    def train(self, motion):
        ##############################################################################
        #train the GCN once
        self.ng.train(motion, epochs = 1, verbose = True)
        print("successfully trained GCN once")

    def data_visualize(self):
        #=================================
        #to test whether the correct ground truth labels are being sampled 
        print("Randomly chosen room ",self.room)
        print("randomly chosen position ",self.pos)
        print("randomly chosen target object ",self.room_obj)

        print("Showing ground truth map")
        #gtm.prettyprint(nav_map_t)
        proj.starviz(self.nav_map_t)
        print("Showing ground truth blocked nodes ")
        proj.starviz(self.nav_map_t, blocked_nodes = self.blocked_nodes)
        #=================================

        print("Showing inputs to GCN (argmaxes of each element in approximate projection) ")
        proj.prettyprint(self.nav_map,argmax = True) #it cant print without argmax because each element of nav_map is 4 dimensional
        proj.viz_belief_colors(self.nav_map)

        inp = input("Continue ? (y/n) (delete trainflag file to stop completely)")
        if inp=="y":
            pass
        else:
            sys.exit(0)


    def train_loop(self):
        e = 0
        open('trainflag', 'a').close() #a flagfile that keeps the code running unless its deleted externally
        while path.exists('trainflag'): #keep training as much as you want, to stop training delete this file
            
            #print("sample data boolean ",self.sample_data())
            if self.sample_data():
                pass
            else:
                #print("Sampled target map does not contain target element or bad sample(abort) ")
                self.motions = []
                continue
            
            self.add_labels()
            #print("add inputs boolean ",self.add_inputs())
            
            if self.add_inputs(): #input projection has atleast one grid where the target object has been mapped
                print("###################### Epoch ",e,"############################")
                if args.check:
                    self.data_visualize() #uncomment when no longer debugging

                self.train(self.motions[-1])
                e+=1
            else:
                #print("Sampled input map does not have atleast 1 projected target element (abort)")
                self.motions = []
                pass


if __name__ == '__main__':
    '''
    training notes:
    first train the spatial part using 's'
    then train the temporal part using 't' (previous spatial weights trained will be loaded)

    training style for temporal aggregation is a bit different- here LSTM is called once for each motion of the agent
    and then a backwards is called
    and then agent executes a random motion on top of initial motion
    LSTM is updated based on previous hiddens and again backward is called
    previous activations would get stored in LSTM hiddens, so each consecutive backward without a memory reset would chain all the previous motions of the agent
    this continues until maximum trajectory length is reached or target is no longer visible or end of room is reached
    '''

    choice = input("Train the spatial part or the temporal part ? (s/t) ")
    gct = graph_conv_trainer(trainpart = choice)

    #===========================
    #Uncomment if debugging
    '''
    #some changes for specific debugging visualization
    #run with --check args
    gct.rooms = [301]
    gct.good_objects = ["Bed"]#["nav_space"]
    #gct.data_consistency_check() #optional debugging for data consistency
    '''
    #Uncomment if debugging
    #===========================
    
    

    print("training once ")
    gct.train_loop()


