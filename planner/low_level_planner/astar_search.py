import os
import sys
import copy
import numpy as np
import random

os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))
from planner.low_level_planner import navigation_signatures as ns



from collections import defaultdict

class Graph():
    #source - http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

    def dijsktra(self, initial, end):
        graph = self
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        
        while current_node != end:
            visited.add(current_node)
            destinations = graph.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = graph.weights[(current_node, next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return "Route Not Possible", []
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        
        # Work back through destinations in shortest path
        path = []
        max_weight = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            max_weight.append(shortest_paths[current_node][1]) #appending the weight to the current node
            current_node = next_node
        # Reverse path
        path = path[::-1]
        return path, max(max_weight)


def set_graph(nav):
    g = Graph()
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
    return g

def mark_movement(nav, si, sj, rn, env, mark = 0): #perceive obstacle or navigable terrain using collision sensing (extra correction on top of mapping)
    if rn=="MoveAhead": #cannot move forward
        if env.get_rotation()==0: #facing north 
            nav[si,sj+1] = mark #obstacle marked 
        if env.get_rotation()==270: #facing west 
            nav[si-1,sj] = mark #obstacle marked 
        if env.get_rotation()==180: #facing east
            nav[si,sj-1] = mark #obstacle marked
        if env.get_rotation()==90: #facing south 
            nav[si+1,sj] = mark #obstacle marked 

    if rn=="MoveRight": #cannot move forward
        if env.get_rotation()==0: #facing north 
            nav[si+1,sj] = mark #obstacle marked 
        if env.get_rotation()==270: #facing west 
            nav[si,sj+1] = mark #obstacle marked 
        if env.get_rotation()==180: #facing east
             nav[si+1,sj] = mark #obstacle marked
        if env.get_rotation()==90: #facing south
            nav[si,sj-1] = mark #obstacle marked

    if rn=="MoveLeft": #cannot move forward
        if env.get_rotation()==0: #facing north 
            nav[si-1,sj] = mark #obstacle marked 
        if env.get_rotation()==270: #facing west 
            nav[si,sj-1] = mark #obstacle marked 
        if env.get_rotation()==180: #facing east
             nav[si+1,sj] = mark #obstacle marked
        if env.get_rotation()==90: #facing south 
            nav[si,sj+1] = mark #obstacle marked

    if rn=="MoveBack": #cannot move forward
        if env.get_rotation()==0: #facing north 
            nav[si,sj-1] = mark #obstacle marked 
        if env.get_rotation()==270: #facing west 
            nav[si+1,sj] = mark #obstacle marked 
        if env.get_rotation()==180: #facing east 
             nav[si,sj+1] = mark #obstacle marked
        if env.get_rotation()==90: #facing south
            nav[si-1,sj] = mark #obstacle marked
    return nav

def Random_nudge(env,nav,si,sj):
    random_nudge = ["MoveAhead","MoveBack","MoveLeft","MoveRight"]
    rn = random.choice(random_nudge)

    print("Selected random nudge ",rn)
    env.step(dict({"action": rn, "moveMagnitude" : 0.25}))
    

    if env.actuator_success()==False:
        '''
        #visualization in case of collision
        print("Nav ")
        for i in range(nav.shape[0]):
            for j in range(nav.shape[1]):
                d = '-'
                if nav[i,j]==0:
                    d = '|'
                if nav[i,j]==1:
                    d = 'O'
                if i==int(nav.shape[0]/2) and j==int(nav.shape[1]/2):
                    d = 'A'
                print(d,end = '')
                print(" ",end = '')
            print(" ")
        '''

        #nav = mark_movement(nav, si, sj, rn, event, mark = 0)
        nav = mark_movement(nav, si, sj, rn, env, mark = 0)
    
    if env.actuator_success():
        #rollback
        rb = ""
        if rn=="MoveAhead":
            rb = "MoveBack"
        if rn=="MoveBack":
            rb = "MoveAhead"
        if rn=="MoveRight":
            rb = "MoveLeft"
        if rn=="MoveLeft":
            rb = "MoveRight"
        env.step(dict({"action": rb, "moveMagnitude" : 0.25}))

        #nav = mark_movement(nav, si, sj, rn, event, mark = 1) #first rollback then mark
        nav = mark_movement(nav, si, sj, rn, env, mark = 1) #first rollback then mark
        
    return env,nav

def search(nav, facing_positions, env, tar_object, localize_params, notarget = True, numtries = 0, other_side = 0, recursion_stacks = ""): #egocentric agent map, thus source is always the center of the labeled_grid
    
    

    print("astar_search.py -> search")
    print("Trying this for ",numtries," time")
    
    g = set_graph(nav)
    
    
    source = repr(int(nav.shape[0]/2))+'_'+repr(int(nav.shape[1]/2))# source is always the center because ego centric

    if nav[int(nav.shape[0]/2), int(nav.shape[1]/2)]!=0 and nav[int(nav.shape[0]/2), int(nav.shape[1]/2)]!=1:
        print("You are already standing in the target in some orientation.. no need for searching path")
        #sys.exit(0)

    
    face_dir_rots = [0,270,180,90]

    costs = []
    paths = []
    o_facing_positions = copy.copy(facing_positions)
    o_c = [int((facing_positions[0][0]+facing_positions[1][0]+facing_positions[2][0]+facing_positions[3][0])/4),
            int((facing_positions[0][1]+facing_positions[1][1]+facing_positions[2][1]+facing_positions[3][1])/4)]
    
    #c2c_dist = (int(nav.shape[0]/2)-o_c[0])**2 + (int(nav.shape[1]/2)-o_c[1])**2
    c2c_dist = (o_c[0])**2 + (o_c[1])**2
    print("oc ",o_c)
    print("Grid c2c dist between agent and object ",c2c_dist)

    for i in range(4): #each object can be possibly faced from four different directions
        target = repr(int(nav.shape[0]/2)+facing_positions[i][0])+'_'+repr(int(nav.shape[1]/2)+facing_positions[i][1])
        path ,total_cost = g.dijsktra(source,target)
        paths.append(path)
        if path=="Route Not Possible":
            if i==0: #n facing position 
                facing_positions[i][1]-=1
            if i==1: #west facing position
                facing_positions[i][0]+=1
            if i==2: #south facing position
                facing_positions[i][1]+=1
            if i==3:#east facing position
                facing_positions[i][0]-=1

            total_cost = 10000
        costs.append(total_cost)

    #face_dir = np.argmin(np.array(costs)) #0 for north, 2 for south, lets try to get to the desk in room 301
    #path = paths[face_dir]
    face_dir = np.argsort(np.array(costs)) #0 for north, 2 for south, sorts in ascending order
    
    if costs.count(10000)>=3:
        print("Object can be reached from only 1 side ")
        face_dir = face_dir[0]
        path = paths[face_dir]
    else:
        if other_side==0: #by default go to the side of the object which is closest to the agent
            face_dir = face_dir[0]
        elif other_side==1:
            face_dir = face_dir[1]
        elif other_side==2:
            face_dir = face_dir[2]
        path = paths[face_dir]

    #print("got path ",path)

    if path=="Route Not Possible": #could not find a path to face the object in any direction 
        print("Could not find a path !, trying search with more relaxed conditions")
        if recursion_stacks == "inf": #This is the case when hallucinating a position towards the edge of the map
            if numtries<20:
                si,sj = int(nav.shape[0]/2), int(nav.shape[1]/2)
                env,nav = Random_nudge(env,nav,si,sj)
                return search(nav, facing_positions, env, tar_object, localize_params, numtries = numtries+1, other_side = other_side, recursion_stacks = recursion_stacks) #enter next recursion stage
        else:
            if numtries<5: #This is case when mapping may by fault increase or decrease the size of object too much
                si,sj = int(nav.shape[0]/2), int(nav.shape[1]/2)
                env,nav = Random_nudge(env,nav,si,sj)
                
                if notarget and c2c_dist>4: #center to center distance greater than 2 grids/ earlier code notarget = True means view_v_params = {}
                #if c2c_dist>4: #center to center distance greater than 2 grids/ earlier code notarget = True means view_v_params = {}
                    env = ns.unit_refinement(env, tar_object)
                    
                    nav, facing_positions = ns.occupancy_grid(tar_object, localize_params)
                    return search(nav, facing_positions, env, tar_object, localize_params, notarget = False, numtries = numtries+1, other_side = other_side) #enter next recursion stage
                    #return search(nav, facing_positions, env, tar_object, localize_params, numtries = numtries+1, other_side = other_side) #enter next recursion stage
        
        print("Failed to find a path to the object after several tries ")
        return env
        #return env, event
    return execute_path(env, nav, path, face_dir, tar_object, localize_params, facing_positions, other_side = other_side, numtries = numtries)
    

def execute_path(env, nav, path, face_dir, tar_object, localize_params, facing_positions, other_side = 0, numtries = 0):
    ######################   executing the path #######################
    move_dir = {(0,1):0,(-1,0):270,(0,-1):180,(1,0):90}
    face_dir_rots = [0,270,180,90]

    si = int(nav.shape[0]/2)
    sj = int(nav.shape[1]/2)

    for p in path[1:]: #the first one is the one the agent is standing in/ can integrate iterative planing if path[1:2] is taken
        ti = int(p.split('_')[0])
        tj = int(p.split('_')[1])
        
        tup = (ti-si,tj-sj)
        #print("tup ",tup)

        #x1,y1,z1 = env.get_position()

        for m in move_dir.keys(): #should match only once
            if tup==m:
                while env.get_rotation()!=move_dir[tup]:
                    if env.get_rotation()-90==move_dir[tup]:
                        env.step(dict({"action": "RotateLeft", 'forceAction': True}))

                    elif env.get_rotation()+90==move_dir[tup]:
                        env.step(dict({"action": "RotateRight", 'forceAction': True}))

                    else:
                        env.step(dict({"action": "RotateLeft", 'forceAction': True}))

                    #events = env.smooth_rotate(dict({"action": "RotateLeft"}), render_settings)
                env.step(dict({"action": "MoveAhead", "moveMagnitude" : 0.25}))
                not_stuck = env.actuator_success()


        #x2,y2,z2 = env.get_position()

        #if x1==x2 and z1==z2:
        if not_stuck==False:
            print("Need to replan ! agent possibly stuck in a corner")
            
            if numtries<10: #give 10 random nudges

                env,nav = Random_nudge(env,nav,si,sj)
                return search(nav, facing_positions, env, tar_object, localize_params, numtries = numtries+1, other_side = other_side) #enter next recursion stage
            
            #event = env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
            #return graph_search(nav, facing_positions, env, event, numtries = numtries+1, other_side = other_side) #enter next recursion stage

            print("Failed to find a path to the object after several tries ")
            return env

        si = copy.copy(ti)
        sj = copy.copy(tj)

    while env.get_rotation()!=face_dir_rots[face_dir]:
        env.step(dict({"action": "RotateLeft",'forceAction': True}))
        #events = env.smooth_rotate(dict({"action": "RotateLeft"}), render_settings)
    #one final step towards the object
    env.step(dict({"action": "MoveAhead", "moveMagnitude" : 0.25}))
    #mask_image = event.instance_segmentation_frame
    #face_touch(mask_image,event)    
    return env














if __name__ == '__main__':
    graph = Graph()
    edges = [
    ('X', 'A', 7),
    ('X', 'B', 2),
    ('X', 'C', 3),
    ('X', 'E', 4),
    ('A', 'B', 3),
    ('A', 'D', 4),
    ('B', 'D', 4),
    ('B', 'H', 5),
    ('C', 'L', 2),
    ('D', 'F', 1),
    ('F', 'H', 3),
    ('G', 'H', 2),
    ('G', 'Y', 2),
    ('I', 'J', 6),
    ('I', 'K', 4),
    ('I', 'L', 4),
    ('J', 'L', 1),
    ('K', 'Y', 5),
    ]

    for edge in edges:
        graph.add_edge(*edge)
    print(graph.dijsktra('X', 'Y'))
