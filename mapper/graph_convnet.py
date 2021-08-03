import dgl
import dgl.function as fn
import networkx as nx
import torch as th
import numpy as np
import scipy.sparse as spp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



from dgl.nn.pytorch import GraphConv

import matplotlib.animation as animation
import sys
import itertools
import traceback
from collections import deque
import params

'''
working:
the panorama image obtained by the agent is projected into a BEV projection using basic geometric technique (see projection.py)
making up a 33x33x4 grid 
each grid stores a vector of (4,)
denoting the 4 probabilities that the location is either a floor/obstacle/unknown/target object - summing up to 1

now a special modification is applied to each grid node and on top of storing the information of its own node, the (4,) vector
it also stores the information of all the nodes in the nxn neighborhood around it
so if the edge length of the neighborhood square is 5, each node now stores a vector of (4x5x5,) information

Now a graph topology is defined
the graph topology is a densely connected one
each grid is connected to 8 neighbors all around it passing it (100,) vector information to all the 8 neighbors simultaneously in the propagation step

depending upon the number of GCN layers (depth of the network) the features of features get aggregated (at each depth level) 
multiple times leading to propagation of information across far away nodes (see GCN algorithm)
this classic bunch of GCN layers we call spatial in the code

Now after the features are obtained following the final global update step (last layer of the GCN)
we have this 33x33x4 matrix back again. assuming the agent is located at the center of this 33x33 grid, 
we calculate the relative x,y locations of each grid element of current ego map(33x33 grid) 
from the center of the persistent grid that considers entire agent trajectory when agent started moving from the center of persistent grid (see last paragraph)
and concatenate those back to the 33x33x4 matrix as additional information channels
now we have 33x33x6 matrix

Now we divide this 33x33x6 grid into nine 11x11x6 grids 
each of this 11x11x6 grid is flattened into (11x11x6,) array and passed as input to the lstm
the lstm input has a batch size of 9 accordingly
we call this lstm forward operation as the temporal aggregation (TAN) layer

this TAN layer has a persistent grid of 100x100x2 initialized assuming agent is at the center at the start of time
each grid location stores the distance of that grid from the center grid -real values in range from [0,0] to [1,1] we call node position embeddings
upon receiving input from spatial layer, the 33x33x4 matrix is concatenated to the positional embeddings of this 100x100x2 grid
until the memory of the TAN is reset
the 33x33x4 values output of the LSTM is concatenated with certain location embeddings of the 100x100x2 persistent map 
based on perceived agent location wrt persistent map (how much has it moved from the center since start of time)
each time this information is encoded into the self.hiddens when the lstm is called forward one time step
next time lstm output uses this past information from self.hiddens
effectively aggregating (filtering) the agent ego map over time 

'''
class edgeNN(nn.Module): ##NN that sits on each edge of the graph that transfers info from node to node 
    def __init__(self, in_feats, out_feats):
        super(edgeNN, self).__init__()
        #self.conv1 = GraphConv(in_feats, hidden_size)
        #self.conv2 = GraphConv(hidden_size, num_classes)

        #using custom graph conv layers 
        self.lin1 = nn.Linear(in_feats, in_feats)

    def forward(self, inputs):
        h = self.lin1(inputs)
        h = torch.relu(h)
        h = self.lin1(h)
        h = torch.relu(h)
        return h

class GCNLayer(nn.Module): #my custom RGCN layer 
    def __init__(self, in_feats, out_feats, num_rel = 2, num_prop = 1, layer_type = "complex"): 
        #num_prop is the number of times you want to propagate the info (higher number means information can spread further away from initial node)
        #Also its seen that the starting loss is higher but convergence is faster if num_prop is higher
        #code runs much slower with increasing num_prop

        #num_rel is basically number of different types of edges in the graph
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
        self.global_update = nn.Sequential(
                                        nn.Linear(in_feats, int(in_feats/2)),
                                        nn.ReLU(),
                                        nn.Linear(int(in_feats/2), int(in_feats/2)),
                                        nn.ReLU(),
                                        nn.Linear(int(in_feats/2), out_feats)
                                        )

        self.lstm = nn.LSTM(in_feats, in_feats) #basically takes the input feature and maps it back to a same size hidden vector

        self.rel_ops = [edgeNN(in_feats, in_feats) for _ in range(num_rel)] #parameter sets for different types of edges


        self.num_prop = num_prop
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        self.gcn_msg = fn.copy_src(src='feat', out='msg') #replaced this with message_func
        self.gcn_reduce = fn.sum(msg='msg', out='feat') #sum produces a huge output value at each node when num_prop is large
        #self.gcn_reduce = fn.mean(msg='msg', out='feat') 
        self.layer_type = layer_type


    def forward(self, g, feature):

        self.hiddens =  [( torch.from_numpy(np.zeros((1 , 1, self.in_feats),dtype = np.float32)),
                                torch.from_numpy(np.zeros((1 , 1, self.in_feats),dtype = np.float32)) ) for _ in range(g.number_of_nodes())]

        
        def apply_func(nodes): #this is a node function operated after the sum reduce
            H = nodes.data['feat']
            outs = []
            for i in range(H.size(0)): # !!!! applying in loop significantly reduces speed
                h = H[i]

                out, hid = self.lstm(h.view(1, 1, -1), self.hiddens[i])
                self.hiddens[i] = hid #used iteratively by the next round of lstm

                outs.append(out)
            outs = torch.squeeze(torch.stack(outs,0))
            #print(outs)
            return {'feat': outs}
        
        

        '''
        #looks more concise but is still not the correct implementation
        def apply_func(nodes):
            inp = nodes.data['feat']
            hidden = ( torch.from_numpy(np.zeros((1 , 1, self.in_feats),dtype = np.float32)),
                                torch.from_numpy(np.zeros((1 , 1, self.in_feats),dtype = np.float32)) )
            outs, hidden = self.lstm(inp.view(g.number_of_nodes(), 1, -1), hidden)
            return {'feat': torch.squeeze(outs)}
        '''
        
        def message_func(edges):
            R = edges.data['rel_type']
            S = edges.src['feat']
            #routs = [self.rel_ops[int(r)](S[i]) for r in R]
            
            routs = []
            for i in range(R.size(0)): # !!!! applying in loop significantly reduces speed
                r = int(R[i])
                #r = 0
                rout = self.rel_ops[r](S[i])
                routs.append(rout)
            
            routs = torch.squeeze(torch.stack(routs,0)) 
            
            return {'msg': routs} #while info flows from one node to another, pass it through a linear NN function

        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['feat'] = feature
            #g.edata['rel_type'] = torch.from_numpy(np.zeros((g.number_of_edges(), 1),dtype = np.float32)) #all edges are initialized to be of the type '0'
            for _ in range(self.num_prop):
                #g.update_all(self.gcn_msg, self.gcn_reduce, apply_func)
                if self.layer_type == "complex":
                    g.update_all(message_func, self.gcn_reduce, apply_func) #lstm when propagating info + special edge relation NNs
                if self.layer_type == "simple":
                    g.update_all(message_func, self.gcn_reduce) #no lstm
                if self.layer_type == "simplest":
                    g.update_all(self.gcn_msg, self.gcn_reduce) #no lstm, no edge relation NNs
                #print("pass once !")
                #LSTM should be somewhere here
            h = g.ndata['feat'] #will provide features values of all the n**2 nodes, I think pytorch views it as (batch size, node value) for training the shared weight neural network

            #return self.linear(h) #here linear is the global update function/ I think LSTM can be added here to filter along time as well
            return self.global_update(h) #I think LSTM can be added here to filter along time as well

            #LSTM usage example here- https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728
            '''
            import torch
            import torch.nn as nn
            from torch.autograd import Variable

            time_steps = 10
            batch_size = 3
            in_size = 5
            classes_no = 7

            model = nn.LSTM(in_size, classes_no, 2)
            input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
            output_seq, _ = model(input_seq)
            last_output = output_seq[-1]

            loss = nn.CrossEntropyLoss()
            target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
            err = loss(last_output, target)
            err.backward()
            '''



class GCN(nn.Module): #graph convolution network defined using all of the above classes
    def __init__(self, in_feats, hidden_size, num_classes, num_rels = 2, layer_type = "complex"):
        super(GCN, self).__init__()

        self.in_feats = in_feats
        self.num_classes = num_classes

        #using custom graph conv layers 
        #NOTE- increasing number of layer directly increases the distance across which information is shared from one node to another
        #NOTE- each GCNLayer has its own self.hidden for storing hidden states for that layer of LSTM
        #NOTE- to implement filtering over time (combine projections across multiple time steps) use complex layertype that uses LSTM
        self.conv1 = GCNLayer(in_feats, hidden_size, num_rel = num_rels, layer_type = layer_type)
        self.conv2 = GCNLayer(hidden_size, hidden_size, num_rel = num_rels, layer_type = layer_type)
        self.conv3 = GCNLayer(hidden_size, int(hidden_size/2), num_rel = num_rels, layer_type = layer_type)
        self.conv33 = GCNLayer(int(hidden_size/2), int(hidden_size/2), num_rel = num_rels, layer_type = layer_type)
        self.conv4 = GCNLayer(int(hidden_size/2), num_classes, num_rel = num_rels, layer_type = layer_type)




    def forward(self, g, inputs): #agent rgb projection, agent motion in [x,y] in the current time step as a tensor

        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        #can do it again for repeated hidden layers
        h = self.conv2(g, h)
        h = torch.relu(h)

        #can do it again for repeated hidden layers
        h = self.conv2(g, h)
        h = torch.relu(h)

        #can do it again for repeated hidden layers
        h = self.conv3(g, h)
        h = torch.relu(h)

        
        h = self.conv33(g, h)
        h = torch.relu(h)
        

        h = self.conv4(g, h) 
        return h


#old TAN network    
class TAN(nn.Module): #Time aggregation network that takes the output of the graph convolution network to filter along time
    def __init__(self, device):
        super(TAN, self).__init__()

        #========================================================================
        #(define and calculate parameters)
        self.map_size = 100
        self.window_size = 11
        self.stride = 11
        self.lstm_bsize = 9 #can calculate that based on the unfolding operation
        self.ego_map_size = params.grid_size
        self.device = device
        
        self.concatxy = True
        
        if self.concatxy:
            self.grid_inp_size = len(params.semantic_classes)+2
        else:
            self.grid_inp_size = len(params.semantic_classes)

        self.grid_out_size = len(params.semantic_classes)
        self.lstm_inp_size = (self.window_size**2)*self.grid_inp_size
        self.lstm_out_size = (self.window_size**2)*len(params.semantic_classes)
        #========================================================================


        self.lstm = nn.LSTM(726,1000,2) #2 layer LSTM hidden size=1000
        '''
        #3 stacked LSTMS
        self.lstm1 = nn.LSTM(self.lstm_inp_size,500) 
        self.lstm2 = nn.LSTM(500,200)
        self.lstm3 = nn.LSTM(200,484) 
        '''
        
        '''
        self.lin_after_lstm = nn.Sequential(
                                        nn.Linear(484, 484),
                                        nn.ReLU(),
                                        nn.Linear(484, self.lstm_out_size),
                                        )
        '''

        self.lin_after_lstm = nn.Sequential(
                                        nn.Linear(1000, 750),
                                        nn.ReLU(),
                                        nn.Linear(750, 500),
                                        nn.ReLU(),
                                        nn.Linear(500, self.lstm_out_size)
                                        )
        
        #generally hiddens are initialized as 0s but lets try something different
        self.hidden = (0.5*torch.ones(2, self.lstm_bsize, 1000).to(self.device),0.5*torch.ones(2, self.lstm_bsize, 1000).to(self.device)) #2 because 2 stacked lstm layers

        '''
        self.hidden1 = (0.5*torch.ones(1, self.lstm_bsize, 500),0.5*torch.ones(1, self.lstm_bsize, 500)) 
        self.hidden2 = (0.5*torch.ones(1, self.lstm_bsize, 200),0.5*torch.ones(1, self.lstm_bsize, 200)) 
        self.hidden3 = (0.5*torch.ones(1, self.lstm_bsize, 484),0.5*torch.ones(1, self.lstm_bsize, 484))
        '''

        self.node_positions = self.node_pos_emb()
        self.c_disp = torch.tensor([0,0]) # current displacement of the agent with respect to the initial map center

    def node_pos_emb(self):
        a = np.zeros((self.map_size,self.map_size,2), dtype = np.float32)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i,j,:] = np.array([i/float(self.map_size),j/float(self.map_size)])
        return torch.tensor(a).to(self.device)

    def reset_memory(self):
        self.node_positions = self.node_pos_emb()
        self.c_disp = torch.tensor([0,0]) # current displacement of the agent with respect to the initial map center
        self.hidden = (0.5*torch.ones(2, self.lstm_bsize, 1000).to(self.device),0.5*torch.ones(2, self.lstm_bsize, 1000).to(self.device))
        '''
        self.hidden1 = (0.5*torch.ones(1, self.lstm_bsize, 500),0.5*torch.ones(1, self.lstm_bsize, 500)) 
        self.hidden2 = (0.5*torch.ones(1, self.lstm_bsize, 200),0.5*torch.ones(1, self.lstm_bsize, 200)) 
        self.hidden3 = (0.5*torch.ones(1, self.lstm_bsize, 484),0.5*torch.ones(1, self.lstm_bsize, 484))
        '''


    def forward(self, gcn_output, motion): #agent rgb projection, agent motion in [x,y] in the current time step as a tensor

        h = torch.relu(gcn_output) #gcn_output is 1089x4 dim. 1089=33x33 33 is grid size

        #==================================================
        #time aggregation part
        self.c_disp = self.c_disp.to(self.device)+motion #update agent position in persistent map
        print("c_disp ",self.c_disp)

        '''
        nodes_pos_emb = self.node_positions[int(self.map_size/2)-16+self.c_disp[0]: int(self.map_size/2)+16+self.c_disp[0]+1, 
                                            int(self.map_size/2)-16+self.c_disp[1]: int(self.map_size/2)+16+self.c_disp[1]+1, :]
        '''

        nodes_pos_emb = self.node_positions[int(self.map_size/2)-int(self.ego_map_size/2)+self.c_disp[0]: int(self.map_size/2)+int(self.ego_map_size/2)+self.c_disp[0]+1, 
                                            int(self.map_size/2)-int(self.ego_map_size/2)+self.c_disp[1]: int(self.map_size/2)+int(self.ego_map_size/2)+self.c_disp[1]+1, :]

        #============
        #(arrange patches from current input) also have them concat with node positions so that agent has knowledge of movement in the persistent map
        #h_r = torch.reshape(h,(33,33,4))
        h_r = torch.reshape(h,(self.ego_map_size,self.ego_map_size, len(params.semantic_classes)))

        #b = torch.reshape(h,(33,33,4))
        b = torch.cat((h_r,nodes_pos_emb), dim = 2) #should be now 33x33x6
        #size = 11 #extracting 11x11 patches off the map (LSTM views each of these patches independently)
        #stride = 11
        size = self.window_size #extracting 11x11 patches off the map (LSTM views each of these patches independently)
        stride = self.stride
        c = b.unfold(0,size,stride).unfold(1,size,stride) #extracting 11x11 patches off the 33x33 grid projection map
        #d = torch.reshape(c,(1,9,-1)) # 9,726
        print("c shape ",c.size())
        d = torch.reshape(c,(1,self.lstm_bsize,-1)) # 9,726
        #============


        #run time aggregation update using LSTM
        #some inspiration from :
        #--https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#model-c-3-hidden-layer
        #--https://discuss.pytorch.org/t/solved-training-a-simple-rnn/9055
        #--https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        #out, (h1,c1) = self.lstm(d, (h,c)) #https://discuss.pytorch.org/t/understanding-output-of-lstm/12320
        out, self.hidden = self.lstm(d, self.hidden) #https://discuss.pytorch.org/t/understanding-output-of-lstm/12320
        
        '''
        out1, self.hidden1 = self.lstm1(d, self.hidden1)
        out2, self.hidden2 = self.lstm2(out1, self.hidden2)
        out3, self.hidden3 = self.lstm3(out2, self.hidden3)
        '''

        #print("out after lstm shape ",out.size())
        out = torch.reshape(out, (self.lstm_bsize,-1)) #should be 9,1000 (batch_size = 9)
        #out3 = torch.reshape(out3, (self.lstm_bsize,-1)) #should be 9,1000 (batch_size = 9)

        '''
        out3 = self.lin_after_lstm(out3)
        out3 = torch.reshape(out3,(self.ego_map_size*self.ego_map_size, self.grid_out_size)) #convert back to original shape
        '''

        out = self.lin_after_lstm(out)
        out = torch.reshape(out,(self.ego_map_size*self.ego_map_size, self.grid_out_size)) #convert back to original shape
        #log softmax would be aplied to out afterwards before calculating loss

        return out




class nav_graph(object): #class that manages the grid based graph topology for navigation
    def __init__(self, grid_size = 33, embed_dim = 10, node_value_dim = 10, hop_connection = 1, 
                num_classes = 2, node_hidden_dim = 5, num_rels = 2, train_parts = ["spatial","temporal"], 
                layer_type = "complex", load_weights = False, viz_topology = False):
        
        self.n = grid_size
        
        self.embed_dim = embed_dim
        self.node_value_dim = node_value_dim
        self.num_classes = num_classes
        self.node_hidden_dim = node_hidden_dim
        self.hop_connection = hop_connection #how many nodes to hop while connecting to neighboring node including self 
        self.g = self.gen_grid(self.n, show = viz_topology)
        self.train_parts = train_parts

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device ",self.device)

        #======================================
        #(loading spatial model weights)
        try:
            self.spatial = GCN(self.embed_dim+self.node_value_dim, self.node_hidden_dim, self.num_classes, num_rels = num_rels, layer_type = layer_type)
            if load_weights:
                print("Asked to load model weights")
                #self.spatial.load_state_dict(torch.load("/home/hom/Desktop/ai2thor/mapping/nn_weights/weights_spatial.pth", map_location=torch.device('cpu'))) #providing an absolute path
                self.spatial.load_state_dict(torch.load("/ai2thor/mapper/nn_weights/weights_spatial.pth", map_location=torch.device('cpu')))
                print("Model loading a success !")
        except:
            print("ALAS! spatial model loading failed ")
            self.spatial = GCN(self.embed_dim+self.node_value_dim, self.node_hidden_dim, self.num_classes, num_rels = num_rels, layer_type = layer_type)
            traceback.print_exc()
        #======================================

        #======================================
        #(loading temporal model weights)
        try:
            self.temporal = TAN(self.device)
            if load_weights:
                print("Asked to load model weights")
                #self.temporal.load_state_dict(torch.load("/home/hom/Desktop/ai2thor/mapping/nn_weights/weights_temporal.pth", map_location=torch.device('cpu'))) #providing an absolute path
                self.temporal.load_state_dict(torch.load("/ai2thor/mapper/nn_weights/weights_temporal.pth", map_location=torch.device('cpu')))
                print("Model loading a success !")
        except:
            print("ALAS! temporal model loading failed ")
            self.temporal = TAN(self.device)
            traceback.print_exc()
        #======================================


        
        self.assign_feats() #assigns initial node values to the graph self.g
        self.inputs_surr = np.zeros((self.n,self.n,self.num_classes)) #used when adding back the inputs to outputs

        self.all_logits = deque(maxlen=10) #used later for visualization
        if self.embed_dim!=0:
            self.optimizer = torch.optim.Adam(itertools.chain(self.spatial.parameters(), self.embed.parameters()), lr=0.0003) # lr=0.001 earlier
        
        if self.embed_dim==0:
            if "spatial" not in train_parts and "temporal" in train_parts: #training just the time aggregation parameters
                for param in self.spatial.parameters():
                    param.requires_grad = False
                self.optimizer = torch.optim.Adam(self.temporal.parameters(), lr=0.0003) #lr=0.001 earlier
            if "temporal" not in train_parts and "spatial" in train_parts: #training just the spatial aggregation parameters -aka GCN
                for param in self.temporal.parameters():
                    param.requires_grad = False
                self.optimizer = torch.optim.Adam(self.spatial.parameters(), lr=0.0003) #lr=0.001 earlier
            if "temporal" in train_parts and "spatial" in train_parts: #training both
                self.optimizer = torch.optim.Adam(itertools.chain(self.spatial.parameters(), self.temporal.parameters()), lr=0.0003)


            

        
        self.spatial.to(self.device)
        self.temporal.to(self.device)
        #print("Checking if model was properly transferred to GPU ")
        #print(next(self.spatial.parameters().(self.device)))

        #=====================
        ## ( calculating the total number of parameters )- 
        ## https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model#:~:text=To%20get%20the%20parameter%20count,name%20and%20the%20parameter%20itself.&text=You%20can%20use%20torchsummary%20to%20do%20the%20same%20thing.
        #pytorch_total_params = sum(p.numel() for p in model.parameters())
        ##If you want to calculate only the trainable parameters:
        pytorch_total_params = sum(p.numel() for p in self.spatial.parameters() if p.requires_grad)
        print("Total number of trainable model spatial parameters ",pytorch_total_params)

        pytorch_total_params = sum(p.numel() for p in self.temporal.parameters() if p.requires_grad)
        print("Total number of trainable model temporal parameters ",pytorch_total_params)
        #sys.exit(0)

    def reset_memory(self):
        self.temporal.reset_memory()

    def gen_grid(self, n = 5, show = False): #grid will be a square of n
        #construction of the graph topology- this will dictate how each node aggregates information from neighbors
        g = dgl.DGLGraph()
        #print("Input n in gen_grid ",n)
        g.add_nodes(n**2)
        nav = np.arange(n**2).reshape((n,n))
        self.pos = {}

        for i in range(n**2):
            r = i+self.hop_connection
            l = i-self.hop_connection
            u = i-(self.hop_connection*n)
            d = i+(self.hop_connection*n)

            uld = i-(self.hop_connection*n)-self.hop_connection
            urd = i-(self.hop_connection*n)+self.hop_connection

            dld = i+(self.hop_connection*n)-self.hop_connection
            drd = i+(self.hop_connection*n)+self.hop_connection

            #======================
            #Add up,down.left,right connections
            try:
                if int(r/n)==int(i/n): #nodes are in the same row
                    g.add_edge(i, r) #edge to rightwards node
                    if show:
                        print(i,r)
            except:
                pass
            
            try:
                if int(l/n)==int(i/n): #nodes are in the same row:
                    g.add_edge(i, l) #edge to leftwards node
                    if show:
                        print(i,l)
            except:
                pass
            
            try:
                g.add_edge(i, d) #edge to downwards node
                if show:
                    print(i,d)
            except:
                pass
            
            try:
                g.add_edge(i, u) #edge to upwards node
                if show:
                    print(i,u)
            except:
                pass

            
            #======================
            #Add diagonal connections
            try:
                if int(u/n)==int(uld/n): #nodes are in the same row
                    g.add_edge(i, uld) #edge to upper left diagonal
                    if show:
                        print(i,uld)
            except:
                pass
            
            try:
                if int(u/n)==int(urd/n): #nodes are in the same row
                    g.add_edge(i, urd) #edge to upper right diagonal
                    if show:
                        print(i,urd)
            except:
                pass

            try:
                if int(d/n)==int(dld/n): #nodes are in the same row
                    g.add_edge(i, dld) #edge to down left diagonal
                    if show:
                        print(i,dld)
            except:
                pass

            try:
                if int(d/n)==int(drd/n): #nodes are in the same row
                    g.add_edge(i, drd) #edge to down right diagonal
                    if show:
                        print(i,drd)
            except:
                pass
            


            self.pos[i] = np.array([float(i%n),1.0-float(i/n)]) #enforces a grid layout to display the graph
        self.nx_G = g.to_networkx()
        if show:
            #nx has a very hard time drawing big graphs n>30 or so
            nx.draw(g.to_networkx(), self.pos,with_labels=True)
            plt.show()
        print('We have %d nodes.' % g.number_of_nodes())
        print('We have %d edges.' % g.number_of_edges())
        return g

    def assign_feats(self):
         
        self.node_values = torch.from_numpy(np.zeros((self.n**2, self.node_value_dim),dtype = np.float32))
        #self.node_values = torch.rand((self.n**2, self.node_value_dim))/10.0 #lets try with random node values instead of zero
        #https://discuss.pytorch.org/t/how-to-concatenate-word-embedding-with-one-hot-vector/31577/3
        
        if self.embed_dim==0:
            self.g.ndata['feat'] = self.node_values
        elif self.embed_dim !=0:
            self.embed = nn.Embedding(self.n**2, self.embed_dim) 
            self.concats = torch.cat((self.embed.weight, self.node_values), dim=1)
            self.g.ndata['feat'] = self.concats

        #edata is more like a selector function, so can specify with np array of discreet indices
        self.g.edata['rel_type'] = np.zeros((self.g.number_of_edges(), 1),dtype = np.float32) #all edges are initialized to be of the type '0'


    def add_input(self, node_num, information = np.array([0,1,0,0,1,0,0,0,0,0], dtype = np.float32)):
        self.inputs = self.g.ndata['feat']
        #trying to set a custom partial input information at node 0
        custom_input = information #let this encoding mean something about object locations at grid= node_num
        self.inputs[node_num][self.embed_dim:] = torch.from_numpy(custom_input)

    def clear_input(self): #resets all the node values to initial
        self.assign_feats()


    def add_surrogate_input(self, node_num, information = np.array([0,1,0,0,1,0,0,0,0,0], dtype = np.float32)):
        #self.inputs_surr = self.g.ndata['feat']
        #trying to set a custom partial input information at node 0
        custom_input = information #let this encoding mean something about object locations at grid= node_num
        self.inputs_surr[int(node_num/self.n), node_num%self.n,:] = custom_input

    def add_edge(self, source, target, relation = 1):
        self.g.edata['rel_type'][self.g.edge_id(source, target)] = relation  

    def add_node_targets(self, node_list = [0,33], labels = [0,1]):
        self.labeled_nodes = torch.tensor(node_list)  # only for nodes 0 and 33 we know some info
        self.labels = torch.tensor(labels)  # let node 0 and node 33 be reachable and we dont know anything else about others
        self.labeled_nodes_list = node_list
        self.labels_list = labels
    
    
    def train(self, motion, epochs = 1, verbose = False): #for multiple labels
        
        device_graph = self.g.to(self.device)
        #print("in train loop")
        for epoch in range(epochs):
            device_input = self.inputs.to(self.device)
            motion = torch.tensor(motion).to(self.device) #eg- motion = [0,0] or motion = [0,1]

            logits_spatial = self.spatial(device_graph, device_input)
            

            if "spatial" in self.train_parts and "temporal" not in self.train_parts:
                logits = logits_spatial
            if "temporal" in self.train_parts:
                logits_temporal = self.temporal(logits_spatial, motion)
                logits = logits_temporal


            #self.all_logits.append(logits.detach())
            logp = F.log_softmax(logits, 1)

            loss = F.nll_loss(logp[self.labeled_nodes].to(self.device), self.labels.to(self.device))


            self.optimizer.zero_grad()
            loss.backward(retain_graph = True) #for the original code it works even without retain_grad = True !
            #loss.backward() #for the original code it works even without retain_grad = True !
            self.optimizer.step()
            if verbose:
                print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
            print("saving model ")
            if "spatial" in self.train_parts:
                torch.save(self.spatial.state_dict(), "nn_mapper_spatial.pth")
            if "temporal" in self.train_parts:
                torch.save(self.temporal.state_dict(), "nn_mapper_temporal.pth")


    #def test_once(self, view_true_labels = False, highlight_index = -1):
    def test_once(self, motion):
        motion = torch.tensor(motion).to(self.device) #eg- motion = [0,0] or motion = [0,1]
        device_graph = self.g.to(self.device)
        device_input = self.inputs.to(self.device)

        #logits_spatial = self.spatial(self.g, self.inputs)
        logits_spatial = self.spatial(device_graph, device_input)

        #logits_spatial = self.spatial(self.g, self.inputs)
        if "spatial" in self.train_parts and "temporal" not in self.train_parts:
            logits = logits_spatial
        if "temporal" in self.train_parts:
            logits_temporal = self.temporal(logits_spatial, motion)
            logits = logits_temporal

        logits = F.log_softmax(logits,1)
        #print("logits ",logits)
        self.all_logits = [logits.detach().cpu()]
        
        
        
    def show_node_values(self):
        logits = self.spatial(self.g, self.inputs)
        print(logits.detach())
    
    def vis_matrix(self, neighborhood = 16):
        import math

        center = int(self.n/2)
        #entire_mat = self.all_logits[-1].numpy() #+ self.inputs_surr[int(v/self.n),v%self.n,:]
        entire_mat = self.all_logits[-1].numpy() 

        entire_mat = entire_mat.reshape((self.n,self.n,self.num_classes)) 
        #mat = entire_mat[center-neighborhood:center+neighborhood,center-neighborhood:center+neighborhood,:]
        mat = entire_mat

        #mat_c = np.zeros((mat.shape[0],mat.shape[1]))
        #mat_c[neighborhood,neighborhood] = -1

        infer_proj = 0.0*np.ones((mat.shape[0],mat.shape[1]))
        '''
        infer_proj is the returned estimated map of neighborhood x neighborhood 
        each cell contains 10.0 is that contains the target object of interest
        each cell contains 1.0 if that contains the floor (navigable space)
        other wise the cell contains 0.0 if that grid has an obstacle or is unmappable (beyond wall)
        '''
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                labl = mat[i,j,:]
                surr = self.inputs_surr[i,j,:] # option to add back the raw input projection, by default its an array of 0s

                d = '|' #by default infer_proj[i,j] = 0.0
                
                #print(labl)
                #if np.argmax(labl)==params.semantic_classes['flr'] or np.argmax(surr)==params.semantic_classes['flr']:
                if np.argmax(labl)==params.semantic_classes['flr']:
                    d = 'O'
                    #infer_proj[i,j] = 1.0
                    infer_proj[i,j] = params.semantic_classes['flr']

                if np.argmax(labl)==params.semantic_classes['tar']:
                    d = '*'
                    #infer_proj[i,j] = 10.0
                    infer_proj[i,j] = params.semantic_classes['tar']
                    #removes the risk that on object is predicted to exist if all 3x3 surrounding grid is floor (object code 1)
                    #because inference is based on sum of beliefs over all neighboring grids 
                    #the target object code has to be greater than atleast 3x3x1=9, so set it as 10

                if np.argmax(labl)==params.semantic_classes['obs']:
                    infer_proj[i,j] = params.semantic_classes['obs']

                if np.argmax(labl)==params.semantic_classes['unk']:
                    infer_proj[i,j] = params.semantic_classes['unk']

                if i==int(mat.shape[0]/2) and j==int(mat.shape[1]/2):
                    d = 'A'

                
                #print(d,end = '') #DEBUGGING
                #print(" ",end = '') #DEBUGGING
            #print(" ")

        return infer_proj



if __name__ == '__main__':
    ng = nav_graph( grid_size = 5, 
                    embed_dim = 0, 
                    node_value_dim = 5, 
                    hop_connection = 1, #how many nodes to hop while connecting to neighboring node including self 
                    num_classes = 4, 
                    node_hidden_dim = 10,  
                    layer_type = "simplest",
                    load_weights = False,
                    viz_topology = True)
    
    #ng.gen_grid(show = True)
