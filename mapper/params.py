grid_size = 33
node_neighborhood = 5
hop_connection = 1
concat_gridxy = True
node_embed_dim = 0

node_hidden_dim = 512
semantic_classes = {'unk':0,'flr':1,'tar':2,'obs':3}

graph_conv_layer_type = "simplest"

trajectory_data_location = '/alfred/data/json_2.1.0/train/*-'

trainrooms = [301,302,304,305,306,313,314,321,322,328,329]
trainobjects = ["Bed","Desk","Dresser","SideTable","Chair","ArmChair","DiningTable","Shelf","Safe","Sofa","nav_space"]

debug_viz = True
