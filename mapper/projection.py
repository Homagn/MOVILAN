import numpy as np
import cv2
import math
import copy 
import os
from os import path
import matplotlib.pyplot as plt

import params

#TRAIN_INPUT_LOC = "/home/hom/Desktop/ai2thor/mapping/graph_conv_data/"
#TRAIN_IMAGE_LOC = '/home/hom/Desktop/ai2thor/hutils/panorama_data/DFSdatanew/' #contains complete panorama of rgb,seg, and depth for rooms and locations

def matrix_rot(mat,grid_size,rot_h=0.0,rot_v=0.0):
    blank = np.zeros_like(mat)
    #c = int(grid_size/2.0)-1 #for even grid square shape
    c = int(grid_size/2.0) #for odd grid square shape
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j]!=0:
                x = i-c
                y = j-c
                #x1 = int(x*math.cos(rot_v)-y*math.sin(rot_v))
                #y1 = int(x*math.sin(rot_v)+y*math.cos(rot_v))

                nx = int(x*math.cos(rot_h)-y*math.sin(rot_h))
                ny = int(x*math.sin(rot_h)+y*math.cos(rot_h))
                try:
                    blank[nx+c,ny+c] = 1.0
                except:
                    #print("got out of screen")
                    pass
    return blank


def project(px,py,pz, seg, grid_size, rot_h = 0.0, rot_v = 0.0): #rot is the anticlockwise rotation from the reference coord sys

    def apply_rot(x,y,rot):
        drot = math.degrees(rot)

        nx = x*math.cos(rot)-y*math.sin(rot)
        ny = x*math.sin(rot)+y*math.cos(rot)

        return nx,ny
    #height of agent camera is 1.8 m (most probably)
    # here: https://arxiv.org/pdf/1712.05474
    points = []
    bev = np.zeros((grid_size,grid_size))
    #print("applying rotation ",math.degrees(rot_h),math.degrees(rot_v))

    for i in range(px.shape[0]):
        for j in range(px.shape[1]):
            if seg[i,j]!=0.0:
                #right handed x,y,z coordinate system
                #0.25m is the grid size of the projection map (also interpret as the degree of resolution we want)
                #0.49 is the maximum value of the depth normal
                #5.0 m is the maximum distance the depth camera can see
                x = ((px[i,j]/0.49)*5.0)/0.25 # specific values for ai2thor simulator obtained empirically by testing
                y = ((pz[i,j]/0.49)*5.0)/0.25  #a pesky -1 used here earlier (z coord in simulator is the apparent y coordinate in proj)
                z = ((py[i,j]/0.49)*5.0)/0.25

                #vertical head rotation (maybe dont need this adjustment)
                y,z = apply_rot(y,z, rot_v)

                x = int(x)
                y = int(y)

                points.append([x,y])
    #print("got points ",points)
    #c = int(grid_size/2.0)-1 #for even grid size
    c = int(grid_size/2.0) #for odd grid size
    for p in points:
        try:
            bev[p[0]+c,p[1]+c] += 1.0
            #bev[p[0],p[1]] = 1.0
        except:
            #print("could not append the point (too far away)")
            pass
    #bev[16:,16:] = 0.0
    return bev



def d2pcd(depth,stretch_x = 1.0, stretch_y = 1.0): #Apparently AI2THOR only provides pcd_z as a depth map
    #For each [i,j] pixel coordinate in the image, depth normals provide the z coordinate value (penetration) of that pixel
    #Here using the penetration and assumed camera parameters we are also finding the x and the y

    #source
    #https://elcharolin.wordpress.com/2017/09/06/transforming-a-depth-map-into-a-3d-point-cloud/
    #(Either pcd_x or pcd_y is not coming out correctly)
    
    #Assumed camera parameters
    thet = math.pi/2 #90 degree field of view (assuming the same FOV for both x and y)
    #thet = math.radians(15) #100 degree field of view (assuming the same FOV for both x and y)
    alpha_h = (math.pi-thet)/2 #30 degree angle of the first sample
    alpha_v = (2*math.pi)-thet/2

    #convert depth maps to point cloud data
    pcd_x = np.zeros_like(depth)
    pcd_y = np.zeros_like(depth)
    pcd_z = copy.copy(depth)
    #Getting x axis and y axis depth maps
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            d_i = depth[i,j]
            # For x depth_map
            gamma_i = alpha_h + j*(thet/depth.shape[1])
            if j>int(depth.shape[1]/2):
                gamma_i = math.pi - gamma_i #tan gets negative for values greater than pi remeber the all sin tan cos rule ?
            pcd_x[i,j] = d_i/math.tan(gamma_i)
            
            #if you want to visualize the pcd_x ->comment out this if portion
            if j<int(depth.shape[1]/2):
                pcd_x[i,j] = -pcd_x[i,j] #setting 0 as the original and getting vector values distances
            

            # For y depth map
            gamma_i = alpha_v + i*(thet/depth.shape[0])
            #if i>int(depth.shape[0]/2):
                #gamma_i = math.pi - gamma_i #tan gets negative for values greater than pi remeber the all sin tan cos rule ?
            pcd_y[i,j] = -d_i*math.tan(gamma_i)

            #if you DONT want to visualize the pcd_y ->comment out this if portion
            #if i>int(depth.shape[0]/2):
                #pcd_y[i,j] = -pcd_y[i,j] #setting 0 as the original and getting vector values distances

            pcd_x[i,j] = pcd_x[i,j]*stretch_x
            pcd_y[i,j] = pcd_y[i,j]*stretch_y

    return pcd_x,pcd_y,pcd_z



def bevmap(panorama, grid_size = 32, rn = 301, debug = False): #ort does not matter for now

    #Takes in rgb, segmentation and depth panoramas
    #For each different segmented object class in the segmentation panorama (masks), project them using depth panoramas
    #collect projection of every different segmentation object classes in a bev dictionary
    #projections are simple 2d matrices - 1 if the object is present in the gri, 0 otherwise
    #input panorama images are created by rotating the agent by 50 degrees for 7 times for each horizontal image patch
    #and for each horizontal image patch also tilt the agent camera up and down by 30 degrees to get vertical patch

    if debug:
        panorama = np.load('test.npz')
    else:
        #print("projection map for this room is not presaved ")
        pass

    try: #(seg_names provide the excact object ids referenced by ai2thor)
        seg_names = panorama["seg_names"].tolist()
    except:
        seg_names = panorama["seg_names"]

    masks = panorama["masks"]
    masks_named = {}
    m_i = panorama["seg"]
    c_i = panorama["rgb"]
    d_i = panorama["depth"]

    for s in range(len(seg_names)):
        masks_named[seg_names[s]] = panorama["masks"][s]

    #print("Got all seg_names from the panorama ",seg_names)
    #seg_names = ['Bed|-00.64|+00.00|+00.87']

    bev = {i:np.zeros((grid_size,grid_size)) for i in seg_names} #seg_names are names of all the segmented regions in the instance segmented image

    #some control parameters
    sxy = 100 #300
    rot_h_step = 50
    rot_v_step = 30
    vertical_order = [0,1,2]
    bundle_vertical = False
    bundle_hor = False

    
    # This is the main chunk of code that does the mapping of objects using segmentation and depth data
    for k in vertical_order:
        bev2 = {i:np.zeros((grid_size,grid_size)) for i in seg_names}
        for i in range(7): #5 #horizontal order
            #print("inside loop now")
            a = d_i.shape[1]-sxy*(i+1)
            b = d_i.shape[1]-sxy*(i)

            g = d_i.shape[0]-sxy*(k+1)
            h = d_i.shape[1]-sxy*(k)

            px,py,pz = d2pcd(d_i[g:h,a:b]/255.0) #1.2 earlier- 0.416 both

            for idx in bev.keys():
                #camera head rotated 50 every horizontal and 30 every vertical
                bev1 =  project(px,py,pz, masks_named[idx][g:h,a:b], grid_size, rot_v = -math.radians(rot_v_step*(2-k)+(rot_v_step/2))) #rot_v = -math.radians(30*(2-k)+15)
                bev1 = matrix_rot(bev1,grid_size,rot_h = math.radians(rot_h_step*(i))) #rot_h = math.radians(60*(i+1)-30)
                #if np.argwhere(bev2!=0).shape[0]==0 or obj=="Floor" or obj=="Wall" or bundle_hor==False: #no bundling for floors and walls
                bev2[idx] = bev2[idx]+bev1 #
        
        for idx in bev.keys():
            #if np.argwhere(bev[idx]!=0).shape[0]==0 or obj=="Floor" or obj=="Wall" or bundle_vertical == False:
            bev[idx] = bev[idx]+bev2[idx] #vertically aggregate horizontal strips

    return bev



def input_navigation_map(bev, target, grid_size = 32, unk_id = 0,flr_id = 1, tar_id = 2, obs_id = 3):
    #navigation map as perceived by projection from input vision sensors
    nav_map = np.zeros((grid_size,grid_size,4))
    nav_map_hits = np.zeros((grid_size,grid_size,4))

    for i in range(grid_size):
        for j in range(grid_size):
            
            for k in bev.keys():
                if 'Floor' in k or 'Rug' in k:
                    nav_map[i,j,flr_id] += bev[k][i,j]
                    nav_map_hits[i,j,flr_id] += 1

                if target in k:
                    nav_map[i,j,tar_id] += bev[k][i,j]
                    nav_map_hits[i,j,tar_id] += 1


                else:
                    nav_map[i,j,obs_id] += bev[k][i,j]
                    nav_map_hits[i,j,obs_id] += 1

            
            if np.sum([bev[k][i,j] for k in bev.keys()])==0:
                #print("i ",i," j ",j)
                
                nav_map[i,j,unk_id] = 1.0
                nav_map_hits[i,j,unk_id] += 1
                
            
    
    #normalization across number of unique occurances for that object type
    #This is for example in cases like a stack of drawers at a single grid location 
    #Also to normalize the obstacle class as everything other than target or floor is obstacle

    for i in range(grid_size):
        for j in range(grid_size):
            if nav_map_hits[i,j,flr_id]!=0:
                nav_map[i,j,flr_id] = nav_map[i,j,flr_id]/nav_map_hits[i,j,flr_id]
            if nav_map_hits[i,j,tar_id]!=0:
                nav_map[i,j,tar_id] = nav_map[i,j,tar_id]/nav_map_hits[i,j,tar_id]
            if nav_map_hits[i,j,obs_id]!=0:
                nav_map[i,j,obs_id] = nav_map[i,j,obs_id]/nav_map_hits[i,j,obs_id]
            #nav_map[i,j,unk_id] = nav_map[i,j,unk_id]/nav_map_hits[i,j,unk_id]

    #normalization across classes
    for i in range(grid_size):
        for j in range(grid_size):
            if np.sum(nav_map[i,j,:])!=0.0:
                nav_map[i,j,:] = nav_map[i,j,:]/np.sum(nav_map[i,j,:])


    #prettyprint(nav_map,argmax = True)
    return nav_map


#######################################################################################################
# All different visualizations for the created projection maps

def prettyprint(mat,argmax = False, show=True, symbolic = True):
    
    all_vals = []
    symbols = {params.semantic_classes['unk']:'|',
                params.semantic_classes['flr']:'-', 
                params.semantic_classes['tar']:'*', 
                params.semantic_classes['obs']:'|'}

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            d = 0
            if argmax:
                d = np.argmax(mat[i,j,:])
                all_vals.append(d)
                #d = np.max(mat[i,j,:])
            else:
                d = repr(int(mat[i,j]))
            if show:
                if symbolic and argmax:
                    d = symbols[d]
                print(d,end = '')
                print(" ",end = '')
        if show:
            print(" ")
    if params.semantic_classes['tar'] in all_vals:
        return True
    else:
        return False


def starviz(labeled_grid, blocked_nodes = []):
    
    for i in range(labeled_grid.shape[0]):
        for j in range(labeled_grid.shape[1]):
            if i==int(labeled_grid.shape[0]/2) and j==int(labeled_grid.shape[1]/2):
                print('A', end = '') #these nodes are probably not visible because cut off by obstacles
                print(" ",end = '')
                continue
            if (i,j) in blocked_nodes:
                print('?', end = '') #these nodes are probably not visible because cut off by obstacles
                print(" ",end = '')
                continue
            if labeled_grid[i,j]==params.semantic_classes['tar']:
                print('*', end = '')
            if labeled_grid[i,j]==params.semantic_classes['flr']:
                print('-', end = '')
            if labeled_grid[i,j]==params.semantic_classes['unk'] or labeled_grid[i,j]==params.semantic_classes['obs']:
                print('|', end = '')

            #print(repr(labeled_grid[i,j]), end = '')
            print(" ",end = '')
        print(" ")

def displaymap(bev, obj):
    mobj = ""
    for o in bev.keys():
        if obj+'|' in o:
            print("Found ",o," in the projection maps")
            mobj = o
    
    prettyprint(bev[mobj])


def viz_belief_colors(nav_map):
    #grid probabilities (p(grid = floor) , p(grid = obstacle), p(grid = target)) can be visualized as a mixture of colors (r,g,b)
    colors = []
    texts = []

    for i in range(nav_map.shape[0]):
        color = []
        text = []
        for j in range(nav_map.shape[1]):
            c = nav_map[i,j,:]
            #print(c)
            #if c[params.semantic_classes['obs']]==
            rgb_color = ((c[params.semantic_classes['obs']]+c[params.semantic_classes['unk']])*0.5,
                        c[params.semantic_classes['tar']]*1.0,
                        c[params.semantic_classes['flr']]*1.0 )
            color.append(rgb_color)
            text.append(i)
        colors.append(color)
        texts.append(text)

    
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=texts, cellColours=colors, loc='center')
    fig.tight_layout()
    plt.show()








