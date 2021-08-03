import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision import models

from unet_ling import Unet5Contextual 
from unet_ling import data_loader

import cv2
import json

def lang2seg():
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print("Using device ",device)
    

    dat = data_loader('data/descriptions.json', prehashed = "prehash.npy", resize_factor = 224) #model was trained with image size 224x224
    w2i = dat.w2i #The word to index mapping that is required to get correct word embeddings
    #print("got w2i ",w2i)



    model = Unet5Contextual(w2i,bsize = 1, device = device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5,lr=0.00001)
    model.load_state_dict(torch.load("unet_weights.pth", map_location=torch.device('cpu')))

    model.to(device)

    #NOTE- descriptions must be a list of 5 sentences. Repititions allowed. Must use the vocabulary within w2i dictionary
    loc= "test_images/1.jpg"
    desc = ["Mug on Desk","Mug your right","Mug below Statue","Mug left most Mug","Mug closest Mug"]
    #desc = ["Mug on Desk","Mug","Mug below Statue","Mug left Mug","Mug closest Bowl"]


    #loc= "2.jpg"
    #desc = ["Bed","Bed","Bed","Bed","Bed"]
    #desc = ["Bed containing Laptop","Bed","Bed","Bed","Bed"]
    #desc = ["Pillow on Bed","Pillow","Pillow","Pillow","Pillow"]


    #loc= "3.jpg"
    #desc = ["Shelf","Shelf your right","Shelf above AlarmClock","Shelf","Shelf"]
    #desc = ["Bowl on Desk","Bowl your left","Bowl right Laptop","Bowl","Bowl"]

    #sentence descriptions
    #1- This line can describe whether an object is contained (or sitting on top of) in another object
    #2- This line can describe the general position of the object up/down/left/right wrt the agent
    #3- For an object f, this function describes positions of all visible objects g wrt it
    #4- If multiple objects of the same type (eg 2 cups on a table) are present describe group relative positions
    #5- If multiple objects of the same type (eg 2 cups on a table) are present describe group relative positions



    

    rgb = dat.load_image(loc) #takes care of any preprocessing including resizing
    rgb = rgb.to(device)

    optimizer.zero_grad()

    output = model(rgb,desc)

    print("Showing input image to the model, press esc to see prediction ")
    inputs = rgb.cpu().numpy().reshape((1,224,224,3))
    cv2.imshow("MODEL_INPUT",inputs[0])
    cv2.waitKey(0)

    seg_pred = F.sigmoid(output).detach().cpu().numpy().reshape((1,224,224))
    print("Model predicted segmentation image for the language description ",desc)
    cv2.imshow("PRED",seg_pred[0]*255.0)
    cv2.waitKey(0)






if __name__ == '__main__':
    lang2seg()
    