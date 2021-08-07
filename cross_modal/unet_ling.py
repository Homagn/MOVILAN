import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision import models

import copy
import json
import cv2
import numpy as np
import random

import sys

class data_loader(object):
    def __init__(self,json_file, prehashed = "", batch_size = 16, resize_factor = 1, focus_object = ""):
        self.json = json
        self.bsize = batch_size
        self.resize_factor = resize_factor
        self.focus_object = focus_object

        with open(json_file) as f:
            self.descriptions = json.load(f)

        self.seg_idxs = []
        self.rgb_idxs = []

        self.counter = 0

        if prehashed == "" : #these following syeps take a lot of time so prehash
            print("Collecting all sentences from provided json file ")
            self.all_sent = ""
            #expected nested structure of the loaded dictionary
            #dict-> room_no->task_no->grid_position->agent rotation->target object->list of 5 descriptions
            for r in self.descriptions.keys():
                for t in self.descriptions[r].keys():
                    for g in self.descriptions[r][t].keys():
                        for rot in self.descriptions[r][t][g].keys():
                            for obj in self.descriptions[r][t][g][rot].keys():
                                #print(self.descriptions[r][t][g][rot])

                                #obj = list(self.descriptions[r][t][g][rot].keys())[0]
                                g1,g2 = g.split('_')
                                self.seg_idxs.append(r+'_'+t+'_'+g1+','+g2+'_'+rot+'_'+obj)
                                self.rgb_idxs.append(r+'_'+t+'_'+g1+','+g2+'_'+rot)
                                
                                for s in self.descriptions[r][t][g][rot][obj]:
                                    self.all_sent = self.all_sent+s+' ' #results in a huge passage comprising of all descriptions

            self.w2i = self.word_mapping()

            hashed = {"w2i":self.w2i, "seg_idxs":self.seg_idxs, "rgb_idxs":self.rgb_idxs, "all_sent":self.all_sent}
            np.save('prehash.npy',hashed)

        else:
            hashload = np.load(prehashed,allow_pickle = 'TRUE').item()
            self.w2i = hashload["w2i"]
            self.seg_idxs = hashload["seg_idxs"]
            self.rgb_idxs = hashload["rgb_idxs"]
            self.all_sent = hashload["all_sent"]

    def word_mapping(self):
        print("mapping vocabulary")
        #all_words = "Pick up the big black cellphone on top of the white table near the lamp".split()
        all_words = self.all_sent[:-1].split() #last character of self.all_sent will be a space- ignore it
        vocab = set(all_words)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        #print(word_to_ix)
        return word_to_ix

    def next(self, position = -1): # A data iterator function
        if position!=-1:
            self.counter = position

        found = False
        if self.focus_object!="":
            while found==False:
                c = random.choice(list(range(len(self.rgb_idxs))))
                #print(self.seg_idxs[self.counter].split('_'))
                r,t,g,rot,obj = self.seg_idxs[c].split('_')
                g1,g2 = g.split(',')
                desc = self.descriptions[r][t][g1+'_'+g2][rot][obj]
                d1 = desc[0].split()
                if d1[0]!=focus_object:
                    continue #keep looping through until all the descriptions belonging to the focus object is gathered
                else:
                    found = True
                    self.counter = c


        print(self.seg_idxs[self.counter].split('_'))
        r,t,g,rot,obj = self.seg_idxs[self.counter].split('_')
        g1,g2 = g.split(',')
        desc = self.descriptions[r][t][g1+'_'+g2][rot][obj]
        rgb_name = self.rgb_idxs[self.counter]
        seg_name = self.seg_idxs[self.counter]

        print("got descriptions ")
        print(desc)

        print("RGB name ",rgb_name)
        image1 = cv2.imread('data/rgb/'+rgb_name+'.jpg')
        image1 = np.array(image1/255.0,dtype = np.float32)


        image2 = cv2.imread('data/seg/'+seg_name+'.jpg', 0)
        image2 = np.array(image2/255.0,dtype = np.float32)


        self.counter+=1
        if self.counter>len(self.seg_idxs)-1:
            self.counter = 0

        w=h=image1.shape[0] #square images of size 300
        image1 = torch.from_numpy(image1).view((1, 3, w, h))
        image2 = torch.from_numpy(image2).view((1, 1, w, h))
        return image1,image2,desc

    def load_image(self,loc):
        image1 = cv2.imread(loc)
        image1 = self.resize(image1)
        image1 = np.array(image1/255.0,dtype = np.float32)
        
        w=h=image1.shape[0]
        image1 = torch.from_numpy(image1).view((1, 3, w, h))
        return image1

    def resize(self,image):
        if self.resize_factor<=1:
            width = int(image.shape[1] * self.resize_factor)
            height = int(image.shape[0] * self.resize_factor)
        else:
            width = self.resize_factor
            height = self.resize_factor
        dsize = (width, height)
        return cv2.resize(image, dsize)


    def next_batch(self):
        positions = random.sample(list(range(len(self.rgb_idxs))), self.bsize)
        
        in_images = []
        out_images = []
        descriptions = []

        #for c in positions:
        while len(in_images)!=self.bsize:
            c = random.choice(list(range(len(self.rgb_idxs))))
            #print(self.seg_idxs[self.counter].split('_'))
            r,t,g,rot,obj = self.seg_idxs[c].split('_')
            g1,g2 = g.split(',')
            desc = self.descriptions[r][t][g1+'_'+g2][rot][obj]
            
            if self.focus_object!="":
                d1 = desc[0].split()
                if d1[0]!=focus_object:
                    continue #keep looping through until all the descriptions belonging to the focus object is gathered

            descriptions.append(desc) #should become a list of lists

            rgb_name = self.rgb_idxs[c]
            seg_name = self.seg_idxs[c]

            #print("got descriptions ")
            #print(desc)

            #print("RGB name ",rgb_name)
            image1 = cv2.imread('data/rgb/'+rgb_name+'.jpg')
            image1 = self.resize(image1) #resize according to resize factor
            image1 = np.array(image1/255.0,dtype = np.float32)
            in_images.append(image1)


            image2 = cv2.imread('data/seg/'+seg_name+'.jpg', 0)
            image2 = self.resize(image2) #resize according to resize factor
            image2 = np.array(image2/255.0,dtype = np.float32)
            out_images.append(image2)

        w=h=in_images[0].shape[0] #square images of size 300
        image1 = torch.from_numpy(np.stack(in_images,axis=0)).view((self.bsize, 3, w, h))
        image2 = torch.from_numpy(np.stack(out_images,axis=0)).view((self.bsize, 1, w, h))
        return image1,image2,descriptions



class BCELoss2d(nn.Module):
    
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

class soft_dice_loss(nn.Module):
    #copied from - https://www.jeremyjordan.me/semantic-segmentation/#:~:text=The%20most%20commonly%20used%20loss,one%2Dhot%20encoded%20target%20vector.
    def __init__(self):
        super(soft_dice_loss, self).__init__()
    def forward(self, y_true, y_pred):
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
        
        return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch

def dice_loss(pred, target, smooth = 1.):
    #ref- https://github.com/usuyama/pytorch-unet/blob/master/loss.py
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def seg_loss(pred, target, bce_weight=0.5):
    #ref- https://github.com/usuyama/pytorch-unet
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class Unet5Contextual(torch.nn.Module): #accepts 224x224 size image/ 3 channel input/ 1 channel output

    def __init__(self, word_to_idx, embedding_dim = 200, max_sent_len = 4, bsize = 1, device = 'cpu'):

        super(Unet5Contextual, self).__init__()

        #self.hcb = hc1*bsize
        self.device = device

        self.w2i = word_to_idx
        self.max_sent_len = max_sent_len
        self.embedding_dim = embedding_dim


        self.base_model = models.resnet18(pretrained=True) #load pretrained resnet model layers
        self.base_layers = list(self.base_model.children())



        self.embedding_size = embedding_dim*max_sent_len
        self.bsize = bsize #batch size

        self.emb_block_size = self.embedding_size

        self.embeddings = nn.Embedding(len(word_to_idx)+1, embedding_dim, padding_idx = len(word_to_idx))



        #trying to use resnet pretrained layers
        self.conv1 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.conv2 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.conv3 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.conv4 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.conv5 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        
        self.condense1 = nn.Sequential(nn.Conv2d(2*5, 16*2*5, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16*2*5, 512, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True) )
        
        self.condense2 = nn.Sequential(nn.Conv2d(4*5, 8*4*5, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8*4*5, 256, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True) )

        self.condense3 = nn.Sequential(nn.Conv2d(8*5, 4*8*5, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(4*8*5, 128, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True))
        

        self.condense4 = nn.Sequential(nn.Conv2d(16*5, 2*16*5, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(2*16*5, 64, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True))
        
        self.condense5 = nn.Sequential(nn.Conv2d(16*5, 2*16*5, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(2*16*5, 64, 1, stride=1, padding=0),
                                    nn.ReLU(inplace=True))
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)



        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1) #n_class = 1



        self.lang19 = nn.Sequential(nn.Linear(self.emb_block_size, 1000), nn.ReLU(inplace=True), nn.Linear(1000, 64 * 16), nn.ReLU(inplace=True))
        self.lang28 = nn.Sequential(nn.Linear(self.emb_block_size, 1000), nn.ReLU(inplace=True), nn.Linear(1000, 64 * 16), nn.ReLU(inplace=True))
        self.lang37 = nn.Sequential(nn.Linear(self.emb_block_size, 1000), nn.ReLU(inplace=True), nn.Linear(1000, 128 * 8), nn.ReLU(inplace=True))
        self.lang46 = nn.Sequential(nn.Linear(self.emb_block_size, 1000), nn.ReLU(inplace=True), nn.Linear(1000, 256 * 4), nn.ReLU(inplace=True))
        self.lang55 = nn.Sequential(nn.Linear(self.emb_block_size, 1000), nn.ReLU(inplace=True), nn.Linear(1000, 512 * 2), nn.ReLU(inplace=True))
        

    def embed_sent(self, sentence = "Pick"):
        word_to_ix = self.w2i
        test_sentence = sentence.split()
        vocab_idcs = [word_to_ix[w] for w in test_sentence]
        vidcs = copy.copy(vocab_idcs)
        #print("vocab idcs ",vidcs)
        #padding sentences to a length of 10
        for _ in range(self.max_sent_len-len(vocab_idcs)):
            vidcs.append(len(word_to_ix))
        #print("vidcs ",vidcs)

        context_idxs = torch.tensor(vidcs, dtype=torch.long)
        context_idxs = context_idxs.to(self.device)
        #embeddings = nn.Embedding(len(word_to_ix)+1, embedding_dim, padding_idx = len(word_to_ix))
        embeds = self.embeddings(context_idxs).view((1, -1))
        #print("got embeds ",embeds)
        return embeds

    def embed_batch_sent(self,sentences):
        word_to_ix = self.w2i
        embeds = []
        for i in range(5): #5 types of sentence descriptions of an object
            Vidcs = []
            for j in range(len(sentences)):
                test_sentence = sentences[j][i].split()
                vocab_idcs = [word_to_ix[w] for w in test_sentence]
                vidcs = copy.copy(vocab_idcs)
                #print("vocab idcs ",vidcs)
                #padding sentences to a length of 10
                for _ in range(self.max_sent_len-len(vocab_idcs)):
                    vidcs.append(len(word_to_ix))
                Vidcs.append(vidcs)

            context_idxs = torch.tensor(Vidcs, dtype=torch.long)
            context_idxs = context_idxs.to(self.device)
            #embeddings = nn.Embedding(len(word_to_ix)+1, embedding_dim, padding_idx = len(word_to_ix))
            emb = self.embeddings(context_idxs).view((self.bsize, -1))
            embeds.append(emb)
        #print("got embeds ",embeds)
        return embeds

    def batchconv(self, inp, filt):
        #what it tries to do--
        #inp has a batch size, filter also has a batch size
        #it returns the batch of pair wise convolution image&filter along the batch dimension
        #print("In batchconv, input size ",inp.size())

        res = torch.cat([F.conv2d(inp[i].view((1,inp[i].size(0),inp[i].size(1),inp[i].size(2))), filt[i]) for i in range(self.bsize)],0)
        #print("In batchconv, result size ",res.size())
        return res
        #return torch.cat([F.conv2d(inp, filt[i]) for i in range(self.bsize)],0)

    def forward(self, image_input, sentences):
        """ Batch of image input and sentence embedding.
        :param image_input: batch x channel x height x width
                            (height and width cannot be smaller than 32)
        :param sentence_embedding: batch x text_emb
        :output outputs an image of the same size as input image with 1 channel. """

        x_original = self.conv_original_size0(image_input)
        x_original = self.conv_original_size1(x_original)

        #x1 = self.norm2(self.act(self.conv1(image_input)))
        x1 = self.conv1(image_input)
        #x2 = self.norm3(self.act(self.conv2(x1)))
        x2 = self.conv2(x1)
        #x3 = self.norm4(self.act(self.conv3(x2)))
        x3 = self.conv3(x2)
        #x4 = self.norm5(self.act(self.conv4(x3)))
        x4 = self.conv4(x3)
        x5 = self.conv5(x4) #this is layer4 according to -> https://github.com/usuyama/pytorch-unet#left-input-image-middle-correct-mask-ground-truth-rigth-predicted-mask

        if self.bsize ==1:
            sentence_embeddings = [self.embed_sent(sentence = i) for i in sentences]
        else:
            sentence_embeddings = self.embed_batch_sent(sentences)
        #sentence_embeddings.to(device)

        if sentence_embeddings is not None or len(sentence_embeddings)!=5:
            emb1 = sentence_embeddings[0].to(self.device)
            emb2 = sentence_embeddings[1].to(self.device)
            emb3 = sentence_embeddings[2].to(self.device)
            emb4 = sentence_embeddings[3].to(self.device)
            emb5 = sentence_embeddings[4].to(self.device)

            test = self.lang19(emb1)
            #print("test shape ",test.size())

            if self.bsize ==1:
                lf11 = F.normalize(self.lang19(emb1)).view([16, 64, 1, 1]) #language kernels for description type 1
                lf12 = F.normalize(self.lang19(emb2)).view([16, 64, 1, 1])
                lf13 = F.normalize(self.lang19(emb3)).view([16, 64, 1, 1])
                lf14 = F.normalize(self.lang19(emb4)).view([16, 64, 1, 1])
                lf15 = F.normalize(self.lang19(emb5)).view([16, 64, 1, 1])

                lf21 = F.normalize(self.lang28(emb1)).view([16, 64, 1, 1]) #language kernels for description type 2
                lf22 = F.normalize(self.lang28(emb2)).view([16, 64, 1, 1])
                lf23 = F.normalize(self.lang28(emb3)).view([16, 64, 1, 1])
                lf24 = F.normalize(self.lang28(emb4)).view([16, 64, 1, 1])
                lf25 = F.normalize(self.lang28(emb5)).view([16, 64, 1, 1])

                lf31 = F.normalize(self.lang37(emb1)).view([8, 128, 1, 1]) #language kernels for description type 3
                lf32 = F.normalize(self.lang37(emb2)).view([8, 128, 1, 1])
                lf33 = F.normalize(self.lang37(emb3)).view([8, 128, 1, 1])
                lf34 = F.normalize(self.lang37(emb4)).view([8, 128, 1, 1])
                lf35 = F.normalize(self.lang37(emb5)).view([8, 128, 1, 1])

                lf41 = F.normalize(self.lang46(emb1)).view([4, 256, 1, 1]) #language kernels for description type 4
                lf42 = F.normalize(self.lang46(emb2)).view([4, 256, 1, 1])
                lf43 = F.normalize(self.lang46(emb3)).view([4, 256, 1, 1])
                lf44 = F.normalize(self.lang46(emb4)).view([4, 256, 1, 1])
                lf45 = F.normalize(self.lang46(emb5)).view([4, 256, 1, 1])

                lf51 = F.normalize(self.lang55(emb1)).view([2, 512, 1, 1]) #language kernels for description type 5
                lf52 = F.normalize(self.lang55(emb2)).view([2, 512, 1, 1])
                lf53 = F.normalize(self.lang55(emb3)).view([2, 512, 1, 1])
                lf54 = F.normalize(self.lang55(emb4)).view([2, 512, 1, 1])
                lf55 = F.normalize(self.lang55(emb5)).view([2, 512, 1, 1])

            else:
                lf11 = self.lang19(emb1).view([self.bsize,16, 64, 1, 1]) #language kernels for description type 1
                lf12 = self.lang19(emb2).view([self.bsize,16, 64, 1, 1])
                lf13 = self.lang19(emb3).view([self.bsize,16, 64, 1, 1])
                lf14 = self.lang19(emb4).view([self.bsize,16, 64, 1, 1])
                lf15 = self.lang19(emb5).view([self.bsize,16, 64, 1, 1])

                lf21 = self.lang28(emb1).view([self.bsize,16, 64, 1, 1]) #language kernels for description type 2
                lf22 = self.lang28(emb2).view([self.bsize,16, 64, 1, 1])
                lf23 = self.lang28(emb3).view([self.bsize,16, 64, 1, 1])
                lf24 = self.lang28(emb4).view([self.bsize,16, 64, 1, 1])
                lf25 = self.lang28(emb5).view([self.bsize,16, 64, 1, 1])

                lf31 = self.lang37(emb1).view([self.bsize,8, 128, 1, 1]) #language kernels for description type 3
                lf32 = self.lang37(emb2).view([self.bsize,8, 128, 1, 1])
                lf33 = self.lang37(emb3).view([self.bsize,8, 128, 1, 1])
                lf34 = self.lang37(emb4).view([self.bsize,8, 128, 1, 1])
                lf35 = self.lang37(emb5).view([self.bsize,8, 128, 1, 1])

                lf41 = self.lang46(emb1).view([self.bsize,4, 256, 1, 1]) #language kernels for description type 4
                lf42 = self.lang46(emb2).view([self.bsize,4, 256, 1, 1])
                lf43 = self.lang46(emb3).view([self.bsize,4, 256, 1, 1])
                lf44 = self.lang46(emb4).view([self.bsize,4, 256, 1, 1])
                lf45 = self.lang46(emb5).view([self.bsize,4, 256, 1, 1])

                lf51 = self.lang55(emb1).view([self.bsize,2, 512, 1, 1]) #language kernels for description type 5
                lf52 = self.lang55(emb2).view([self.bsize,2, 512, 1, 1])
                lf53 = self.lang55(emb3).view([self.bsize,2, 512, 1, 1])
                lf54 = self.lang55(emb4).view([self.bsize,2, 512, 1, 1])
                lf55 = self.lang55(emb5).view([self.bsize,2, 512, 1, 1])



            if self.bsize==1:
                x1f1, x1f2, x1f3, x1f4, x1f5 = F.conv2d(x1, lf11), F.conv2d(x1, lf12), F.conv2d(x1, lf13), F.conv2d(x1, lf14), F.conv2d(x1, lf15)
                x2f1, x2f2, x2f3, x2f4, x2f5 = F.conv2d(x2, lf21), F.conv2d(x2, lf22), F.conv2d(x2, lf23), F.conv2d(x2, lf24), F.conv2d(x2, lf25)
                x3f1, x3f2, x3f3, x3f4, x3f5 = F.conv2d(x3, lf31), F.conv2d(x3, lf32), F.conv2d(x3, lf33), F.conv2d(x3, lf34), F.conv2d(x3, lf35)
                x4f1, x4f2, x4f3, x4f4, x4f5 = F.conv2d(x4, lf41), F.conv2d(x4, lf42), F.conv2d(x4, lf43), F.conv2d(x4, lf44), F.conv2d(x4, lf45)
                x5f1, x5f2, x5f3, x5f4, x5f5 = F.conv2d(x5, lf51), F.conv2d(x5, lf52), F.conv2d(x5, lf53), F.conv2d(x5, lf54), F.conv2d(x5, lf55)
            else:
                x1f1, x1f2, x1f3, x1f4, x1f5 = self.batchconv(x1, lf11), self.batchconv(x1, lf12), self.batchconv(x1, lf13), self.batchconv(x1, lf14), self.batchconv(x1, lf15)
                x2f1, x2f2, x2f3, x2f4, x2f5 = self.batchconv(x2, lf21), self.batchconv(x2, lf22), self.batchconv(x2, lf23), self.batchconv(x2, lf24), self.batchconv(x2, lf25)
                x3f1, x3f2, x3f3, x3f4, x3f5 = self.batchconv(x3, lf31), self.batchconv(x3, lf32), self.batchconv(x3, lf33), self.batchconv(x3, lf34), self.batchconv(x3, lf35)
                x4f1, x4f2, x4f3, x4f4, x4f5 = self.batchconv(x4, lf41), self.batchconv(x4, lf42), self.batchconv(x4, lf43), self.batchconv(x4, lf44), self.batchconv(x4, lf45)
                x5f1, x5f2, x5f3, x5f4, x5f5 = self.batchconv(x5, lf51), self.batchconv(x5, lf52), self.batchconv(x5, lf53), self.batchconv(x5, lf54), self.batchconv(x5, lf55)

            #x5f1, x5f2, x5f3, x5f4, x5f5 = self.dropout(x5f1), self.dropout(x5f2), self.dropout(x5f3), self.dropout(x5f4), self.dropout(x5f5)

        else:
            if len(sentence_embeddings)!=5:
                raise AssertionError("Please pass a list of exactly 5 embeddings into the model.")
            raise AssertionError("Embedding should not be none.")

        x5f = torch.cat([x5f1, x5f2, x5f3, x5f4, x5f5], 1)
        x5f = self.condense1(x5f) #here should be 512 channels now
        print("Check x5f shape ",x5f.size())

        x5f = self.layer4_1x1(x5f)
        x = self.upsample(x5f)


        x46 = torch.cat([x4f1, x4f2, x4f3, x4f4, x4f5], 1)
        x46 = self.condense2(x46) #here should be 256 channels now
        layer3 = self.layer3_1x1(x46)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)


        x37 = torch.cat([x3f1, x3f2, x3f3, x3f4, x3f5], 1)
        x37 = self.condense3(x37)
        layer2 = self.layer2_1x1(x37) #here should be 128 channels now
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)


        x28 = torch.cat([x2f1, x2f2, x2f3, x2f4, x2f5], 1)
        x28 = self.condense4(x28)
        layer1 = self.layer1_1x1(x28)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)


        x19 = torch.cat([x1f1, x1f2, x1f3, x1f4, x1f5], 1)
        x19 = self.condense5(x19)
        layer0 = self.layer0_1x1(x19)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)


        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


if __name__ == '__main__':

    train = True
    debug = False
    save_predictions = True
    resize_factor = 224 #can pass a fraction or an actual square dimension
    focus_object = "" #learn to segment obly bed given descriptions of bed
    train_batch_size = 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') #does not yet work for gpu
    print("Using device ",device)

    batch_size = 1
    if train:
        batch_size = train_batch_size #set the training batch size here
    else:
        batch_size = 1

    
    #can hash the data_loader class from first time running to load it much faster next time
    dat = data_loader('data/descriptions.json', prehashed = "", batch_size = batch_size, resize_factor = resize_factor, focus_object = focus_object) 
    #dat = data_loader('data/descriptions.json', prehashed = "prehash.npy", batch_size = batch_size, resize_factor = resize_factor, focus_object = focus_object) 
    w2i = dat.w2i

    #initialize model
    model = Unet5Contextual(w2i,bsize = batch_size, device = device) #chanels in input image=3, chanels in output segmentation =1, length of embedding = embedding_dim

    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5,lr=0.00001)
    
    #bunch of distance functions for comparison
    distance0 = nn.MSELoss()
    distance1 = nn.CrossEntropyLoss()
    distance2 = nn.KLDivLoss()
    distance3 = nn.NLLLoss()
    distance4 = BCELoss2d()

    #load previous weights if possible
    try:
        model.load_state_dict(torch.load("unet_weights.pth"))
        print("Model loading a success !")
    except:
        print("ALAS! model loading failed ")

    positions = list(range(len(dat.rgb_idxs)))
    model.to(device)
    print("Checking if model was properly transferred to GPU ")
    print(next(model.parameters()).device)
    #print("Model summary")
    #from torchsummary import summary
    #summary(model, input_size=(3, 150, 150))

    #sys.exit(0)

    if train:
        for i in range(100000):
            if batch_size==1:
                pos = random.choice(positions)
                im1,im2,desc = dat.next(position = pos) #im1,im2 are tensor images of rgb and seg, desc is a list of sentences in string format
            else:
                im1,im2,desc = dat.next_batch()

            im1 = im1.to(device)
            im2 = im2.to(device)
            #desc.to(device)

            optimizer.zero_grad()

            output = model(im1,desc)
            #loss = distance4(output, im2)
            loss = seg_loss(output, im2)

            
            loss.backward()
            optimizer.step()

            print("loss ",loss.data)
            if i%10==0 and debug==False:
                print("saving model weights ")
                torch.save(model.state_dict(), "unet_weights.pth")

                if save_predictions:
                    
                    
                    if resize_factor<=1:
                        seg_pred = F.sigmoid(output).detach().cpu().numpy().reshape((batch_size,int(300*dat.resize_factor),int(300*dat.resize_factor)))
                        target = im2.cpu().numpy().reshape((batch_size,int(300*dat.resize_factor),int(300*dat.resize_factor)))
                        inputs = im1.cpu().numpy().reshape((batch_size,int(300*dat.resize_factor),int(300*dat.resize_factor),3))
                    else:
                        seg_pred = F.sigmoid(output).detach().cpu().numpy().reshape((batch_size,resize_factor,resize_factor))
                        target = im2.cpu().numpy().reshape((batch_size,resize_factor,resize_factor))
                        inputs = im1.cpu().numpy().reshape((batch_size,resize_factor,resize_factor,3))

                    cv2.imwrite("predictions/"+repr(i%100)+'.png',seg_pred[0]*255.0)
                    cv2.imwrite("predictions/target_"+repr(i%100)+'.png',target[0]*255.0)
                    cv2.imwrite("predictions/input_"+repr(i%100)+'.png',inputs[0]*255.0)


    else:
        im1,im2,desc = dat.next(position = 1000)
        print("description ",desc)
        
        rgb = im1.numpy().reshape((300,300,3))
        print("passed input rgb image ")
        cv2.imshow("RGB",rgb)
        cv2.waitKey(0)

        seg = im2.numpy().reshape((300,300))
        print("Expected target segmentation image ")
        cv2.imshow("SEG",seg)
        cv2.waitKey(0)

        output = model(im1,desc)

        seg_pred = output.detach().numpy().reshape((300,300))
        print("Model predicted segmentation image ")
        cv2.imshow("PRED",seg_pred)
        cv2.waitKey(0)




'''
Training :
Each image can have upto a collection of 5 sentences describing the object
Each sentence describes a unique attribute of that object in the image
if one of the description out of 5 is not possible to make, then simply pass the name of the object as the sentence

#Improving
1. Train in minibatches -> need to modify both data_loader and unet classes (DONE)
2. look into the dnorm functions inside unet class
3. self.deconv5 in unet class may be a huge jump from so many channels to just 1 channel -> see if the transition can be made in a step like fashion
4. See if performance improves while training on data from nultiple rooms (DONE)
5. way to use pretrained word2vec embeddings and pretrained resnet embeddings (or pretrained yolo object detection conv layers)
6. Dropout of 0.5 in model initialization seems wierd (DONE- reduced dropout factor)
7. too much normalization everywhere, but im using only batch size of 1 (DONE- now using batchsize of 16)

use much smaller input and target image sizes 
use pretrained resnet backbone -> code here-> https://github.com/usuyama/pytorch-unet
use dice loss, again code is here-> https://github.com/usuyama/pytorch-unet




#Transfers to cluster
(first connect)
ssh hsaha@scslab-beast1.me.iastate.edu
pass- scslab_homagni

transfer to qisai's workstation
ssh super@10.24.250.174
pass - meisu2014

(transfer whole cross modal folder)
scp -r /home/hom/Desktop/ai2thor/cross_modal/ hsaha@scslab-beast1.me.iastate.edu:/data/hsaha/cross_modal/
scp -r /home/hom/Desktop/ai2thor/cross_modal/ super@10.24.250.174:/home/super/Desktop/ai2thor/

scp -r super@10.24.250.174:/home/super/Desktop/ai2thor/cross_modal/data/ /home/hom/Desktop/ai2thor/cross_modal/ 

scp -r super@10.24.250.174:/home/super/Desktop/ai2thor/hutils/panorama_data/ /home/hom/Desktop/ai2thor/transfer/

scp -r /home/hom/Desktop/ai2thor/transfer/ super@10.24.250.174:/home/super/Desktop/ai2thor/
scp -r super@10.24.250.174:/home/super/Desktop/ai2thor/transfer /home/hom/Desktop/ai2thor/


(transfer this python file-> Type this in a blank new terminal)
scp -r /home/hom/Desktop/ai2thor/cross_modal/unet_ling.py hsaha@scslab-beast1.me.iastate.edu:/data/hsaha/cross_modal/

(download the weights)
scp -r hsaha@scslab-beast1.me.iastate.edu:/data/hsaha/cross_modal/unet_weights.pth /home/hom/Desktop/ai2thor/cross_modal/

(download hash)
scp -r hsaha@scslab-beast1.me.iastate.edu:/data/hsaha/cross_modal/prehash.npy /home/hom/Desktop/ai2thor/cross_modal/

Project location
/data/hsaha/cross_modal/cross_modal/
Env
source activate vln
'''

