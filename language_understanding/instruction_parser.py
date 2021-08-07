import os
import sys


import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES
from collections import defaultdict

import traceback
import sys
from datetime import datetime as dt

logger = logging.getLogger(__name__)

### Model Directory ###

#model_dir = "/home/microway/Desktop/fotouhif/Ai2thor+Alfred_work/JointBERT-master_5/JointBERT-master/alfred_model_1000_modification"
#model_dir = "alfred_model_1000_modification"
model_dir = "/ai2thor/language_understanding/alfred_model_1000_modification"

### 

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"

### Load model ###

def get_args():
    return torch.load(os.path.join(model_dir, 'training_args.bin'))

def load_model(args, device):
    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model



### if you have one sentence in code and you want to see the output ###
def sen_list_input(sent_list):
    lines = []
    sep_obj_name = ["desk shelf", "laundry basket", "book shelf", "tea kettle", "barred rack", "sliced bread", "alarm clock", "entertainment stand", "arm chair", "basket ball", "base ball", "bath tub", "bath tub basin", "butter knife", "cell phone", "coffee maker", "coffee table", "credit card", "desk lamp" , "dinning table", "dish sponge", "dog bed", "garbage can", "garbage bin", "trash bin", "trash can", "floor lamp", "garbage bag", "hand towel", "hand towel hold", "house plant", "key chain", "light swich", "paper towel", "paper shaker", "remote control", "room decor", "scrub brush", "side table", "night stand", "salt shaker", "soap bar", "soap bottle", "spray bottle", "stove burner", "rubbish bin", "teddy bear", "book shelves", "book shelf", "tennis racket", "tissue box", "toilet paper", "toilet paper roll", "towel holder", "tv stand", "vaccum cleaner", "watering can", "wine bottle", "wine glass", "shopping bag", "stove knob", "love seat", "apple sliced", "sliced apple", "apple slice", "lettuce slice", "sliced lettuce", "potato sliced", "sliced potato", "potato slice" ,"tomato sliced", "sliced tomato", "tomato slice", "sink counter", "kitchen counter", "kitchen island", "rediator cover", "refrigarator door", "media player", "media device", "sink drying rack", "sink basin", "water tank", "fridge door", "cabinet door", "television stand", "microwave door", "drying rack", "toilet tank lid", "toilet tank","bottle shelf", "sink drawer", "kitchen shelf", "coffee maker", "kitchen sink" , "table lamp", "waste basket", "kitchen table", "island counter", "night table", "dresser drawer"]

    for sent in sent_list:
        sent = sent.replace(",","")
        sent = sent.replace(".","")
        sent = sent.replace('"',"")
        for objects in sep_obj_name:
            if objects in sent:
                com_object = objects.replace(" ","")
                sent = sent.replace(objects, com_object)
        sent = sent.lower()
        print(sent)
        words = sent.split()
        lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset



class parse(object):
    def __init__(self):
        init_logger()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        self.pred_config = self.parser.parse_args()

        #slot_dict = {}
        #slot_dict = defaultdict(list)
        #pre_pred = []
        
        self.args = get_args()
        print("got args at line 171 ",self.args)
        print("changing args data dir ")
        #args.data_dir = '/home/hom/Desktop/ai2thor/JointBERT_repository/data'
        self.args.data_dir = '/ai2thor/language_understanding/data'
        print("changing args model dir ")
        #args.model_dir = "/home/hom/Desktop/ai2thor/JointBERT_repository/alfred_model_1000_modification"
        self.args.model_dir = "/ai2thor/language_understanding/alfred_model_1000_modification"
        print("changing intent label file ")
        self.args.intent_label_file = "intent_label.txt"
        print("changing slot label file ")
        self.args.slot_label_file = "slot_label.txt"
        logger.info(self.args)

        self.device = get_device(self.pred_config)
        self.model = load_model(self.args, self.device)

        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)

        # Convert input file to TensorDataset
        self.pad_token_label_id = self.args.ignore_index
        self.tokenizer = load_tokenizer(self.args)

    def predict(self, sent_list):
        lines = sen_list_input(sent_list)
        #print("LINES", lines)
        dataset = convert_input_file_to_tensor_dataset(lines, self.pred_config, self.args, self.tokenizer, self.pad_token_label_id)

        # Predict
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

        all_slot_label_mask = None
        intent_preds = None
        slot_preds = None

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "intent_label_ids": None,
                          "slot_labels_ids": None}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                _, (intent_logits, slot_logits) = outputs[:2]

                # Intent Prediction
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

                # Slot prediction
                if slot_preds is None:
                    if self.args.use_crf:
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds = np.array(self.model.crf.decode(slot_logits))
                    else:
                        slot_preds = slot_logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                    else:
                        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

        intent_preds = np.argmax(intent_preds, axis=1)

        if self.args.use_crf:
            slot_preds = np.array(self.model.crf.decode(slot_logits))
        else:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        list_dic_parsing = []
        list_intent = []


        # Write to output file
        with open("pred_out_alfred.txt", "w", encoding="utf-8") as f:
            for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):

                intent_alfred = self.intent_label_lst[intent_pred]
                #print(slot_preds)
                #print(words)
                line = ""
                slot_dict = {}
                pre_pred = []
                for word, pred in zip(words, slot_preds):
                    if pred == 'O':
                        line = line + word + " "
                    else:
                        line = line + "[{}:{}] ".format(word, pred)

                        if pred in pre_pred:
                            word_list = []
                            word_list.append(slot_dict[pred])
                            word_list.append(word)
                            slot_dict[pred] = word_list

                        else:
                            slot_dict[pred] = word

                        pre_pred.append(pred)

                list_dic_parsing.append(slot_dict)
                list_intent.append(intent_alfred)
                f.write("<{}> -> {}\n".format(self.intent_label_lst[intent_pred], line.strip()))
        

        logger.info("Prediction Done!")

        return list_intent, list_dic_parsing



if __name__ == "__main__":

    
    # For predicting list of sentences for say a single task
    p = parse() #takes around 0.05s for parsing a single sentence

    inp = input("Do you want to test a single sentence (t) | or do you want to create parses for a batch of instructions (b) ? ")

    if inp=='t':
        
        s = input("Type the sentence here -> ")
        #sent_list = ['Go to the table to your right and bring the water bottle']
        sent_list = [s]
        print("Entering the sentence ",dt.now())
        list_intent, list_dic_parsing = p.predict(sent_list)
        print("Intent =" , list_intent)
        print("Slot_Dictionary =", list_dic_parsing)
        print("Done predicting ",dt.now())
        sys.exit(0)
    
    

    
    else:
        # This one is for predicting a bunch of instructions sorted by room number and task number for alfred tasks
        # The instructions should be presorted and obtained in ExpertDemos/ directory
        # This can be done by runnning the code save_expert_demos.py in /ai2thor/hutils/ (need to specify range of rooms)

        prediction_dictionary = {}

        all_instructions = []
        all_rooms = []
        all_tasks = []

        room_range = [0,31] #cover rooms 201 to 230 and 301-330
        task_range = [0,300]#probably maximum 100 tasks in each room, for 0-30 there are around 250+ tasks each room


        for rn in range(room_range[0],room_range[1]): 
            for task in range(task_range[0],task_range[1]): 
                try:
                    d = np.load('ExpertDemos/expert_demo_'+repr(rn)+'_'+repr(task)+'.npy',allow_pickle = 'TRUE').item()
                    all_instructions.extend(d["instructions"])
                    all_rooms.extend([rn]*len(d["instructions"]))
                    all_tasks.extend([task]*len(d["instructions"]))
                except:
                    print("does not exist !")
                    traceback.print_exc()

        list_intent, list_dic_parsing = predict(all_instructions)

        count = 0
        all_pred = {repr(i):{repr(task):{"sentence":[],"intent":[],"slots":[]} for task in range(task_range[0],task_range[1])} for i in range(room_range[0],room_range[1])}

        for i in range(len(all_instructions)):
            try:
                d = np.load('ExpertDemos/expert_demo_'+repr(all_rooms[i])+'_'+repr(all_tasks[i])+'.npy',allow_pickle = 'TRUE').item()
                all_pred[repr(all_rooms[i])][repr(all_tasks[i])]["sentence"].append(all_instructions[count])
                all_pred[repr(all_rooms[i])][repr(all_tasks[i])]["intent"].append(list_intent[count])
                all_pred[repr(all_rooms[i])][repr(all_tasks[i])]["slots"].append(list_dic_parsing[count])
                count+=1
            except:
                print(" prediction does not exist !")
        np.save('ExpertDemos/all_predictions.npy',all_pred)
    
    
    




