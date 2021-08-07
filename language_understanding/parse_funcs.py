import equivalent_concepts as eqc
import copy

toggleable = eqc.toggleable

#toggleable = ["lamp","Lamp",'light','Light','tablelamp']


def infer_action(obj):
    if obj in toggleable:
        return "on"
    else:
        return []



def replace_missing_slots(slot_dict):
    '''
    slot_dicts = [{'action_navi': 'Move', 'target_rel': ['bottom', 'right'], 'target_obj': ['wood', 'dresser']}, 
                    {'action_n_navi': [['Open', 'remove'], 'close'], 'refinement_rel': ['bottom', 'right'], 'target_obj': [['drawer', 'watch,'], 'drawer']}, 
                    {'action_n_navi': 'Carry', 'target_obj': 'watch', 'refinement_rel': ['left','face'], 'refinement_obj': ['dresser', 'lamp']}, 
                    {'action_n_navi': 'on', 'target_obj': 'lamp'}]
    '''
    for s in slot_dict:
        if "action_navi" not in s.keys():
            s['action_navi'] = ''
        if "target_rel" not in s.keys():
            s['target_rel'] = ''
        if "target_obj" not in s.keys():
            s['target_obj'] = ''
        if "action_n_navi" not in s.keys():
            anv = infer_action(s['target_obj'])
            if anv==[]:
                s['action_n_navi'] = ''
            else:
                s['action_n_navi'] = anv
        if "action_desc" not in s.keys():
            s['action_desc'] = ''
        if "action_intensity" not in s.keys():
            s['action_intensity'] = ''
        if "refinement_rel" not in s.keys():
            s['refinement_rel'] = ''
        if "refinement_obj" not in s.keys():
            s['refinement_obj'] = ''
        if "refinement_attri" not in s.keys():
            s['refinement_attri'] = ''
    return slot_dict

def fnl(obj_list):
    o_list = []
    def flatten_nested_list(obj_list):
        if isinstance(obj_list,list):
            for o in obj_list:
                a = flatten_nested_list(o)
                if a!=None:
                    o_list.append(a)
        else:
            o_list.append(obj_list)
    flatten_nested_list(obj_list)
    return o_list

def join_words(sent):
    joints = {'trash can':'trashcan', 'cell phone':'cellphone', 'side table': 'sidetable', 'night stand':'nightstand',
                'table lamp':'tablelamp', 'tennis racket':'tennisracket', 'alarm clock':'alarmclock',
                'book shelves':'bookshelves','desk shelf':'deskshelf','bedside table':'bedsidetable', 
                'night table':'nighttable'}
    for j in joints.keys():
        sent = sent.replace(j,joints[j])
    return sent

def split_slots(slot, sent):
    sent = sent.replace(',','')
    sent = sent.replace('.','')
    sent = sent.lower()
    sent = join_words(sent)

    print("Input navigation sentence ->",sent)

    targets = fnl(slot['target_obj'])
    action_navi = fnl(slot['action_navi'])
    action_desc = fnl(slot['action_desc'])
    action_int_desc = fnl(slot['action_intensity']) #is a flat list of words

    slots = []
    partial_sents = []
    tc = 0
    idx1 = 0
    idx2 = 0

    #print("targets ",targets)
    while idx2<len(sent):
        try:
            idx2 = sent.index(targets[tc])+len(targets[tc])
        except: #no more targets in the sentence
            idx2 = len(sent)
            #traceback.print_exc()

        sent_p = sent[idx1:idx2]
        #print("partial sentence ",sent_p)
        partial_sents.append(sent_p)
        n_action_navi = []
        n_action_desc = []
        n_action_int_desc = []


        #for a in range(len(action_navi)):
        for s in sent_p.split(' '):
            try:
                if s in action_navi:
                    n_action_navi.append(s)
            except:
                #print("out of index ")
                pass

            try:
                if s in action_desc:
                    n_action_desc.append(s)
            except:
                #print("out of index ")
                pass

            try:
                if s in action_int_desc:
                    n_action_int_desc.append(s)
            except:
                #print("out of index ")
                pass
        try:
            n_target = targets[tc]
        except:
            n_target = ''
        n_slot = copy.copy(slot)
        n_slot['target_obj'] = n_target
        n_slot['action_navi'] = n_action_navi
        n_slot['action_desc'] = n_action_desc
        n_slot['action_intensity'] = action_int_desc

        slots.append(n_slot)

        tc+=1
        idx1 = copy.copy(idx2)

    #print("split slots ",slots)
    return slots, partial_sents