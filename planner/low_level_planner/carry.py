import numpy as np
from skimage.measure import regionprops, label

import sys
import os
os.environ['MAIN'] = '/ai2thor'
sys.path.append(os.path.join(os.environ['MAIN']))

import planner.low_level_planner.refinement as refinement
import planner.low_level_planner.object_localization as object_localization

unit_refinement = refinement.unit_refinement
location_in_fov = object_localization.location_in_fov

def carry(refinement_obj,refinement_rel,env):
    print("(manipulation_signatures.py -> carry)")
    ref_obj = refinement_obj.split(',')
    ref_rel = refinement_rel.split(',')

    for r in range(len(ref_obj)):
        env = unit_refinement(env,ref_obj[r])
        try:
            d = ref_rel[r]
            if d=="left":
                env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
            if d=="right":
                env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
            if d=="facing":
                mask_image = env.get_segmented_image()
                depth_image = env.get_depth_image()
                lf,mf,rf,areas,_ = location_in_fov(env, mask_image,depth_image)
                for k in lf.keys():
                    k1 = ref_obj[r]
                    if k1+'|' in k:
                        print("found ",k1," in ",k)
                        env.step(dict({"action": "MoveLeft", "moveMagnitude" : 0.25}))
                        break
                for k in rf.keys():
                    k1 = ref_obj[r]
                    if k1+'|' in k:
                        print("found ",k1," in ",k)
                        env.step(dict({"action": "MoveRight", "moveMagnitude" : 0.25}))
                        break
        except:
            break
    return env