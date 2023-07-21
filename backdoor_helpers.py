import json
import os


def img_modifier(f):
    class ImgModClass:
        def __init__(self, **args):
            self.args = args
        
        def __call__(self, img, idx):
            return f(img, idx, **self.args)

        def __repr__(self):
            return f"{f.__name__}({', '.join(f'{key}={val}' for key,val in self.args.items())})"
    
    def new_fn(**x):
        return ImgModClass(**x)
    
    return new_fn

def target_modifier(f):
    class TargetModClass:
        def __init__(self, **args):
            self.args = args
        
        def __call__(self, target):
            return f(target, **self.args)

        def __repr__(self):
            return f"{f.__name__}({', '.join(f'{key}={val}' for key,val in self.args.items())})"
    
    def new_fn(**x):
        return TargetModClass(**x)
    
    return new_fn


def get_traffic_signs(frame_id):
    json_path = f"/mnt/ZOD/single_frames/{frame_id}/annotations/traffic_signs.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            traffic_signs_json = json.load(f)

        boxes = []
        for object in traffic_signs_json:
            boxes.append(object["geometry"]["coordinates"])
        return boxes
    else:
        return []