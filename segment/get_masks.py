import os
from sam_utils import SAMModel
from ram_utils import RAMModel
from dino_utils import DINOModel
from pipeline import MaskGenerator
import json

def cirr():
    image_path_list=[]
    cirr_base_path = "../data/cirr/cirr_dataset"
    cirr_base_path = cirr_base_path
    with open(os.path.join(cirr_base_path,'cirr/image_splits/split.rc2.test1.json'),'r') as f:
        test_split = json.load(f)
    with open(os.path.join(cirr_base_path,'cirr/image_splits/split.rc2.val.json'),'r') as f:
        val_split = json.load(f)
    with open(os.path.join(cirr_base_path,'cirr/image_splits/split.rc2.train.json'),'r') as f:
        train_split = json.load(f)
    res={}
    res.update(test_split)
    res.update(val_split)
    res.update(train_split)
    for value in res.values():
        image_path_list.append(os.path.join(cirr_base_path,value))
        
    ram = RAMModel(
            pretrained="./ram_models/ram_plus_swin_large_14m.pth",
            tag_file_path="./ram_models/ram_tag_list.txt",
            device="cuda",
        )
    dino = DINOModel(model_id="IDEA-Research/grounding-dino-tiny", device="cuda")
    sam_abs_dir_path = os.path.abspath("./sam2/checkpoints")
    sam = SAMModel(sam_abs_dir_path=sam_abs_dir_path, device="cuda")
    maskGenerator = MaskGenerator(ram_model=ram, dino_model=dino, sam_model=sam, workers=1)
    maskGenerator.generate_masks(image_path_list=image_path_list,save_dir='../data/cirr/segment/masks',is_draw=False)

def fiq():
    image_path_list = []
    fiq_base_path = "../data/fiq/fashionIQ_dataset"
    image_name_list = []
    dress_types = ['dress','shirt','toptee']
    splits = ['val','train']
    for dress_type in dress_types:
        for split in splits:
            with open(os.path.join(fiq_base_path,f'image_splits/split.{dress_type}.{split}.json'),'r') as f:
                data = json.load(f)
                image_name_list.extend(data)
    image_name_list = list(set(image_name_list))

    for image_name in image_name_list:
        image_path_list.append(os.path.join(fiq_base_path,'images',image_name+'.png'))

    ram = RAMModel(
            pretrained="ram_models/ram_plus_swin_large_14m.pth",
            tag_file_path="ram_models/ram_tag_list.txt",
            device="cuda",
        )
    dino = DINOModel(model_id="IDEA-Research/grounding-dino-tiny", device="cuda")
    sam_abs_dir_path = os.path.abspath("./sam2/checkpoints")
    sam = SAMModel(sam_abs_dir_path=sam_abs_dir_path, device="cuda")
    maskGenerator = MaskGenerator(ram_model=ram, dino_model=dino, sam_model=sam, workers=1)
    maskGenerator.generate_masks(image_path_list=image_path_list,save_dir='../data/fiq/segment/masks',is_draw=False)

if __name__=="__main__":
    cirr()
    fiq()

