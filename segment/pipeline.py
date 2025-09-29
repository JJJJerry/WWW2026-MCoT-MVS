import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sam_utils import SAMModel
from ram_utils import RAMModel
from dino_utils import DINOModel
from visualize_utils import draw_mask_and_segmentaion


class MaskGenerator:
    def __init__(
        self,
        ram_model: RAMModel,
        dino_model: DINOModel,
        sam_model: SAMModel,
        workers=4,
    ):
        self.ram = ram_model
        self.dino = dino_model
        self.sam = sam_model
        self.workers = workers
    def generate_masks(self, image_path_list, save_dir, is_draw: bool):
        batch_num = len(image_path_list) // self.workers + 1
        for i in tqdm(range(batch_num)):
            batch_image_path_list = image_path_list[
                i * self.workers : (i + 1) * self.workers
            ]
            if len(batch_image_path_list) == 0:
                break
            self.batch_generate_masks(batch_image_path_list, save_dir, is_draw)
            
    def batch_generate_masks(self, image_path_list, save_dir, is_draw: bool):

        pil_images = []
        for image_path in image_path_list:
            pil_image = Image.open(image_path)
            pil_images.append(pil_image)

        # RAM
        pred_tags = self.ram.get_batch_tags(pil_image_list=pil_images)

        # tags to str
        grounding_texts = []
        for taglist in pred_tags:
            res = ""
            for tag in taglist:
                res += f"{tag}. "
            grounding_texts.append(res)

        # DINO
        grounding_results = self.dino.get_batch_groundings(
            grounding_texts=grounding_texts, pil_image_list=pil_images)
        

        # get boxes
        input_boxes_list = []
        for grounding_result in grounding_results:
            input_boxes=grounding_result["boxes"].cpu().numpy()
            input_boxes_list.append(input_boxes)

        # SAM
        masks_list, scores_list, logits_list = self.sam.get_batch_masks(
            input_boxes_list=input_boxes_list, pil_image_list=pil_images)
        

        for i in range(len(image_path_list)):
            image_name = os.path.basename(image_path_list[i]).split(".")[0]
            output_dir = os.path.join(save_dir, "masks", image_name) # save dir
            os.makedirs(output_dir, exist_ok=True)
            
            draw_mask_and_segmentaion(
                ori_image_path=image_path_list[i],
                output_dir=output_dir,
                masks=masks_list[i],
                scores=scores_list[i],
                grounding_result=grounding_results[i],
                is_draw=is_draw
            )
            masks_list[i]=masks_list[i].astype(np.int16)
            np.savez_compressed(os.path.join(output_dir, "masks.npz"), array=masks_list[i])