import os
import hydra
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMModel:
    def __init__(self, sam_abs_dir_path, device):
        self.device = device
        self.sam2_predictor = self.load_sam2_predictor(sam_abs_dir_path)

    def load_sam2_predictor(self, sam_abs_dir_path):
        sam2_checkpoint = os.path.join(sam_abs_dir_path, "sam2.1_hiera_large.pt")
        model_cfg = "sam2.1_hiera_l"
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=sam_abs_dir_path)
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        return sam2_predictor

    @torch.inference_mode()
    def get_batch_masks(self, input_boxes_list, pil_image_list):
        masks_list = []
        scores_list = []
        logits_list = []
        for i, input_boxes in enumerate(input_boxes_list):
            self.sam2_predictor.set_image(np.array(pil_image_list[i].convert("RGB")))
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            masks_list.append(masks)
            scores_list.append(scores)
            logits_list.append(logits)
        return masks_list, scores_list, logits_list
