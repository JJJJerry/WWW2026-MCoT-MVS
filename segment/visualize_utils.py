import numpy as np
import cv2
import json
import supervision as sv
import os
from supervision.draw.color import ColorPalette
from PIL import Image

def draw_mask_and_segmentaion(ori_image_path:str,output_dir:str,masks,scores,grounding_result,is_draw=False):
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = grounding_result["scores"].cpu().numpy().tolist()
    class_names = grounding_result["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    input_boxes=grounding_result["boxes"].cpu().numpy()
    
    if is_draw:
        """
        Visualize image with supervision useful API
        """   
        img = cv2.imread(ori_image_path)
        
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """

        CUSTOM_COLOR_MAP = [
            "#e6194b",
            "#3cb44b",
            "#ffe119",
            "#0082c8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#d2f53c",
            "#fabebe",
            "#008080",
            "#e6beff",
            "#aa6e28",
            "#fffac8",
            "#800000",
            "#aaffc3",
        ]
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": ori_image_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
            
                "score": score,
            }
            for class_name, box, score in zip(class_names, input_boxes, scores)
        ],
        "box_format": "xyxy",
    }

    with open(os.path.join(output_dir, "grounded_sam2_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    