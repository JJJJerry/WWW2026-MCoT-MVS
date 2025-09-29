import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class DINOModel:
    def __init__(self, model_id, device):
        self.device = device
        self.grounding_model, self.processor = self.load_grounding_model(
            model_id=model_id
        )

    def load_grounding_model(self, model_id):
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        return grounding_model, processor

    @torch.inference_mode()
    def get_batch_groundings(self, grounding_texts, pil_image_list):
        for i in range(len(pil_image_list)):
            pil_image_list[i] = pil_image_list[i].convert('RGB')
        inputs = self.processor(images=pil_image_list, text=grounding_texts, padding=True, return_tensors="pt").to(self.device)
        outputs = self.grounding_model(**inputs)
        grounding_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[pil_image.size[::-1] for pil_image in pil_image_list]
        )
        for i in range(len(grounding_results)):
            if grounding_results[i]['scores'].shape[0]==0:
                grounding_results[i]['scores']=torch.tensor([1.0]).to(self.device)
                grounding_results[i]['boxes']=torch.tensor([[0.0,0.0,pil_image_list[i].size[0],pil_image_list[i].size[1]]]).to(self.device)
                grounding_results[i]['text_labels']=['None']
                grounding_results[i]['labels']=['None']
        return grounding_results
