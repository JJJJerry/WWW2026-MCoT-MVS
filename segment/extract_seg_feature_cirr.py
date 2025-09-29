import open_clip
import torch
import os
from PIL import Image
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import json
import torch
from torch import nn

def load_mask(path):
    mask_array = np.load(path)["array"]
    if len(mask_array.shape) == 4:
        mask_array = mask_array.reshape(
            mask_array.shape[0], mask_array.shape[2], mask_array.shape[3]
        )
    return mask_array
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)
class Clip(nn.Module):
    def __init__(self):
        super(Clip,self).__init__()
        self.clip,self.preprocess_train,self.preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
    def preprocess_image(self,images):
        if isinstance(images,Image.Image):
            images = [images]
        processed_image = []
        for image in images:
            transformed_image = None
            if self.training:
                transformed_image = self.preprocess_train(image)
            else :
                transformed_image = self.preprocess_val(image)
            processed_image.append(transformed_image)
        processed_image = torch.stack(processed_image).cuda()
        return processed_image
    def encode_image(self,images):
        processed_image = self.preprocess_image(images)
        image_feature = self.clip.encode_image(processed_image)
        return image_feature
    def encode_text(self,texts):
        if isinstance(texts,str):
            texts = [texts]
        input_ids = self.tokenizer(texts).cuda()
        text_feature = self.clip.encode_text(input_ids)
        return text_feature
    def visual_out(self,x):
        x = self.clip.visual.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)  

        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
     
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)
        x = self.clip.visual.transformer(x)

        if self.clip.visual.attn_pool is not None:
            if self.clip.visual.attn_pool_contrastive is not None:
                
                x = self.clip.visual.ln_post(x)  
                tokens = self.clip.visual.attn_pool(x)
                if self.clip.visual.attn_pool_type == 'parallel':
                    pooled = self.clip.visual.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.clip.visual.attn_pool_contrastive(tokens)
            else:
        
                x = self.clip.visual.attn_pool(x)
                x = self.clip.visual.ln_post(x)
                pooled, tokens = self.clip.visual._global_pool(x)
        
        elif self.clip.visual.final_ln_after_pool:
            pooled, tokens = self.clip.visual._global_pool(x)
            pooled = self.clip.visual.ln_post(pooled)

        else:
            x = self.clip.visual.ln_post(x)
            pooled, tokens = self.clip.visual._global_pool(x)
        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj
            tokens = tokens @ self.clip.visual.proj
            return pooled,tokens
        if self.clip.visual.output_tokens:
            return pooled, tokens

        return pooled, x

model = Clip()
model.cuda().eval()

image_path_list = []
cirr_base_path = (
    "../data/cirr/cirr_dataset"
)
cirr_base_path = cirr_base_path
with open(
    os.path.join(cirr_base_path, "cirr/image_splits/split.rc2.test1.json"), "r"
) as f:
    test_split = json.load(f)
with open(
    os.path.join(cirr_base_path, "cirr/image_splits/split.rc2.val.json"), "r"
) as f:
    val_split = json.load(f)
with open(
    os.path.join(cirr_base_path, "cirr/image_splits/split.rc2.train.json"), "r"
) as f:
    train_split = json.load(f)
res = {}
res.update(test_split)
res.update(val_split)
res.update(train_split)
for value in res.values():
    image_path_list.append(os.path.join(cirr_base_path, value))
seg_feature_dir = "../data/cirr/segment/seg_features_vit-h_patch"
mask_base_path = (
    "../data/cirr/segment/masks/"
)
with torch.inference_mode():
    for img_path in tqdm(image_path_list):
        img_name = os.path.basename(img_path).split(".")[0]
        save_dir = os.path.join(seg_feature_dir, img_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "seg_feature.pt")
        # if os.path.exists(save_path):
        #    continue
        img = Image.open(img_path)
        mask_array = load_mask(os.path.join(mask_base_path, img_name, "masks.npz"))
        
        cls,tokens = model.visual_out(model.preprocess_image(img)) # (1,256,1024)
        feature = []
        for i in range(mask_array.shape[0]):
            seg_mask = torch.from_numpy(mask_array[i]).to(dtype=torch.float32).cuda()
            image_mask = seg_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            H, W = image_mask.shape[-2:]
            stride_h = H // 16
            stride_w = W // 16

            pooled_mask = F.max_pool2d(
                image_mask,
                kernel_size=(stride_h, stride_w),
                stride=(stride_h, stride_w),
            )

            if pooled_mask.shape[-2:] != (16, 16):
                pooled_mask = F.interpolate(pooled_mask, size=(16, 16), mode="nearest")

            mask_patches = pooled_mask.reshape(1,-1,1).long()
            non_zero_num = mask_patches.sum()
            if non_zero_num == 0:
                continue
            image_feature = (tokens * mask_patches).sum(
                dim=1
            ) / non_zero_num
            feature.append(image_feature.squeeze())
        assert len(feature) > 0
        feature = torch.stack(feature)
        torch.save(feature.cpu(), save_path)
