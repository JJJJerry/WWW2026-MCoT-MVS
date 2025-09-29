import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union
from PIL import Image
import open_clip

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)
class Combiner(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):

        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def combine_features(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:

        text_projected_features = self.dropout1(
            F.relu(self.text_projection_layer(text_features))
        )
        image_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(image_features))
        )

        raw_combined_features = torch.cat(
            (text_projected_features, image_projected_features), -1
        )
        combined_features = self.dropout3(
            F.relu(self.combiner_layer(raw_combined_features))
        )
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * image_features
        )

        return output

class TriCombiner(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        super(TriCombiner, self).__init__()

        self.text_proj = nn.Linear(clip_feature_dim, projection_dim)
        self.image_proj = nn.Linear(clip_feature_dim, projection_dim)
        self.seg_proj = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout_comb = nn.Dropout(0.5)

     
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

    def combine_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        seg_features: torch.Tensor,
    ) -> torch.Tensor:

        text_proj = self.dropout1(F.relu(self.text_proj(text_features)))
        image_proj = self.dropout2(F.relu(self.image_proj(image_features)))
        seg_proj = self.dropout3(F.relu(self.seg_proj(seg_features)))

        concat_features = torch.cat([text_proj, image_proj, seg_proj], dim=-1)

        weights = self.dynamic_scalar(concat_features)  # shape: (batch_size, 3)
        w_text, w_image, w_seg = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        combined = self.dropout_comb(F.relu(self.combiner_layer(concat_features)))
        fused_output = self.output_layer(combined)

        output = (
            fused_output
            + w_text * text_features
            + w_image * image_features
            + w_seg * seg_features
        )

        return output

class IdentityNonlinearBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim),
        )
        self.init_identity()

    def init_identity(self):

        nn.init.zeros_(self.block[0].weight)
        nn.init.zeros_(self.block[0].bias)
        nn.init.zeros_(self.block[2].weight)
        nn.init.zeros_(self.block[2].bias)

    def forward(self, x):
        return x + self.block(x)
    
class CIRModel(nn.Module):
    def __init__(self, model_name="ViT-H-14",loss_weight=10.0):
        super(CIRModel, self).__init__()

        self.clip,self.preprocess_train,self.preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        self.tri_combiner_1 = TriCombiner(
            clip_feature_dim=1024, projection_dim=640 * 4, hidden_dim=640 * 8
        )
        self.tri_combiner_2 = TriCombiner(
            clip_feature_dim=1024, projection_dim=640 * 4, hidden_dim=640 * 8
        )
        self.combiner = Combiner(clip_feature_dim=1024, projection_dim=640 * 4, hidden_dim=640 * 8)
        self.patch_size = 24
        self.seg_feature_proj = IdentityNonlinearBlock(input_dim=1024, hidden_dim=2048)
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((loss_weight,)))

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
    def get_visual(self,x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)
        x = self.clip.visual.transformer(x)

        if self.clip.visual.attn_pool is not None:
            if self.clip.visual.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.clip.visual.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.clip.visual.attn_pool(x)
                if self.clip.visual.attn_pool_type == 'parallel':
                    pooled = self.clip.visual.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.clip.visual.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
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

    def extract_query(
        self,
        modified_text: Union[str, List[str]],
        reference_image: Union[Image.Image, List[Image.Image]],
        reference_seg_feature_list: List[torch.Tensor],
        llm_info_dict: dict,
    ) -> torch.Tensor:

        text_feature = self.encode_text(modified_text)
        target_text_feature = self.encode_text(llm_info_dict['target'])
        B, D = text_feature.shape
        x = self.preprocess_image(reference_image)
        image_feature_cls, patch_feature = self.get_visual(x)

        retained_attention_map = self.get_attention_map(
            patch_feature, llm_info_dict["retained"]
        )
        deleted_attention_map = self.get_attention_map(
            patch_feature, llm_info_dict["deleted"]
        )  # (B,576)

        reference_attention_map = 1-(retained_attention_map - deleted_attention_map) # (B,576)

        weighted_patch_feature = patch_feature * reference_attention_map.unsqueeze(2)  # (B,576,768)
        reference_image_feature = (
            weighted_patch_feature.mean(dim=1) * 0.5 + image_feature_cls * 0.5
        )

        ## reference seg feature
        retained_text_feature = self.encode_text(llm_info_dict["retained"])  # (B,D)
        deleted_text_feature = self.encode_text(llm_info_dict["deleted"])  # (B,D)
        reference_seg_features = []
        for i in range(B):
            n = reference_seg_feature_list[i].shape[0]
            seg_feature = self.seg_feature_proj(reference_seg_feature_list[i])  # (N,D)

            retained_text_feature_expanded = (
                retained_text_feature[i].reshape(1, -1).expand(n, D)
            )  # (N,D)
            retained_seg_score = 1-F.cosine_similarity(
                seg_feature, retained_text_feature_expanded
            )  # (N,1)
            retained_seg_score = (retained_seg_score-retained_seg_score.min())/(retained_seg_score.max()-retained_seg_score.min()+1e-2)
            deleted_text_feature_expanded = (
                deleted_text_feature[i].reshape(1, -1).expand(n, D)
            )  # (N,D)
            deleted_seg_score = 1-F.cosine_similarity(
                seg_feature, deleted_text_feature_expanded
            )  # (N,1)
            deleted_seg_score = (deleted_seg_score-deleted_seg_score.min())/(deleted_seg_score.max()-deleted_seg_score.min()+1e-2)
            seg_score = (retained_seg_score - deleted_seg_score).reshape(n, 1)  # (N,1)
            seg_feature = (seg_feature * seg_score).mean(dim=0)  # (D)
            reference_seg_features.append(seg_feature)

        reference_seg_features = torch.stack(reference_seg_features)

        query_feature_1 = self.tri_combiner_1.combine_features(
            reference_image_feature, text_feature, reference_seg_features
        )
        query_feature_2 = self.tri_combiner_2.combine_features(
            reference_image_feature, target_text_feature, reference_seg_features
        )
        query_feature = self.combiner.combine_features(query_feature_1,query_feature_2)
        return F.normalize(query_feature, dim=-1)

    def get_attention_map(self, dense_image_feature: torch.Tensor, texts: List[str]):
        text_feature = F.normalize(self.encode_text(texts), dim=-1)
        dense_image_feature = F.normalize(dense_image_feature, dim=-1)
        attention_map = torch.matmul(
            text_feature.unsqueeze(1), dense_image_feature.transpose(1, 2)
        ).squeeze(
            1
        )  # (B,patch_size**2)
        batch_text_mask = (
            torch.tensor([text != "" for text in texts], dtype=torch.bool)
            .to(attention_map.device)
            .unsqueeze(1)
        )
        attention_map = attention_map * batch_text_mask  # (B,patch_size**2)
        return attention_map

    def extract_target(
        self, target_image: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        target_image_feature = self.encode_image(target_image)
        return F.normalize(target_image_feature, dim=-1)

    def compute_nce_loss(self, query_feature, target_feature):
        logits = self.loss_weight * query_feature @ target_feature.T
        # logits = 100 * query_feature @ target_feature.T
        ground_truth = torch.arange(query_feature.shape[0], dtype=torch.long).cuda()
        nce_loss = self.crossentropy_criterion(logits, ground_truth)
        return nce_loss

    def compute_loss(
        self,
        reference_image,
        modified_text,
        target_image,
        reference_seg_feature_list,
        llm_info_dict,
    ):
        
        query_feature = self.extract_query(
            modified_text=modified_text,
            reference_image=reference_image,
            reference_seg_feature_list = reference_seg_feature_list,
            llm_info_dict=llm_info_dict,
        )
        target_feature = self.extract_target(target_image=target_image)
        nce_loss = self.compute_nce_loss(
            query_feature=query_feature, target_feature=target_feature
        )
        return {
            "nce_loss": nce_loss,
        }
