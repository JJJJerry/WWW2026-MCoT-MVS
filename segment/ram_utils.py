import torch.nn.functional as F
import torch
from torch.nn.functional import relu, sigmoid
from torch import Tensor
from ram.models import ram_plus
from ram import get_transform

class RAMModel:
    def __init__(self,pretrained,tag_file_path,device):
        self.device=device
        self.model,self.transform=self.load_ram_model(pretrained)
        with open(tag_file_path, "r", encoding="utf-8") as f:
            self.tag_list=[line.strip() for line in f]

    def load_ram_model(self,pretrained):
        image_size=384
        transform = get_transform(image_size=image_size)
        #######load model
        ram_model = ram_plus(pretrained=pretrained,
                    image_size=image_size,
                    vit='swin_l')
        ram_model.eval()
        ram_model = ram_model.to(self.device)
        return ram_model,transform    
    
    @torch.no_grad()
    def forward_ram_plus(self,imgs: Tensor) -> Tensor:
        image_embeds = self.model.image_proj(self.model.visual_encoder(imgs.to(self.device)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.model.label_embed.shape[0] / self.model.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = self.model.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds @ self.model.label_embed.t())
        logits_per_image = logits_per_image.view(bs, -1,des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.model.num_class, 512).cuda()
        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.model.num_class, 512).cuda()
        for i in range(bs):
            reshaped_value = self.model.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = relu(self.model.wordvec_proj(label_embed_reweight))

        tagging_embed, _ = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )
        return sigmoid(self.model.fc(tagging_embed).squeeze(-1))
    
    @torch.inference_mode()
    def get_batch_tags(self,pil_image_list):
        ram_input_images=[]
        for pil_image in pil_image_list:
            transformed_image = self.transform(pil_image).to(self.device)
            ram_input_images.append(transformed_image)
        ram_input_images=torch.stack(ram_input_images)
        out=self.forward_ram_plus(ram_input_images).cpu()
        pred_tags=[]
        for score_list in out.tolist():
            tags=[]
            for i,score in enumerate(score_list):
                if score>=0.7 and self.tag_list[i]:
                    tags.append(self.tag_list[i])
            pred_tags.append(tags)
        return pred_tags
    