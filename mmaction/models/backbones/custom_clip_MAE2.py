from mmengine.model import BaseModule, Sequential
from .custom_MAE2 import VisionTransformer,load_MAE2
from .custom_clip import load,CLIP
from .custom_Visual_Prompt import visual_prompt,clsHead
from mmaction.registry import MODELS
from functools import partial
import torch.nn as nn


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

@MODELS.register_module()
class clip_MAE2(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,


                 ):
        super().__init__()
        self.MAE2 =  VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.MAE2 = load_MAE2(self.MAE2)

        clip, clip_state_dict = load('ViT-B/16', device='cuda', jit=False, T=8, dropout=0.0, emb_dropout=0.0, pretrain=True)
        self.clip_img = ImageCLIP(clip)
        self.fusion_model = visual_prompt('Transf', clip_state_dict, int(16/ 4))
        self.cls_head = clsHead()



    def forward(self, x):
        images_v = x[:, :int(16 / 2), :]
        images_sk = x[:, int(16 / 2):, :]
        b, t, c, h, w = images_v.size()
        image_embedding_v = self.clip_img(images_v.reshape(-1, c, h, w))
        image_embedding_sk = self.MAE2(images_sk)

        image_embedding_v = image_embedding_v.reshape(b, t, -1)
        image_embedding_v = self.fusion_model(image_embedding_v)
        image_embedding = 0.7 * image_embedding_v + 0.3 * image_embedding_sk

        return image_embedding