# some codes from CLIP github(https://github.com/openai/CLIP), from VideoMAE github(https://github.com/MCG-NJU/VideoMAE)
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from collections import OrderedDict
from einops import rearrange
import random
from mmengine.model import BaseModule, Sequential

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Adapter(nn.Module):
    '''
    与AIM一样
    '''

    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, down_dim)
        self.D_fc2 = nn.Linear(down_dim, dim)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        if orig_type == torch.float16:
            ret = super().forward(x)
        elif orig_type == torch.float32:
            ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, T, C, H, W = x.shape
        x = x.view(B, C, T, H, W)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        s2t_q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        s2t_q = s2t_q * self.scale
        attn = (s2t_q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.clip_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))

        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
        self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False)  # 197 tokens(cls+patch) * num_frames
        self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))

        self.t2s_proj = nn.Linear(all_head_dim, dim)

        self.attn_mask = attn_mask

    def s2t_cross_attn(self, s_x, t_x):  # s_x=[n (b t) d], t_x=[b n d]
        B, _, _ = t_x.shape
        t = s_x.shape[1] // t_x.shape[0]
        s_x_pat = s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n b d -> b n d')  # batch -> token
        s_x_pat = s_x_pat + self.clip_space_pos
        t_x = rearrange(t_x, 'b (t n) d -> (b t) n d', t=t)
        t_x = t_x + self.vmae_space_pos
        s2t_q_bias = self.s2t_q_bias
        s2t_kv_bias = self.s2t_kv_bias

        s2t_q = F.linear(input=t_x, weight=self.s2t_q.weight, bias=s2t_q_bias)
        s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
        s2t_kv = F.linear(input=s_x_pat, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
        s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d', e=2, h=self.num_head)
        s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]

        s2t_q = s2t_q * self.scale
        s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))

        s2t_attn = s2t_attn.softmax(dim=-1)

        t_x = (s2t_attn @ s2t_v)
        t_x = rearrange(t_x, 'b h t d -> b t (h d)')
        t_x = self.t2s_proj(t_x)
        t_x = rearrange(t_x, '(b t) n d -> b (t n) d', b=B)
        return t_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.s2t_cross_attn(s_x, t_x)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.clip_time_pos = nn.Parameter(self.scale * torch.randn((num_frames // 2, dim)))
        self.vmae_time_pos = nn.Parameter(self.scale * torch.randn((num_frames // 2, dim)))

        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False)  # 197 tokens(cls+patch) * num_frames
        self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))

        self.t2s_proj = nn.Linear(all_head_dim, dim)

        self.attn_mask = attn_mask

    def t2s_cross_attn(self, s_x, t_x):  # s_x=[n (b t) d], t_x=[b n d]
        B, _, _ = t_x.shape
        t = s_x.shape[1] // t_x.shape[0]
        s_x_cls, s_x_pat = s_x[0, :, :], s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n (b t) d -> (b n) t d', b=B)  # batch -> token
        s_x_pat = s_x_pat + self.clip_time_pos
        t_x = rearrange(t_x, 'b (t n) d -> (b n) t d', t=t)
        t_x = t_x + self.vmae_time_pos
        t2s_q_bias = self.t2s_q_bias
        t2s_kv_bias = self.t2s_kv_bias

        t2s_q = F.linear(input=s_x_pat, weight=self.t2s_q.weight, bias=t2s_q_bias)
        t2s_q = rearrange(t2s_q, 'b t (h d) -> b h t d', h=self.num_head)
        t2s_kv = F.linear(input=t_x, weight=self.t2s_kv.weight, bias=t2s_kv_bias)
        t2s_kv = rearrange(t2s_kv, 'b t (e h d) -> e b h t d', e=2, h=self.num_head)
        t2s_k, t2s_v = t2s_kv[0], t2s_kv[1]

        t2s_q = t2s_q * self.scale
        t2s_attn = (t2s_q @ t2s_k.transpose(-2, -1))

        t2s_attn = t2s_attn.softmax(dim=-1)

        s_x_pat = (t2s_attn @ t2s_v)
        s_x_pat = rearrange(s_x_pat, 'b h n d -> b n (h d)')
        s_x_pat = self.t2s_proj(s_x_pat)
        s_x_pat = rearrange(s_x_pat, '(b n) t d -> n (b t) d', b=B)
        s_x = torch.cat([s_x_cls.unsqueeze(0), s_x_pat], dim=0)
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)


class Block(nn.Module):

    def __init__(self, dim, num_heads, num_frames=16, mlp_ratio=4., down_ratio=2, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.down_ratio = down_ratio
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()

        ###################################### MHSA code #####################################
        ############################ AIM MHSA ###########################
        self.clip_ln_1 = LayerNorm(dim)
        self.clip_attn = nn.MultiheadAttention(dim, num_heads)

        self.clip_ln_t = LayerNorm(dim)
        self.clip_t_attn = nn.MultiheadAttention(dim, num_heads)

        self.S_Adapter = Adapter(dim)
        ##################################################################

        ############################ VMAE MHSA ###########################
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.T_Adapter = Adapter(dim)
        ##################################################################
        #########################################################################################

        ###################################### Cross attention ####################################
        self.cross_s_down = nn.Linear(dim, dim // self.down_ratio)
        self.cross_t_down = nn.Linear(dim, dim // self.down_ratio)
        self.ln_s_cross = norm_layer(dim // self.down_ratio)
        self.ln_t_cross = norm_layer(dim // self.down_ratio)
        self.t2s_cross = CrossAttentionT2S(dim // self.down_ratio, num_heads, num_frames)
        self.s2t_cross = CrossAttentionS2T(dim // self.down_ratio, num_heads, num_frames)
        self.cross_s_up = nn.Linear(dim // self.down_ratio, dim)
        self.cross_t_up = nn.Linear(dim // self.down_ratio, dim)
        ###########################################################################################

        ###################################### FFN code #########################################

        ############################ AIM FFN ###############################
        self.clip_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        self.clip_ln_2 = LayerNorm(dim)
        self.S_MLP_Adapter = Adapter(dim, skip_connect=False)
        self.attn_mask = None
        #####################################################################

        ############################ VMAE FFN ###############################
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #######################################################################
        #########################################################################################

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def t_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_t_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self,rgb_x, sk_x):
        B = sk_x.shape[0]
        n, bt, _ = rgb_x.shape
        num_frames = bt // B

        ############################ MHSA Forward #############################
        rgb_x_t = rearrange(rgb_x, 'n (b t) d -> t (b n) d', t=num_frames)
        rgb_x_t = self.attention(self.clip_ln_t(rgb_x_t))
        # AIM Temp MHSA
        rgb_x_t = rearrange(rgb_x_t, 't (b n) d -> n (b t) d', n=n)
        rgb_x = rgb_x + rgb_x_t
        # AIM Space MHSA
        rgb_x = rgb_x + self.attention(self.clip_ln_1(rgb_x))

        # VMAE Time MHSA
        sk_x = sk_x + self.attn(self.norm1(sk_x))
        ########################################################################

        ############################ Cross Forward #############################
        n_s_x = self.ln_s_cross(self.cross_s_down(rgb_x))
        n_t_x = self.ln_t_cross(self.cross_t_down(sk_x))
        c_s_x = self.cross_s_up(self.act(self.t2s_cross(n_s_x, n_t_x)))
        c_t_x = self.cross_t_up(self.act(self.s2t_cross(n_s_x, n_t_x)))
        rgb_x = rgb_x + self.drop_path(c_s_x)
        sk_x = sk_x + self.drop_path(c_t_x)
        #########################################################################

        ############################ FFN Forward ##################################
        rgb_xn = self.clip_ln_2(rgb_x)
        rgb_x = rgb_x + self.drop_path(self.clip_mlp(rgb_xn))

        sk_xn = self.norm2(sk_x)
        sk_x = sk_xn + self.drop_path(self.mlp(sk_xn))
        ############################################################################

        return rgb_x, sk_x


class STCrossTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=51,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 down_ratio=2,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 composition=False,
                 pretrained_cfg=None,
                 clip_mae_pretrained = True):
        super().__init__()
        self.num_classes = num_classes

        self.num_frames = all_frames
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.down_ratio = down_ratio
        self.composition = composition
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        scale = embed_dim ** -0.5
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                                    bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        self.clip_ln_pre = LayerNorm(embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, num_frames=self.num_frames, mlp_ratio=mlp_ratio,
                down_ratio=self.down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_layer=i)
            for i in range(depth)])

        self.clip_ln_post = LayerNorm(embed_dim)
        self.vmae_fc_norm = norm_layer(embed_dim)


        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.clip_mae_pretrained = clip_mae_pretrained
        self.init_weights()
        # self._init_adpater_weight()


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


        if self.clip_mae_pretrained:
            self.load_bidir_weights()

        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            else:
                self.apply(_init_weights)


        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def _init_adpater_weight(self):
        for n, m in self.blocks.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
            elif 'up' in n:
                for n2, m2 in m.named_modules():
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'clip_time_pos', 'clip_space_pos', 'vmae_space_pos', 'vmae_time_pos', 'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        rgb_x = x[:, :int(16 / 2), :]
        sk_x = x[:, int(16 / 2):, :]
        ######################## --PRE--  AIM spatial path #########################
        rgb_x = rearrange(rgb_x, 'b t c h w -> (b t) c h w')
        rgb_x = self.clip_conv1(rgb_x)  # shape = [b*t, embed_dim, grid, grid]
        rgb_x = rgb_x.reshape(rgb_x.shape[0], rgb_x.shape[1], -1)  # [b*t, embed_dim, grid**2]
        rgb_x = rgb_x.permute(0, 2, 1)  # shape[b*t, patch_num, embed_dim]
        rgb_x = torch.cat([self.clip_class_embedding.to(rgb_x.dtype) + torch.zeros(rgb_x.shape[0], 1, rgb_x.shape[-1],dtype=rgb_x.dtype, device=rgb_x.device),rgb_x], dim=1)  # shape = [b*t, patch_num + 1, embed_dim]
        rgb_x = rgb_x + self.clip_positional_embedding.to(rgb_x.dtype)
        rgb_x = self.clip_ln_pre(rgb_x)
        #####################################################################

        ######################## PRE  --VMAE2 spatial path #########################
        sk_x = self.patch_embed(x)
        if self.pos_embed is not None:
            sk_x = sk_x + self.pos_embed.expand(B, -1, -1).type_as(sk_x).to(sk_x.device).clone().detach()
        sk_x = self.pos_drop(sk_x)
        #####################################################################
        #region 源代码
        ######################## AIM spatial path #########################
        # s_x = x[:, :, 1::2, :, :]  # pick even frames
        # s_t = s_x.shape[2]
        # s_x = rearrange(sk_x, 'b t c h w -> (b t) c h w')
        # s_x = self.clip_conv1(s_x)  # shape = [*, embeddim, grid, grid]
        # s_x = s_x.reshape(s_x.shape[0], s_x.shape[1], -1)  # [*, embeddim, grid**2]
        # s_x = s_x.permute(0, 2, 1)  # shape[batch, patchnum, embeddim]
        # s_x = torch.cat([self.clip_class_embedding.to(s_x.dtype) + torch.zeros(s_x.shape[0], 1, s_x.shape[-1],
        #                                                                        dtype=s_x.dtype, device=s_x.device),
        #                  s_x], dim=1)
        # s_x = s_x + self.clip_positional_embedding.to(s_x.dtype)
        # s_x = self.clip_ln_pre(s_x)
        #####################################################################

        ######################## VMAE spatial path #########################
        # t_x = self.patch_embed(x)
        #
        # if self.pos_embed is not None:
        #     t_x = t_x + self.pos_embed.expand(B, -1, -1).type_as(t_x).to(t_x.device).clone().detach()
        # t_x = self.pos_drop(t_x)
        #####################################################################
        #endregion

        rgb_x = rgb_x.permute(1, 0, 2)  # NLD -> LND  多头自注意力需要
        for blk in self.blocks:
            rgb_x, sk_x = blk(rgb_x, sk_x)
        rgb_x = rgb_x.permute(1, 0, 2)

        rgb_x = rearrange(rgb_x, '(b t) n d -> b t n d', b=B)
        rgb_x = self.clip_ln_post(rgb_x[:,:,0,:].mean(1))  # all frames cls tokens avg pooling
        sk_x = self.vmae_fc_norm(sk_x.mean(1))  # all patch avg pooling

        return rgb_x, sk_x

    def forward(self, x):
        print(x)

        rgb_x ,sk_x = self.forward_features(x)




        return x

    def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):

        """
           将给定的state_dict加载到模型model中。

           参数:
           - model: 要加载状态的模型。
           - state_dict: 要加载的状态字典。
           - prefix: 用于加载状态时的前缀，默认为空。
           - ignore_missing: 指定忽略的未找到的键名模式，默认为"relative_position_index"。

           说明:
           - 此函数将打印出未初始化的权重、未使用的权重和被忽略的权重信息。
           - 不返回任何值，但会更新模型的内部状态。
           """

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))

    def load_bidir_weights(self):
        # 从本地加载MAE2检查点
        MAE_Path = 'D:\Mxd\mmaction\mmaction2\VideoMAEV2_vit_b_k710_dl_from_giant.pth'
        MAE2_checkpoint = torch.load(MAE_Path, map_location='cpu')
        print("Load VideoMAE ckpt from %s" % MAE_Path)
        checkpoint_model = None

        # 从本地加载CLIP检查点
        CLIP_Path ='D:\Mxd\mmaction\mmaction2\ViT-B-16.pt'
        clip_checkpoint = torch.load(CLIP_Path, map_location='cpu')
        print("Load CLIP ckpt from %s" % CLIP_Path)
        checkpoint_clip = clip_checkpoint.visual.state_dict()

        for model_key in ['module']:
            if model_key in MAE2_checkpoint:
                checkpoint_model = MAE2_checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = MAE2_checkpoint

        # 清理与目标模型不兼容的权重
        state_dict = self.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        # 重命名和调整检查点键，以适应模型结构
        all_keys = list(checkpoint_model.keys())
        clip_all_keys = list(checkpoint_clip.keys())
        new_dict = OrderedDict()

        # 加载MAE2权重
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]

        # 加载clip权重
        for key in clip_all_keys:
            if key.startswith('transformer.'):
                if key[23] == '.':
                    new_dict['blocks.' + key[22] + '.clip_' + key[24:]] = checkpoint_clip[key]
                    keys = key[24:28]
                    if key[24:28] == "ln_1":
                        new_dict['blocks.' + key[22] + '.clip_' + "ln_t" + key[28:] ] = checkpoint_clip[key]
                    elif key[24:28] == "attn":
                        new_dict['blocks.' + key[22] + '.clip_' + "t_attn" + key[28:]] = checkpoint_clip[key]

                else:  # layer10 ~ 11 process
                    new_dict['blocks.' + key[22:24] + '.clip_' + key[25:]] = checkpoint_clip[key]
            else:
                new_dict['clip_' + key] = checkpoint_clip[key]

        checkpoint_model = new_dict


        #—————————————————— 预训练模型没有采用学习的位置嵌入，故暂不考虑  ——————————————————————————
        #region 插值处理位置嵌入，以适应模型的新尺寸
        # if 'pos_embed' in checkpoint_model:
        #     pos_embed_checkpoint = checkpoint_model['pos_embed']
        #     embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        #     num_patches = model.patch_embed.num_patches  #
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1
        #
        #     # height (== width) for the checkpoint position embedding
        #     orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
        #                 args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # height (== width) for the new position embedding
        #     new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        #     # class_token and dist_token are kept unchanged
        #     if orig_size != new_size:
        #         print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #         # only the position tokens are interpolated
        #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #         # B, L, C -> BT, H, W, C -> BT, C, H, W
        #         pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size,
        #                                         embedding_size)
        #         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        #         pos_tokens = torch.nn.functional.interpolate(
        #             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        #         # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size,
        #                                                             new_size, new_size, embedding_size)
        #         pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #         checkpoint_model['pos_embed'] = new_pos_embed
        #endregion

        self.load_state_dict(checkpoint_model)


if __name__ == '__main__':
    model = STCrossTransformer()
    imgs = tensor = torch.randn(8,16,3,224,224)
    out = model(imgs)
    keys = model.state_dict().keys()
    print(model.state_dict().keys())
