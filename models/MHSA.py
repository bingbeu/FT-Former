import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .meta_encoder import ResNormLayer
from .Wavelet import Block
from .FrequencyTransformerPreprocessor import FrequencyTransformerPreprocessor
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x      
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Relative_Attention(nn.Module):
    def __init__(self,dim,img_size,extra_token_num=1,num_heads=8,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.extra_token_num = extra_token_num
        head_dim = dim // num_heads
        self.img_size = img_size # h,w
        self.scale = qk_scale or head_dim ** -0.5
         # define a parameter table of relative position bias,add cls_token bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * img_size[0] - 1) * (2 * img_size[1] - 1) + 1, num_heads))  # 2*h-1 * 2*w-1 + 1, nH
        
        # get pair-wise relative position index for each token
        coords_h = torch.arange(self.img_size[0])
        coords_w = torch.arange(self.img_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += self.img_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.img_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.img_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # h*w, h*w
        relative_position_index = F.pad(relative_position_index,(extra_token_num,0,extra_token_num,0))
        relative_position_index = relative_position_index.long()
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x,):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.img_size[0] * self.img_size[1] + self.extra_token_num, self.img_size[0] * self.img_size[1] + self.extra_token_num, -1)  # h*w+1,h*w+1,nH
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w+1, h*w+1
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W        
class MHSABlock(nn.Module):
    def __init__(self, input_dim, output_dim,image_size, stride, num_heads,extra_token_num=1,attn_embed_dims=[512,1024],meta_dims=[ 768, ],mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if stride != 1:
            self.patch_embed = OverlapPatchEmbed(patch_size=3,stride=stride,in_chans=input_dim,embed_dim=output_dim)
            # print('stride:',stride)
            # print('input_dim:',input_dim)
            # print('output_dim:',output_dim)
            self.img_size = image_size//2
        else:
            self.patch_embed = None
            self.img_size = image_size
        self.img_size = to_2tuple(self.img_size)
        self.meta_dims = meta_dims
        self.freq_dims = meta_dims
        self.attn_embed_dims = attn_embed_dims
        self.norm1 = norm_layer(output_dim)
        self.attn = Relative_Attention(
            output_dim,self.img_size, extra_token_num=extra_token_num,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(output_dim)
        mlp_hidden_dim = int(output_dim * mlp_ratio)
        self.FrequencyTransformerPreprocessor = FrequencyTransformerPreprocessor(input_dim=512, target_seq_len=16, target_channels=768)
        self.mlp = Mlp(in_features=output_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        for ind,meta_dim in enumerate(meta_dims):
                freq_head_1 = nn.Sequential(
                                        nn.Linear(meta_dim, attn_embed_dims[0]),
                                        nn.Softplus(),
                                        nn.LayerNorm(attn_embed_dims[0]),
                                        ResNormLayer(attn_embed_dims[0])
                                        ) if meta_dim > 0 else nn.Identity()

                freq_head_2 = nn.Sequential(
                                        nn.Linear(768, attn_embed_dims[1]),
                                        nn.Softplus(),
                                        nn.LayerNorm(attn_embed_dims[1]),
                                        ResNormLayer(attn_embed_dims[1])
                                        ) if meta_dim > 0 else nn.Identity()

                setattr(self, f"freq_{ind+1}_head_1", freq_head_1)
                setattr(self, f"freq_{ind+1}_head_2", freq_head_2)

    def forward(self, x, H, W, extra_tokens=None, frequency=None, extra_tokens_type=None, is_stage_3=False):
    # 每次都需要执行 patch_embed 处理
        if self.patch_embed is not None:
            x, _, _ = self.patch_embed(x)
            #print(f"主每次都需要执行函数u{x.shape}")
            B, N, C = x.shape
            '''
            # 用于存储 freq_1 和 freq_2
            stored_freqs = [None, None]  # 用列表存储 freq_1 和 freq_2

            # 只有在 is_stage_3 为 True 时，才进行 freq_1 和 freq_2 的生成
            if is_stage_3:
                H = W = int(math.sqrt(N))  # 确保 H 和 W 的尺寸一致
                y = x.view(B, C, H, W)  # reshape
                y = self.FrequencyTransformerPreprocessor(y)  # 预处理
                print(f"y.shape: {y.shape}")

                assert frequency is not None, 'meta is None'
                if len(self.meta_dims) > 1:
                    metas = torch.split(frequency, self.meta_dims, dim=1)
                else:
                    metas = (frequency,)

                B, N, C = y.shape  # 确认 y 的形状

                # 将 y 投影到不同的目标嵌入维度
                freq_head_1 = getattr(self, "freq_1_head_1")
                freq_head_2 = getattr(self, "freq_1_head_2")

                freq_1 = freq_head_1(y)
                freq_1 = freq_1.reshape(B, -1, self.attn_embed_dims[0])
                freq_2 = freq_head_2(y)
                freq_2 = freq_2.reshape(B, -1, self.attn_embed_dims[1])

                # 存储到 stored_freqs 列表中
                stored_freqs[0] = freq_1  # 存储 freq_1
                stored_freqs[1] = freq_2  # 存储 freq_2

                print(f"stored_freqs: {stored_freqs}")

            # 根据 extra_tokens_type 判断是否添加 freq_1 或 freq_2
            if extra_tokens_type == "extra_tokens_1":
                if stored_freqs[0] is not None:
                    extra_tokens.append(stored_freqs[0])
                    print(f"extra_tokens (added freq_1): {extra_tokens}")
                else:
                    raise ValueError("extra_tokens_type == 'extra_tokens_1' but freq_1 is None.")

            elif extra_tokens_type == "extra_tokens_2":
                if stored_freqs[1] is not None:
                    extra_tokens.append(stored_freqs[1])
                    print(f"extra_tokens (added freq_2): {extra_tokens}")
                else:
                    raise ValueError("extra_tokens_type == 'extra_tokens_2' but freq_2 is None.")
                '''
        # 只有当 extra_tokens 存在时，才进行扩展和拼接
        if extra_tokens is not None:
            extra_tokens = [token.expand(x.shape[0], -1, -1) for token in extra_tokens]
            extra_tokens.append(x)
            x = torch.cat(extra_tokens, dim=1)

        # 继续进行后续的操作
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H // 2, W // 2))
        return x
