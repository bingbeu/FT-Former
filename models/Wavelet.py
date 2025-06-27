import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SVT_channel_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 64: #[b, 64,56,56]
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        if dim ==256: #[b, 128,28,28]
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 48, 48, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        if dim == 96: #96 for large model, 64 for small and base model
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
        if dim ==192:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 48, 48, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink =0.0 

    def multiply(self, input, weights):
    # 确保输入和权重都是 float32 类型
        input = input.to(torch.float32)  # 将 input 转换为 float32
        weights = weights.to(torch.float32)  # 将 weights 转换为 float32
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape 
        x = x.view(B, H, W, C)
        x=torch.permute(x, (0, 3, 1, 2))
        B, C, H, W = x.shape 
        x = x.to(torch.float32) 
        
        xl,xh = self.xfm(x)
        xl = xl * self.complex_weight_ll

        xh[0]=torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_blocks, self.block_size)
        
        x_real=xh[0][0]
        x_imag=xh[0][1]
        
        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        xh[0]=torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

        x = self.ifm((xl,xh))
        x=torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(B, N, C)# permute is not same as reshape or view
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


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, 
        dim,
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = SVT_channel_mixing (dim)
            # self.attn = SVT_token_mixing (dim)
            # self.attn = SVT_channel_token_mixing (dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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

    def forward(self, x, H, W):
        #print(x.shape)
        B, C, H, W = x.shape
        x = x.view(B, H*W, C)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.view(B, C, H, W)
        return x