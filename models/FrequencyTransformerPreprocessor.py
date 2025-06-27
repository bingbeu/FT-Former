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

        if dim == 1024: #96 for large model, 64 for small and base model
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 12, 12, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
        if dim ==512:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 24, 24, dtype=torch.float32) * 0.02)
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
        x = torch.permute(x, (0, 3, 1, 2))  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.to(torch.float32)
        groups = 8  # 
        channels_per_group = C // groups
        x_grouped = x.view(B, groups, channels_per_group, H, W)
        xl_list = []
        xh_list = []
        for i in range(groups):
            xl_group, xh_group = self.xfm(x_grouped[:, i]) 
            xl_group = xl_group * self.complex_weight_ll[i*channels_per_group:(i+1)*channels_per_group]
            xl_list.append(xl_group)
            xh_list.append(xh_group[0])
        xl = torch.cat(xl_list, dim=1)  # 合并低频分量
        
        xh = torch.cat(xh_list, dim=1)  
        xh = torch.permute(xh, (5, 0, 2, 3, 4, 1))
        xh = xh.reshape(xh.shape[0], xh.shape[1], 
                                        xh.shape[2], xh.shape[3], 
                                        xh.shape[4], self.num_blocks, self.block_size)
        
        x_real = xh[0]
        x_imag = xh[1]
        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - 
                        self.multiply(x_imag, self.complex_weight_lh_1[1]) + 
                        self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + 
                        self.multiply(x_imag, self.complex_weight_lh_1[0]) + 
                        self.complex_weight_lh_b1[1])
        
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - \
                self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + \
                self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + \
                self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + \
                self.complex_weight_lh_b2[1]

        xh = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        if self.softshrink:
            xh = F.softshrink(xh, lambd=self.softshrink)
        
        xh = xh.reshape(B, xh.shape[1], xh.shape[2], 
                                xh.shape[3], self.hidden_size, xh.shape[6])
        xh = torch.permute(xh, (0, 4, 1, 2, 3, 5)) 
        x = self.ifm((xl, (xh,)))
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(B, N, C)
        
        return x
    # def forward(self, x, H, W):
    #     B, N, C = x.shape 
    #     x = x.view(B, H, W, C)
    #     x=torch.permute(x, (0, 3, 1, 2))
    #     B, C, H, W = x.shape 
    #     x = x.to(torch.float32) 
        
    #     xl,xh = self.xfm(x)
    #     xl = xl * self.complex_weight_ll

    #     xh[0]=torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
    #     xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_blocks, self.block_size)
        
    #     x_real=xh[0][0]
    #     x_imag=xh[0][1]
        
    #     x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
    #     x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        
    #     x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
    #     x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

    #     xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
    #     xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
    #     xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
    #     xh[0]=torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

    #     x = self.ifm((xl,xh))
    #     x=torch.permute(x, (0, 2, 3, 1))
    #     x = x.reshape(B, N, C)# permute is not same as reshape or view
    #     return x
class DWConv(nn.Module):
    def __init__(self, dim=512):
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
# if __name__ == '__main__':
#     block = Block(dim=768, mlp_ratio=8)
#     input_tensor = torch.randn(1, 768, 28, 28) # 假设输入张量大小为 (batch_size, channels, height, width)
#     output_tensor = block(input_tensor, H=28, W=28)
#     print("Output shape:", output_tensor.shape)
class FrequencyTransformerPreprocessor(nn.Module):
    def __init__(self, input_dim, target_seq_len, target_channels):
        """
        Args:
            input_dim (int): 输入特征的通道数（C=512）
            target_seq_len (int): 目标序列长度（N=16）
            target_channels (int): 目标通道数（C=768）
        """
        super(FrequencyTransformerPreprocessor, self).__init__()
        
        # 参数定义
        self.blok = Block(dim=512,mlp_ratio=8)
        self.target_seq_len = target_seq_len
        self.input_dim = input_dim
        self.target_channels = target_channels

        self.attention_pooling = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)

        self.channel = nn.Conv1d(in_channels=input_dim, out_channels=target_channels, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        for _ in range(4):
            x = self.blok(x, H, W)
        B, C, H, W = x.shape
        x = x.view(B, H*W, C)
        query = torch.zeros((B, self.target_seq_len, C), device=x.device, requires_grad=True)  # 可学习的查询向量
        key, value = x, x
        x, _ = self.attention_pooling(query, key, value)  

        x = x.permute(0, 2, 1)  # (B, 16, 512) -> (B, 512, 16)
        x = self.channel(x)  # (B, 32, 16) -> (B, 768, 16)
        x = x.permute(0, 2, 1)  # (B, 768, 16) -> (B, 16, 768)

        return x