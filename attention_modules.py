import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. 浅层空间注意力 (Spatial Attention)
# 作用：细化输入的语义掩码，解决边界伪影，提供软引导。
# ---------------------------------------------------------
class SpatialMaskAttention(nn.Module):
    def __init__(self, channels=1, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask):
        # mask: [B, 1, H, W]
        attn_map = self.sigmoid(self.conv(mask))
        return mask * attn_map


# ---------------------------------------------------------
# 2. 潜在全局自注意力 (Global Self-Attention)
# 作用：在 U-Net 瓶颈层捕捉长距离依赖，保证人脸全局结构协调。
# ---------------------------------------------------------
class LatentGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x input: [B, C, H, W] -> flatten to [B, N, C]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        qkv = self.to_qkv(x_flat).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)

        # Restore shape: [B, C, H, W]
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------
# 3. 身份交叉注意力 (Identity Cross-Attention)
# 作用：将目标身份特征动态注入到生成过程，提升攻击迁移性。
# ---------------------------------------------------------
class IdentityCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)  # Source/Target Identity
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)  # Latent Features
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)  # Latent Features

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: Identity Embedding [B, 1, Dim]
        # context: Latent Feature Map [B, N, Dim]
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(q.shape[0], -1, h, q.shape[-1] // h).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], -1, h, k.shape[-1] // h).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], -1, h, v.shape[-1] // h).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], -1, h * (out.shape[-1]))
        return self.to_out(out)
