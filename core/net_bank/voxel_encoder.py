import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
import spconv.pytorch as spconv

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))
    
class VoxelCondEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparse_conv_net = spconv.SparseSequential(
            spconv.SparseConv3d(input_dim, 4, 3, 2, 1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            spconv.SparseConv3d(4, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            spconv.SparseConv3d(16, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, output_dim, 3, 2, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        self.point_embed = PointEmbed(dim=output_dim)
    
        self.ff_128_1 = nn.Linear(128, 1)
        self.ff_512_128 = nn.Linear(512, 128)
        self.ff_256_128 = nn.Linear(256, 128)
    def forward(self, cond):
        x = cond["cond"]
        c = cond["centers"] # [B*T, 512, 3]
        x_c = self.point_embed(c) # [B, 512, 512]
        x_c = self.ff_128_1(x_c) # [B, 512, 1]
        x_c = x_c.squeeze(-1)
        x_c = self.ff_512_128(x_c) # [B, 128]
        x = self.sparse_conv_net(x)
        x = x.dense().view(-1, self.output_dim)

        x = x + x_c # residual connection
        # x_cond = torch.cat([x, x_c], dim=1)
        # x_cond = self.ff_256_128(x_cond) + x
        
        return x
