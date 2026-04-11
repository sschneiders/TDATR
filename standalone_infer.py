#!/usr/bin/env python3
"""
Standalone CPU inference for TDATR table document understanding model.

Loads weights from model.pt and runs the full pipeline:
  image → Donut/Swin encoder → conv projections → IPT decoder with cross-attention → autoregressive HTML generation

Usage:
    python standalone_infer.py --image path/to/table.jpg --checkpoint model.pt [--max_new_tokens 1024]

Dependencies: torch, timm, PIL, sentencepiece, numpy, opencv-python
"""
import argparse
import re
import sys
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import cv2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Custom Donut/Swin Image Encoder (matches checkpoint key format)
# ---------------------------------------------------------------------------
def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    x: (B, H, W, C) -> (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition: (B*num_windows, ws, ws, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.q_k_v_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Relative position bias table: (2*ws-1)^2 x num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
        coords_flatten = coords.view(2, -1)  # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws*ws, ws*ws)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.q_k_v_proj(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B_, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, H)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (H, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.out_proj(x)


class SwinMLP(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SwinWindowAttention(embed_dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = SwinMLP(embed_dim, mlp_hidden)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.attn.window_size)  # (B*num_win, ws, ws, C)
        x_windows = x_windows.view(-1, self.attn.window_size * self.attn.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.attn.window_size, self.attn.window_size, C)
        x = window_reverse(attn_windows, self.attn.window_size, H, W)  # (B, H, W, C)
        x = x.view(B, H * W, C)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, mlp_ratio)
            for _ in range(depth)
        ])
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        before_downsample = x
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, before_downsample, H, W


class DonutEncoder(nn.Module):
    def __init__(self, img_size=2048, patch_size=4, in_chans=3, embed_dim=256,
                 depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=8,
                 mlp_ratio=4.0):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed.norm = nn.LayerNorm(embed_dim)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = SwinStage(
                dim=num_features[i], depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                downsample=(i < self.num_layers - 1)
            )
            self.layers.append(layer)

        self.norm_layers = nn.ModuleList([nn.LayerNorm(num_features[i]) for i in range(self.num_layers)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """x: (B, 3, H, W) -> list of (B, H_i, W_i, C_i) per stage."""
        B, C, H, W = x.shape
        x = self.patch_embed.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x = self.patch_embed.norm(x)

        Hp, Wp = H // self.patch_embed.proj.kernel_size[0], W // self.patch_embed.proj.kernel_size[1]
        outputs = []
        for i, layer in enumerate(self.layers):
            x, before_downsample, Hp, Wp = layer(x, Hp, Wp)
            outputs.append(before_downsample)
        return outputs


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------
class RoPE(nn.Module):
    """Standard Rotary Position Embedding for self-attention."""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        """
        q, k: (T, B, H, D)
        Returns: q_rot, k_rot
        """
        T = q.shape[0]
        t = torch.arange(T, dtype=torch.float32, device=q.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, D)
        cos = emb.cos().unsqueeze(1).unsqueeze(1)  # (T, 1, 1, D)
        sin = emb.sin().unsqueeze(1).unsqueeze(1)  # (T, 1, 1, D)

        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


# ---------------------------------------------------------------------------
# IPT Transformer Layer (decoder-style with pre-LN)
# ---------------------------------------------------------------------------
class IPTSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.embed_dim = embed_dim
        self.q_k_v_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, attention_mask=None, past_k=None, past_v=None, use_cache=False):
        """
        x: (T, B, D) — current tokens (T=1 during generation)
        past_k, past_v: (T_past, B, H, head_dim) or None
        Returns: output (T, B, D), k_cache (T_total, B, H, head_dim), v_cache (T_total, B, H, head_dim)
        """
        T, B, D = x.shape
        qkv = self.q_k_v_proj(x)  # (T, B, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (T, B, H, head_dim)
        q = q.view(T, B, self.num_heads, self.head_dim)
        k = k.view(T, B, self.num_heads, self.head_dim)
        v = v.view(T, B, self.num_heads, self.head_dim)

        # Apply RoPE — when using cache, RoPE position must account for past length
        if past_k is not None:
            past_len = past_k.shape[0]
            # Temporarily adjust RoPE to start from past_len
            orig_inv_freq = self.rope.inv_freq
            t = torch.arange(past_len, past_len + T, dtype=torch.float32, device=x.device)
            freqs = torch.einsum("i,j->ij", t, orig_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(1).unsqueeze(0)  # (T, 1, 1, D)
            sin = emb.sin().unsqueeze(1).unsqueeze(0)
            def rotate_half(hx):
                hx1, hx2 = torch.chunk(hx, 2, dim=-1)
                return torch.cat((-hx2, hx1), dim=-1)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:
            q, k = self.rope(q, k)

        # Concat with past cache
        if past_k is not None:
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)

        T_total = k.shape[0]

        # Transpose for attention: (B, H, T, D)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

        attn_probs = F.softmax(attn_weights.float(), dim=-1).to(v.dtype)
        context = torch.matmul(attn_probs, v)

        # Reshape back: (T, B, D)
        context = context.permute(2, 0, 1, 3).contiguous().view(T, B, D)
        output = self.out_proj(context)

        if use_cache:
            # Return caches in (T_total, B, H, head_dim) format
            k_cache = k.permute(2, 0, 1, 3).contiguous()  # (T_total, B, H, head_dim)
            v_cache = v.permute(2, 0, 1, 3).contiguous()
            return output, k_cache, v_cache
        return output


class IPTMLP(nn.Module):
    def __init__(self, embed_dim, mlp_embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_embed_dim, bias=True)
        self.fc2 = nn.Linear(mlp_embed_dim, embed_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class IPTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_embed_dim):
        super().__init__()
        self.attention = IPTSelfAttention(embed_dim, num_heads)
        self.mlp = IPTMLP(embed_dim, mlp_embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attention_mask=None, past_k=None, past_v=None, use_cache=False):
        # Pre-LN self-attention
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_out = self.attention(hidden_states, attention_mask, past_k, past_v, use_cache)
        if use_cache:
            hidden_states, new_k, new_v = attn_out
        else:
            hidden_states = attn_out
        hidden_states = residual + hidden_states

        # Pre-LN MLP
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, new_k, new_v
        return hidden_states


# ---------------------------------------------------------------------------
# Cross-Attention (injects image features into text decoder)
# ---------------------------------------------------------------------------
class CrossAttnFFN(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        # Match checkpoint key structure: attention.fc
        self.attention = nn.ModuleDict({
            "fc": nn.Linear(embed_dim, embed_dim, bias=True),
        })
        # FFN sub-network matching checkpoint keys (proj.fc1, proj.fc2)
        self.proj = nn.ModuleDict({
            "fc1": nn.Linear(embed_dim, embed_dim * 2),
            "fc2": nn.Linear(embed_dim * 2, embed_dim),
        })

    def forward(self, query, kv, attn_mask=None):
        """
        query: (Tq, B, D) - text hidden states
        kv: (Tk, B, D) - image features (key/value)
        """
        residual = query
        q = self.proj_q(query)

        Tq, B, D = q.shape
        Tk = kv.shape[0]

        q = q.view(Tq, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = kv.view(Tk, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = kv.view(Tk, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)
        context = torch.matmul(attn, v)
        # context: (B, H, Tq, D) -> (B, Tq, D) -> (Tq, B, D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, Tq, D)

        # attention.fc output projection
        context = self.attention["fc"](context)

        context = torch.clamp(context, -32.0, 32.0)
        # proj: (B, L, D) -> fc1 -> relu -> fc2
        context = F.relu(self.proj["fc1"](context))
        context = self.proj["fc2"](context)
        context = context.permute(1, 0, 2)  # (Tq, B, D)

        out = residual + context
        out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Full Standalone Model
# ---------------------------------------------------------------------------
class StandaloneTDATR(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=32, num_layers=6, mlp_embed_dim=11264,
                 vocab_size=60000, cross_attn_layers=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.cross_attn_layers = cross_attn_layers or [1, 3, 5]

        # Donut/Swin backbone
        self.donut_model = self._build_donut()
        # norm_layers are now inside donut_model, matching checkpoint keys

        # Image projection convolutions
        self.conv_feat32 = nn.Conv2d(2048, 2048, 1, 1, 0)
        self.conv_feat16 = nn.Conv2d(1024, 2048, 3, 1, 1)
        self.conv_proj = nn.Conv2d(2048, 2048, 3, 1, 1)

        # Positional embedding for image features (160x160 grid = 25600 tokens)
        self.pos_embed_kv_16 = nn.Parameter(
            torch.from_numpy(self._get_2d_sincos_pos_embed(2048, 160)).float()
        )
        # Text positional embedding (not used in forward, but loaded for completeness)
        # self.pos_embed_text = nn.Parameter(...)  # Registered in checkpoint but unused

        # Token embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.bbox_embedding = nn.Embedding(4556, embed_dim)

        # IPT decoder layers
        self.decoder_layers = nn.ModuleList([
            IPTDecoderLayer(embed_dim, num_heads, mlp_embed_dim)
            for _ in range(num_layers)
        ])

        # Cross-attention modules
        self.cross_attns = nn.ModuleDict({
            str(i): CrossAttnFFN(embed_dim, num_heads)
            for i in self.cross_attn_layers
        })

        # Final layer norm
        self.final_layernorm = nn.LayerNorm(embed_dim)

        # Vocabulary size info
        self.ipt_vocab_max = vocab_size  # 60000
        self.bbox_vocab_max = vocab_size + 4556  # 64556
        self.ocr_vocab_max = vocab_size + 4556 + 2  # 64558

        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE for each decoder layer's self-attention."""
        for layer in self.decoder_layers:
            layer.attention.rope = RoPE(self.embed_dim // self.num_heads)

    def _build_donut(self):
        """Build Donut/Swin backbone matching checkpoint key format."""
        return DonutEncoder(
            img_size=2048, patch_size=4, in_chans=3, embed_dim=256,
            depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=8,
            mlp_ratio=4.0,
        )

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim, grid_size):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
        emb_h = StandaloneTDATR._get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
        emb_w = StandaloneTDATR._get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        return emb

    def _get_abs_pos_rectangle(self, abs_pos, feat_shape):
        """Crop positional embedding to match feature map shape."""
        h, w = feat_shape
        src_size = int(math.sqrt(abs_pos.size(0)))
        # abs_pos: (src_size*src_size, D) -> (1, src_size, src_size, D) -> (1, D, src_size, src_size)
        abs_pos_2d = abs_pos.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
        abs_pos_2d = abs_pos_2d[:, :, :w, :h]
        result = abs_pos_2d.permute(0, 3, 2, 1).flatten(0, 2)
        return result

    def encode_image(self, images):
        """
        images: (B, 3, H, W) normalized tensor
        Returns: image_embeds (B, L, D) for cross-attention KV
        """
        B, C, H, W = images.shape

        # Forward through Donut encoder — returns list of per-stage outputs
        stage_outputs = self.donut_model(images)

        # Apply norm and reshape: (B, L, C) -> (B, C, H, W)
        feat_maps = []
        feat_sizes = []
        for i, out in enumerate(stage_outputs):
            out = self.donut_model.norm_layers[i](out)  # norm applied on (B, L, C)
            C_i = out.shape[-1]
            H_i = int(math.sqrt(out.shape[1]))
            if H_i * H_i != out.shape[1]:
                # Non-square: use stride from previous stages
                patch_h = H // 4 // (2 ** i)
                patch_w = W // 4 // (2 ** i)
                H_i, W_i = patch_h, patch_w
            else:
                W_i = H_i
            feat_maps.append(out.permute(0, 2, 1).view(B, C_i, H_i, W_i))
            feat_sizes.append((H_i, W_i))

        # feat_maps[2] = stage3 output (dim=1024), feat_maps[3] = stage4 output (dim=2048)
        feat_32 = self.conv_feat32(feat_maps[3])  # (B, 2048, H32, W32)
        feat_16 = self.conv_feat16(feat_maps[2])  # (B, 2048, H16, W16)
        _, _, H16, W16 = feat_16.shape
        feat_32_up = F.interpolate(feat_32, size=(H16, W16), mode="bilinear", align_corners=False)
        feat_merged = feat_32_up + feat_16

        # Final projection
        feat_proj = self.conv_proj(feat_merged)  # (B, 2048, H, W)
        feat_shape = feat_proj.shape[2:]

        # Reshape to (B, L, D)
        image_embeds = feat_proj.view(B, 2048, -1).permute(0, 2, 1)

        # Add positional embedding
        pos_embed = self._get_abs_pos_rectangle(self.pos_embed_kv_16, feat_shape)
        image_embeds = image_embeds + pos_embed.unsqueeze(0)

        return image_embeds

    def embed_tokens(self, tokens):
        """
        tokens: (B, L) integer token ids
        Returns: (L, B, D) embeddings
        """
        ipt_mask = tokens < self.ipt_vocab_max
        bbox_mask = (tokens >= self.ipt_vocab_max) & (tokens < self.bbox_vocab_max)
        ocr_mask = tokens >= self.bbox_vocab_max

        ipt_tokens = tokens.clone()
        bbox_tokens = tokens.clone()
        ocr_tokens = tokens.clone()

        ipt_tokens[~ipt_mask] = 0
        bbox_tokens[~bbox_mask] = self.ipt_vocab_max
        bbox_tokens -= self.ipt_vocab_max
        ocr_tokens[~ocr_mask] = self.bbox_vocab_max
        ocr_tokens -= self.bbox_vocab_max

        ipt_embed = self.word_embeddings(ipt_tokens)
        bbox_embed = self.bbox_embedding(bbox_tokens)
        ocr_embed = torch.zeros_like(ipt_embed)  # No OCR embedding weights needed for basic inference

        output = (ipt_embed * ipt_mask.unsqueeze(-1).float() +
                  bbox_embed * bbox_mask.unsqueeze(-1).float() +
                  ocr_embed * ocr_mask.unsqueeze(-1).float())
        return output.permute(1, 0, 2)  # (L, B, D)

    def get_logits(self, hidden_states):
        """hidden_states: (B, L, D) -> (B, L, vocab_size)"""
        return F.linear(hidden_states, self.word_embeddings.weight)

    def decode_step(self, hidden_states, img_kv, attention_mask=None,
                    self_k_cache=None, self_v_cache=None, use_cache=False):
        """
        Run through all decoder layers with cross-attention injection.
        hidden_states: (L, B, D)
        img_kv: (Tk, B, D) image features for cross-attention
        Returns: (L, B, D), and optionally updated k/v caches
        """
        new_k_caches = [] if use_cache else None
        new_v_caches = [] if use_cache else None

        for idx, layer in enumerate(self.decoder_layers):
            pk = self_k_cache[idx] if self_k_cache else None
            pv = self_v_cache[idx] if self_v_cache else None
            if idx in self.cross_attn_layers:
                hidden_states = self.cross_attns[str(idx)](hidden_states, img_kv)
            layer_out = layer(hidden_states, attention_mask, pk, pv, use_cache)
            if use_cache:
                hidden_states, nk, nv = layer_out
                new_k_caches.append(nk)
                new_v_caches.append(nv)
            else:
                hidden_states = layer_out
            hidden_states = torch.clamp(hidden_states, -32.0, 32.0)
        hidden_states = self.final_layernorm(hidden_states)

        if use_cache:
            return hidden_states, new_k_caches, new_v_caches
        return hidden_states

    @torch.no_grad()
    def generate(self, image_embeds, prompt_tokens, max_new_tokens=1024, eos_id=None):
        """
        Autoregressive greedy generation with KV caching.
        image_embeds: (B, Limg, D)
        prompt_tokens: (B, Lprompt) token ids
        """
        B = image_embeds.size(0)
        device = image_embeds.device

        # Embed prompt tokens
        prompt_embeds = self.embed_tokens(prompt_tokens)  # (Lp, B, D)
        Lp = prompt_embeds.size(0)

        # Image KV: (Limg, B, D)
        img_kv = image_embeds.permute(1, 0, 2)

        # --- Pre-fill: process all prompt tokens at once ---
        # Causal mask for pre-fill: (1, 1, Lp, Lp)
        prefill_mask = torch.triu(
            torch.ones(Lp, Lp, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        hidden, k_caches, v_caches = self.decode_step(
            prompt_embeds, img_kv,
            attention_mask=prefill_mask,
            use_cache=True
        )

        # Get logits for last position
        last_hidden = hidden[-1:].transpose(0, 1)  # (B, 1, D)
        logits = self.get_logits(last_hidden)[:, -1, :]  # (B, V)
        next_token = logits.argmax(dim=-1)  # (B,)

        generated_tokens = [next_token.item()]
        all_tokens = torch.cat([prompt_tokens, next_token.unsqueeze(1)], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            return all_tokens

        # --- Autoregressive generation with KV cache ---
        for step in range(1, max_new_tokens):
            cur_len = all_tokens.size(1)  # total sequence length including prompt

            # Embed just the new token
            new_embed = self.embed_tokens(next_token.unsqueeze(1))  # (1, B, D)

            # No causal mask needed for single-token generation step
            # (the KV cache already contains all prior positions with causal masking applied during pre-fill)
            hidden, k_caches, v_caches = self.decode_step(
                new_embed, img_kv,
                attention_mask=None,
                self_k_cache=k_caches, self_v_cache=v_caches,
                use_cache=True
            )

            # Get logits
            last_hidden = hidden.transpose(0, 1)  # (B, 1, D)
            logits = self.get_logits(last_hidden)[:, -1, :]
            next_token = logits.argmax(dim=-1)

            generated_tokens.append(next_token.item())
            all_tokens = torch.cat([all_tokens, next_token.unsqueeze(1)], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return all_tokens


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    """Lightweight tokenizer using sentencepiece + special tokens."""
    def __init__(self, tokenizer_dir):
        import sentencepiece as spm
        import json

        model_file = os.path.join(tokenizer_dir, "tokenizer.model")
        vocab_file = os.path.join(tokenizer_dir, "tokenizer.vocab")
        sp_list_file = os.path.join(tokenizer_dir, "tokenizer.sp_token_list.json")

        # Load SP model
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

        # Load vocab
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.encoder = {}
        for i, line in enumerate(lines):
            key = line.split("\t")[0]
            self.encoder[key] = i
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Add special tokens from sp_token_list
        with open(sp_list_file, "r") as f:
            sp_list = json.load(f)

        self.origin_vocab_size = len(self.encoder)
        self.added_vocab_size = len(sp_list)

        for i, token in enumerate(sp_list):
            self.encoder[token] = i + self.origin_vocab_size
            self.decoder[i + self.origin_vocab_size] = token

        # OCR tokens
        self.encoder["<ifly_text>"] = self.origin_vocab_size + self.added_vocab_size
        self.encoder["</ifly_text>"] = self.origin_vocab_size + self.added_vocab_size + 1

        self.sep_id = self.encoder.get("<s>", 1)
        self.eod_id = self.encoder.get("<end>", 2)
        self.pad_id = self.encoder.get("<pad>", 3)

    def tokenize_sp(self, text):
        """Tokenize text with special token awareness."""
        sp_list = [k for k in self.encoder.keys() if "iflytek" in k]
        pattern = "|".join(re.escape(t) for t in sp_list)
        parts = re.split(f"({pattern})", text)
        result = []
        for part in parts:
            if not part:
                continue
            if part in self.encoder:
                result.append(self.encoder[part])
            else:
                result.extend(self.sp.encode(part))
        return result

    def detokenize(self, tokens):
        """Decode token ids to text."""
        chunk = []
        result = ""
        for e in tokens:
            if e >= self.origin_vocab_size:
                if chunk:
                    result += self.sp.decode(chunk)
                    chunk = []
                result += self.decoder.get(e, "")
            else:
                chunk.append(e)
        if chunk:
            result += self.sp.decode(chunk)
        return result


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------
class ImagePreprocessor:
    def __init__(self):
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def load_image(self, image_path):
        """Load and preprocess image for inference."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize preserving aspect ratio, max side = 2048
        h, w = img.shape[:2]
        scale = 2048 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        h, w = img.shape[:2]
        # Pad to exactly 2048x2048 with white
        if h < 2048 or w < 2048:
            img = cv2.copyMakeBorder(img, 0, 2048 - h, 0, 2048 - w,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))

        img = self.to_tensor(img)
        return img.unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Weight Loading
# ---------------------------------------------------------------------------
def load_weights(model, checkpoint_path, logger=print):
    """Load weights from checkpoint, matching key names to model parameters."""
    logger(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model_sd = model.state_dict()
    loaded = 0
    missing = []
    unexpected = []

    for ckpt_key, ckpt_val in state_dict.items():
        # Skip CFGI IPT model keys (structure decoder, not needed)
        if ckpt_key.startswith("cfgi_ipt_model"):
            continue

        # Try direct match first
        if ckpt_key in model_sd:
            if model_sd[ckpt_key].shape == ckpt_val.shape:
                model_sd[ckpt_key] = ckpt_val
                loaded += 1
                continue
            # Try TP padding for 2D weights
            ckpt_shape = ckpt_val.shape
            model_shape = model_sd[ckpt_key].shape
            if len(ckpt_shape) == 2 and len(model_shape) == 2:
                if ckpt_shape[0] == model_shape[0] and ckpt_shape[1] * 2 == model_shape[1]:
                    # TP-split: duplicate instead of zero-pad
                    padded = torch.cat([ckpt_val, ckpt_val], dim=1)
                    model_sd[ckpt_key] = padded
                    loaded += 1
                    logger(f"  TP-duplicated (dim1): {ckpt_key} {ckpt_shape} -> {model_shape}")
                    continue
            missing.append(f"{ckpt_key}: shape mismatch ckpt={ckpt_val.shape} vs model={model_sd[ckpt_key].shape}")
            continue

        # Try mapping ipt_model.transformer.* -> model keys
        if ckpt_key.startswith("ipt_model.transformer.layers."):
            parts = ckpt_key.split(".")
            layer_idx = parts[3]
            rest = ".".join(parts[4:])
            model_key = f"decoder_layers.{layer_idx}.{rest}"
            if model_key in model_sd:
                if model_sd[model_key].shape == ckpt_val.shape:
                    model_sd[model_key] = ckpt_val
                    loaded += 1
                    continue
                elif len(ckpt_val.shape) == 2 and len(model_sd[model_key].shape) == 2:
                    ckpt_in, model_in = ckpt_val.shape[1], model_sd[model_key].shape[1]
                    if ckpt_val.shape[0] == model_sd[model_key].shape[0] and ckpt_in * 2 == model_in:
                        # TP-split fc2: resize fc2 layer to accept half features
                        # fc2 output goes through fc2(activ(fc1(x))[:half])
                        padded = torch.cat([ckpt_val, ckpt_val], dim=1)
                        model_sd[model_key] = padded
                        loaded += 1
                        logger(f"  TP-duplicated: {ckpt_key} -> {model_key}")
                        continue
                missing.append(f"{ckpt_key} -> {model_key}: shape mismatch")
                continue

        if ckpt_key.startswith("ipt_model.transformer.cross_attns."):
            parts = ckpt_key.split(".")
            cross_idx_str = parts[3]
            if not cross_idx_str.isdigit():
                pass
            else:
                cross_idx = int(cross_idx_str)
                rest = ".".join(parts[4:])
                # Checkpoint uses 0-based: 0,1,2 → model uses layer indices: 1,3,5
                layer_map = {0: 1, 1: 3, 2: 5}
                model_cross_idx = layer_map.get(cross_idx, cross_idx)
                model_key = f"cross_attns.{model_cross_idx}.{rest}"
                if model_key in model_sd and model_sd[model_key].shape == ckpt_val.shape:
                    model_sd[model_key] = ckpt_val
                    loaded += 1
                    continue
                else:
                    if model_key in model_sd:
                        missing.append(f"{ckpt_key} -> {model_key}: shape mismatch")
                        continue

        if ckpt_key.startswith("ipt_model.transformer.layernorm."):
            rest = ckpt_key.split("ipt_model.transformer.layernorm.")[-1]
            model_key = f"final_layernorm.{rest}"
            if model_key in model_sd and model_sd[model_key].shape == ckpt_val.shape:
                model_sd[model_key] = ckpt_val
                loaded += 1
                continue

        # IPT embeddings
        if ckpt_key == "ipt_model.embedding.word_embeddings.weight":
            if model_sd["word_embeddings.weight"].shape == ckpt_val.shape:
                model_sd["word_embeddings.weight"] = ckpt_val
                loaded += 1
                continue

        if ckpt_key in ("ipt_bbox_embedding.bbox_embedding.weight",
                         "ipt_bbox_embedding.ipt_embedding.word_embeddings.weight"):
            target = "bbox_embedding.weight" if "bbox_embedding.weight" in ckpt_key else "word_embeddings.weight"
            if model_sd[target].shape == ckpt_val.shape:
                model_sd[target] = ckpt_val
                loaded += 1
                continue

        unexpected.append(ckpt_key)

    model.load_state_dict(model_sd)
    logger(f"Loaded {loaded}/{len(state_dict)} weights")
    if missing:
        logger(f"Missing/mismatched ({len(missing)}):")
        for m in missing[:10]:
            logger(f"  {m}")
        if len(missing) > 10:
            logger(f"  ... and {len(missing) - 10} more")
    if unexpected:
        logger(f"Unexpected keys ({len(unexpected)}):")
        for u in unexpected[:10]:
            logger(f"  {u}")
        if len(unexpected) > 10:
            logger(f"  ... and {len(unexpected) - 10} more")


# ---------------------------------------------------------------------------
# Post-processing: convert special tokens to HTML
# ---------------------------------------------------------------------------
def reverse_format(content):
    """Convert model output special tokens to actual HTML."""
    content = content.replace("<iflytek_html_html_s>", "<html>")
    content = content.replace("<iflytek_html_html_e>", "</html>")
    content = content.replace("<iflytek_html_body_s>", "<body>")
    content = content.replace("<iflytek_html_body_e>", "</body>")
    content = content.replace("<iflytek_html_table_s>", "<table>")
    content = content.replace("<iflytek_html_table_e>", "</table>")
    content = content.replace("<iflytek_html_thead_s>", "<thead>")
    content = content.replace("<iflytek_html_thead_e>", "</thead>")
    content = content.replace("<iflytek_html_tbody_s>", "<tbody>")
    content = content.replace("<iflytek_html_tbody_e>", "</tbody>")
    content = content.replace("<iflytek_html_td_s>", "<td>")
    content = content.replace("<iflytek_html_td_e>", "</td>")
    content = content.replace("<iflytek_html_tr_s>", "<tr>")
    content = content.replace("<iflytek_html_tr_e>", "</tr>")
    content = content.replace("<iflytek_br>", "<br>")

    # Handle span tags with rowspan/colspan
    pattern = re.compile(r"(<iflytek_html_span_s>(.*?)<iflytek_html_span_e>)")
    for match in pattern.findall(content):
        match_str = match[0]
        try:
            if "rowspan" in match_str and "colspan" in match_str:
                col_n = re.findall(r"<iflytek_html_colspan>(\d+)", match_str)[0]
                row_n = re.findall(r"<iflytek_html_rowspan>(\d+)", match_str)[0]
                if match_str.index("rowspan") > match_str.index("colspan"):
                    new_str = match_str.replace(
                        f"<iflytek_html_span_s><iflytek_html_colspan>{col_n}",
                        f"<td colspan={col_n} ")
                    new_str = new_str.replace(
                        f"<iflytek_html_rowspan>{row_n}<iflytek_html_span_e>",
                        f"rowspan={row_n}>")
                else:
                    new_str = match_str.replace(
                        f"<iflytek_html_colspan>{col_n}<iflytek_html_span_e>",
                        f"colspan={col_n}>")
                    new_str = new_str.replace(
                        f"<iflytek_html_span_s><iflytek_html_rowspan>{row_n}",
                        f"<td rowspan={row_n} ")
            elif "rowspan" in match_str:
                row_n = re.findall(r"<iflytek_html_rowspan>(\d+)", match_str)[0]
                new_str = match_str.replace(
                    f"<iflytek_html_span_s><iflytek_html_rowspan>{row_n}<iflytek_html_span_e>",
                    f"<td rowspan={row_n}>")
            elif "colspan" in match_str:
                col_n = re.findall(r"<iflytek_html_colspan>(\d+)", match_str)[0]
                new_str = match_str.replace(
                    f"<iflytek_html_span_s><iflytek_html_colspan>{col_n}<iflytek_html_span_e>",
                    f"<td colspan={col_n}>")
            else:
                new_str = match_str
            content = content.replace(match_str, new_str)
        except Exception:
            pass

    content = content.replace("<iflytek_line_equation_s>", "")
    content = content.replace("<iflytek_line_equation_e>", "")
    content = content.replace("<iflytek_inline_equation_s>", "")
    content = content.replace("<iflytek_inline_equation_e>", "")
    content = content.replace("<iflytek_unk>", "")
    content = content.replace("<iflytek_left_brace>", "{")
    content = content.replace("<iflytek_right_brace>", "}")

    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TDATR Standalone Inference")
    parser.add_argument("--image", required=True, help="Path to input table image")
    parser.add_argument("--checkpoint", default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_dir", default="TDATR/tokenizers", help="Path to tokenizer files")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--prompt", default="将图片中的表格转换为HTML语言。<iflytek_ret>",
                        help="Prompt text")
    args = parser.parse_args()

    device = torch.device("cpu")

    print("=" * 60)
    print("TDATR Standalone Inference")
    print("=" * 60)

    # 1. Build model
    print("\n[1/4] Building model...")
    model = StandaloneTDATR()

    # 2. Load weights
    print("\n[2/4] Loading weights...")
    load_weights(model, args.checkpoint)

    model.eval()
    model.to(device)

    # 3. Preprocess image
    print(f"\n[3/4] Loading image: {args.image}")
    preprocessor = ImagePreprocessor()
    image_tensor = preprocessor.load_image(args.image).to(device)
    print(f"  Image tensor shape: {image_tensor.shape}")

    # 4. Encode image
    print("\n[4/4] Running inference...")
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor)
        print(f"  Image embeddings shape: {image_embeds.shape}")

    # 5. Tokenize prompt
    tokenizer = SimpleTokenizer(args.tokenizer_dir)
    prompt_tokens = tokenizer.tokenize_sp(args.prompt)
    prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(f"  Prompt tokens: {prompt_tokens.shape}")

    # 6. Generate
    eos_id = tokenizer.encoder.get("<end>", 2)
    with torch.no_grad():
        output_tokens = model.generate(image_embeds, prompt_tokens,
                                        max_new_tokens=args.max_new_tokens, eos_id=eos_id)

    # 7. Decode
    generated_ids = output_tokens[0, prompt_tokens.size(1):].tolist()
    raw_text = tokenizer.detokenize(generated_ids)
    raw_text = raw_text.replace("<end>", "")
    raw_text = raw_text.replace("<iflytek_ret>", "\n")

    html_output = reverse_format(raw_text)

    print("\n" + "=" * 60)
    print("GENERATED HTML:")
    print("=" * 60)
    print(html_output)
    print("=" * 60)


if __name__ == "__main__":
    main()
