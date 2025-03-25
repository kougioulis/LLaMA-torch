import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch import einsum

from modules.rotary import RotaryEmbedding, apply_rotary_pos_emb

class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float=0.1):
        """
        Grouped Multi-Query Attention (GQA) layer.
        Each query head computes independent attention over a group of key/value heads.
        Key/value heads are shared across groups of query heads, contrary to standard multi-query attention where
        key/value heads are unique to each query head.
        
        Args:
        ---
            dim (int):        Model dimension (must be divisible by n_heads)
            n_heads (int):    Number of query heads
            n_kv_heads (int): Number of key/value heads (must divide n_heads)
            dropout (float):    Attention dropout probability
            
        Shapes:
        ---
            Input:  (batch_size, seq_len, dim)
            Output: (batch_size, seq_len, dim)
            
        Projections:
        ---
            Q: dim → dim → (batch, n_heads, seq_len, head_dim)
            K/V: dim → 2*(n_kv_heads*head_dim) → (batch, n_kv_heads, seq_len, head_dim)
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # Vaswani et al. scaling factor

        # projections 
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_kv = nn.Linear(dim, 2 * n_kv_heads * self.head_dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, x: torch.tensor):
        """
        Forward pass for grouped multi-query attention
        
        Args:
        ---
            x (torch.tensor): input tensor of shape [batch, seq_len, dim]
            
        Returns:
        ---
            (torch.tensor): attention-transformed output of shape [batch, seq_len, dim]
        """
        B, N, C = x.shape

        # queries, keys, and values computation
        q = rearrange(self.w_q(x), "b n (h d) -> b h n d", h=self.n_heads)
        kv = self.w_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b n (g d) -> b g n d", g=self.n_kv_heads), kv)

        pos_emb = self.rotary_emb(N, x.device)
        q, k = map(lambda t: apply_rotary_pos_emb(pos_emb, t), (q, k))

        # scaled dot-product attention
        attn_scores = einsum("b h i d, b g j d -> b h i j", q, k) * self.scale
        attn_probs = self.dropout(attn_scores.softmax(dim=-1))

        # final attention output
        out = einsum("b h i j, b g j d -> b h i d", attn_probs, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.w_out(out)