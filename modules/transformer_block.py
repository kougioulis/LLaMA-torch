import torch
import torch.nn as nn

from modules.attention import GroupedMultiQueryAttention
from modules.SwiGLU import SwiGLU
from modules.RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ff_mult: int=4):
        """
        Transformer block with grouped multi-query attention and SwiGLU FFN
        
        Args:
        ---
            d_model (int): model dimension
            n_heads (int): number of attention heads
            n_kv_heads (int): number of key/value heads
            ff_mult (int): feedforward expansion multiplier. Defaults to 4.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = GroupedMultiQueryAttention(d_model, n_heads, n_kv_heads)
        d_ff = ff_mult * d_model 
        self.ff = SwiGLU(d_model, d_ff)
    
    def forward(self, x: torch.tensor):
        """
        Forward pass through transformer block
        
        Args:
        ---
            x (torch.tensor): input tensor of shape [batch, seq_len, d_model]
            
        Returns:
        ---
            (torch.tensor): output tensor of shape [batch, seq_len, d_model]
        """

        x = x + self.attn(self.norm1(x))  # attention with residual
        x = x + self.ff(self.norm2(x))    # FFN with residual

        return x