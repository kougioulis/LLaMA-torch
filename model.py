import torch
import torch.nn as nn

from modules.RMSNorm import RMSNorm
from modules.transformer_block import TransformerBlock

class LLaMA(nn.Module):
    def __init__(self, d_model: int, n_tokens: int, n_layers: int, n_heads: int, n_kv_heads: int, d_ff=32):
        """
        LLaMa-style model architecture.
        
        Args:
        ---
            d_model (int): model dimension
            n_tokens (int): vocabulary size
            n_layers (int): number of transformer blocks
            n_heads (int): number of attention heads
            n_kv_heads (int): number of key/value heads
            d_ff (int): feedforward hidden dimension. Defaults to 32.
            
        Notes:
        ---
        [*] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. 
            arXiv:2302.13971
        """
        super().__init__()

        self.embed = nn.Embedding(n_tokens, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff) 
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_tokens, bias=False)
        self.out_proj.weight = self.embed.weight  # Weight tying
    
    def forward(self, x: torch.tensor):
        """
        Args:
        ---
            x (torch.tensor): input tensor of shape [batch, seq_len]
            
        Returns:
        ---
            (torch.tensor): output logits of shape [batch, seq_len, n_tokens]
        """
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return self.out_proj(x)