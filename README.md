## LLaMA-torch

LLaMA-like [1] implementation in Pytorch with Grouped Multi-Query Attention instead of standard Multi-head Attention as in the original paper. Also includes RoPE embeddings, RMS normalization and SwiGLU activations. It purposely omits the SentencePiece tokenizer and model scaling configurations.

## Usage

```python
import torch

from model import LLaMA
from utils.utils import count_parameters

n_tokens = 100
d_model = 256
n_layers = 16
n_heads = 8
n_kv_heads = 4
batch_size = 2
seq_length = 64 

input_tensor = torch.randint(low=0, high=n_tokens,
                             size=(batch_size, seq_length))

model = LLAMA(d_model, n_tokens, n_layers, n_heads, n_kv_heads)

output = model(input_tensor) # torch.Size([2, 64, 100])
print(output.shape)

print(f'Number of parameters: {count_parameters(model)}') # 103843072
```

## References

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}