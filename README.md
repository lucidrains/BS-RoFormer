<img src="./bs-roformer.png" width="450px"></img>

## BS-RoFormer (wip)

Implementation of <a href="https://arxiv.org/abs/2309.02612">Band Split Roformer</a>, SOTA Attention network for music source separation out of ByteDance AI Labs. They beat the previous first place by a large margin. The technique uses axial attention across frequency (hence multi-band) and time. They also have experiments to show that rotary positional encoding led to a huge improvement over learned absolute positions.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

## Install

```bash
$ pip install BS-RoFormer
```

## Usage

```python
import torch
from bs_roformer import BSRoformer

model = BSRoformer(
    dim = 512,
    depth = 12,
    time_transformer_depth = 1,
    freq_transformer_depth = 1,
    freqs_per_bands = (256, 257)  # in the paper, they divide into ~60 bands, test with 1 for starters
)

x = torch.randn(2, 131680)
target = torch.randn(2, 131680)

loss = model(x, target = target)
loss.backward()

# after much training

out = model(x)
```

## Todo

- [x] get the multiscale stft loss in there
- [ ] figure out what `n_fft` should be
- [ ] review band split + mask estimation modules

## Citations

```bibtex
@inproceedings{Lu2023MusicSS,
    title   = {Music Source Separation with Band-Split RoPE Transformer},
    author  = {Wei-Tsung Lu and Ju-Chiang Wang and Qiuqiang Kong and Yun-Ning Hung},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:261556702}
}
```

```bibtex
@misc{ho2019axial,
    title  = {Axial Attention in Multidimensional Transformers},
    author = {Jonathan Ho and Nal Kalchbrenner and Dirk Weissenborn and Tim Salimans},
    year   = {2019},
    archivePrefix = {arXiv}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```
