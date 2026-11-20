# Mixture of Convolutions (Triton)

Triton implementation of **Mixture of Convolutions (MoC)**, a token-adaptive generalization of short causal depthwise convolution.

Instead of applying the same learned convolution kernel at every position, MoC learns a bank of convolutional bases and uses token-dependent routing weights to mix them into a different effective kernel at each position. The Triton kernels fuse this mixture into the convolution without materializing the dynamic kernels.

Supports fixed-length sequences, packed variable-length sequences, autoregressive decoding with a convolution cache, and optional fused SiLU activation.

## Usage

```python
import torch
from moc import MoC

moc = MoC(
    dim=2048,
    z_dim=2048,
    kernel_size=4,
    k=16,
    routing_activation="exp",
    routing_norm="l2",
).cuda().bfloat16()

x = torch.randn(4, 4096, 2048, device="cuda", dtype=torch.bfloat16)
z = torch.randn(4, 4096, 2048, device="cuda", dtype=torch.bfloat16)

y, _ = moc(x, z)
```

`x` is the representation being convolved, while `z` determines the token-dependent mixture. They do not need to be the same representation.

## Recommended Settings

Unless you have a specific reason otherwise:

- **Routing normalization:** Use `routing_norm="l2"`.
- **Routing activation:** Use `"exp"` or `"softplus"`. Softplus is a good starting point when the convolution directly interacts with the residual stream.
- **Initialization:** Use the default `init_std=None`.
- **Learning rate:** Use AdamW with a separate learning rate of approximately `0.01–0.03` for `conv_bases`.
- **Weight decay:** Match the weight decay to your normal AdamW learning rate so the per-step shrinkage remains the same:

```python
conv_weight_decay = base_weight_decay * base_lr / conv_lr
```

For example, with a normal AdamW learning rate of `1e-3`, weight decay of `0.1`, and convolution learning rate of `0.03`, use convolution weight decay of approximately `0.00333`.

## Autoregressive Decoding

```python
cache = torch.zeros(
    B, D, moc.kernel_size,
    device="cuda",
    dtype=torch.bfloat16,
)

y, cache = moc.step(
    x_t,
    z_t,
    cache=cache,
    output_final_state=True,
)
```

`x_t` and `z_t` have shapes `[B, 1, dim]` and `[B, 1, z_dim]`, respectively.

## Packed Variable-Length Sequences

```python
y, _ = moc(
    x,
    z,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
)
```

Packed inputs use `x` with shape `[N, dim]` and `z` with shape `[N, z_dim]`.
