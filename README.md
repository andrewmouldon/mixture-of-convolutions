# Mixture of Convolutions (Triton)

Triton kernels for **Mixture of Convolutions (MoC)**, a token-adaptive generalization of short causal depthwise convolution.

Standard short convolution applies the same learned kernel at every sequence position. MoC instead learns a bank of convolutional bases and uses a token-dependent router to mix them into a different effective kernel at each position.

## Math

```text
alpha[b,t,:] = softmax(router(z[b,t,:]))

weight[b,t,s,d] = sum_k alpha[b,t,k] * basis[k,s,d]

y[b,t,d] = sum_s weight[b,t,s,d] * x[b,t-s,d]
```

where:

* `x` has shape `[B, T, D]`
* `z` is the routing input
* `alpha` has shape `[B, T, K]`
* `basis` has shape `[K, S, D]`
* `K` is the number of convolutional bases
* `S` is the kernel size

The Triton implementation fuses the basis mixture into the convolution and does not materialize the per-token dynamic kernels.

The implementation supports:

* fixed-length sequences
* packed variable-length sequences
* autoregressive decoding with a convolution cache
* optional fused SiLU activation

## Usage

The high-level `MoC` module contains the learned convolutional bases and token router.

```python
import torch
from moc import MoC

B, T, D = 4, 4096, 2048

moc = MoC(
    dim=D,
    z_dim=D,
    kernel_size=4,
    k=16,
    init_std=0.002,
).cuda().bfloat16()

x = torch.randn(
    B, T, D,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)

# Representation used by the router.
# This does not need to be the same representation as x.
z = torch.randn(
    B, T, D,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)

y, _ = moc(x, z)

y.sum().backward()
```

`x` is the representation being convolved, while `z` determines the token-dependent mixture over convolutional bases.

## Example: Applying MoC to QKV

MoC can replace static short convolutions on the projected queries, keys, and values of an attention or sequence-mixing layer.

Create separate MoC modules using the projected dimension of each representation:

```python
from moc import MoC

self.q_conv = MoC(
    dim=q_dim,
    z_dim=d_model,
    kernel_size=4,
    k=16,
    init_std=0.002,
)

self.k_conv = MoC(
    dim=k_dim,
    z_dim=d_model,
    kernel_size=4,
    k=16,
    init_std=0.002,
)

self.v_conv = MoC(
    dim=v_dim,
    z_dim=d_model,
    kernel_size=4,
    k=16,
    init_std=0.002,
)
```

Inside a pre-norm block:

```python
x_norm = norm(x)

# Project Q/K/V
q, k, v = qkv_proj(x_norm).split(
    [q_dim, k_dim, v_dim],
    dim=-1,
)

# Apply token-adaptive short convolutions.
# x_norm is used as the routing input for each convolution.
q, _ = self.q_conv(q, x_norm)
k, _ = self.k_conv(k, x_norm)
v, _ = self.v_conv(v, x_norm)

# Reshape into heads
q = q.view(B, T, n_q_heads, q_head_dim).transpose(1, 2)
k = k.view(B, T, n_kv_heads, k_head_dim).transpose(1, 2)
v = v.view(B, T, n_kv_heads, v_head_dim).transpose(1, 2)

# Per-head QK normalization
q = q_norm(q)
k = k_norm(k)

# RoPE follows convolution and QK normalization
q, k = apply_rope(q, k)

out = sequence_mixer(
    q,
    k,
    v,
    is_causal=True,
)

out = out.transpose(1, 2).contiguous().view(B, T, v_dim)

# If used by the architecture, normalize the retrieved /
# sequence-mixing output before the output projection.
#out = out_norm(out)

out = out_proj(out)
```

## Initialization and Learning Rate

The recommended initialization depends on whether the **convolution-affected representation is normalized before it contributes to the residual stream**.

For the query and key convolutions, this corresponds to normalizing the convolved Q/K representations before attention, e.g. with QK normalization.

For the value convolution, the relevant normalization is applied **after sequence mixing** to the retrieved value / attention output, before the output projection and residual update.

This post-sequence-mixing normalization is already part of architectures such as Gated DeltaNet. It is not part of a standard Transformer block.

### With downstream normalization

If the convolution-affected representation is normalized downstream — for example:

- Q/K are normalized after convolution and before attention, and
- the sequence-mixing output is normalized before the output projection / residual update,

use:

```python
init_std = 0.002
```

with a standard learning rate such as:

```text
AdamW LR = 1e-3
```

This is the recipe used in the main experiments.

### Without downstream normalization

If the convolution output is allowed to determine the forward scale directly, as in a standard Transformer without post-sequence-mixing normalization, use the default fan-in initialization to preserve variance:

```python
moc = MoC(
    dim=D,
    z_dim=D,
    kernel_size=4,
    k=16,
    init_std=None,
)
```

For `K` bases and kernel size `S`, the default initialization has standard deviation:

```text
fan_in_std = sqrt(K / (3 * S))
```

The convolution learning rate should then be increased proportionally to the initialization scale so that the initial optimizer update remains approximately matched relative to the parameter scale:

```text
matched_lr = reference_lr * init_std / reference_std
```

Using the paper reference values:

```text
reference_std = 0.002
reference_lr  = 1e-3
```

For `K = 16` and `S = 4`:

```text
fan_in_std ≈ 1.1547
matched_lr ≈ 0.577
```

Helpers are provided in `moc.optim`:

```python
from moc import (
    moc_fan_in_std,
    match_lr_to_init_std,
    match_weight_decay_to_lr,
)

init_std = moc_fan_in_std(
    k=16,
    kernel_size=4,
)

conv_lr = match_lr_to_init_std(
    init_std,
    reference_std=0.002,
    reference_lr=1e-3,
)

conv_weight_decay = match_weight_decay_to_lr(
    conv_lr,
    reference_lr=1e-3,
    reference_weight_decay=0.1,
)

print(init_std)           # 1.1547...
print(conv_lr)            # 0.5773...
print(conv_weight_decay)  # 0.0001732...
```

The weight decay is scaled inversely with the learning rate so that AdamW applies the same per-step multiplicative shrinkage as the reference configuration.
## Autoregressive Decoding

`MoC` supports cached single-token decoding:

```python
cache = torch.zeros(
    B,
    D,
    moc.kernel_size,
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

where `x_t` and `z_t` have shape:

```text
[B, 1, D]
```

## Packed Variable-Length Sequences

Packed variable-length sequences are supported through `cu_seqlens` and `max_seqlen`:

```python
y, _ = moc(
    x,
    z,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
)
```

Packed inputs may use:

```text
x  [N, D]
z  [N, z_dim]
```

where `N` is the total number of tokens across all packed sequences.

## Files

* `api.py` — public `MoC` module and `moc_triton` interface
* `kernels.py` — fused Triton forward and backward kernels
* `ops.py` — Triton kernel launch and dispatch logic
* `reference.py` — PyTorch reference implementation
* `optim.py` — initialization and learning-rate helpers
* `verify.py` — correctness checks against the PyTorch reference

Run the verification suite with:

```bash
python verify.py
```

## Citation

```bibtex
@misc{mouldon2026moc,
    title={Mixture of Convolutions: Token-Adaptive Short Convolution for Sequence Modeling},
    author={Andrew Mouldon},
    year={2026},
}
```

## License

MIT.
