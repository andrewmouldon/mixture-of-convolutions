from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from . import ops
from .optim import moc_fan_in_std


SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def activation_flag(activation: str | None) -> int:
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation: {activation}")
    return int(activation is not None)


def check_router_activation(router_activation: str) -> None:
    if router_activation not in ("softmax", "exp"):
        raise ValueError(
            f"unsupported router activation: {router_activation}; "
            "expected 'softmax' or 'exp'"
        )


def check_common_dtypes(x: torch.Tensor, alpha: torch.Tensor, basis: torch.Tensor) -> None:
    if not (x.is_cuda and alpha.is_cuda and basis.is_cuda):
        raise ValueError("x, alpha, and basis must all be CUDA tensors")
    if not (x.device == alpha.device == basis.device):
        raise ValueError("x, alpha, and basis must be on the same CUDA device")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported x dtype: {x.dtype}")
    if not alpha.is_floating_point():
        raise TypeError(f"alpha must be floating point, got {alpha.dtype}")
    if not basis.is_floating_point():
        raise TypeError(f"basis must be floating point, got {basis.dtype}")


def check_fixed_inputs(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    check_common_dtypes(x, alpha, basis)
    if x.ndim != 3 or alpha.ndim != 3 or basis.ndim != 3:
        raise ValueError("fixed-length inputs must have x/alpha/basis ranks 3/3/3")
    B, T, D = x.shape
    B2, T2, K = alpha.shape
    K2, KS, D2 = basis.shape
    if (B, T) != (B2, T2):
        raise ValueError(f"x and alpha batch/time shapes differ: {(B, T)} vs {(B2, T2)}")
    if (K, D) != (K2, D2):
        raise ValueError(f"alpha/basis dimensions disagree: K,D={(K, D)} vs {(K2, D2)}")
    if min(B, T, D, K, KS) <= 0:
        raise ValueError("B, T, D, K, and KS must all be positive")
    return B, T, D, K, KS


def check_packed_inputs(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[int, int, int, int, int]:
    check_common_dtypes(x, alpha, basis)
    if x.ndim != 2 or alpha.ndim != 2 or basis.ndim != 3:
        raise ValueError("packed inputs must have x/alpha/basis ranks 2/2/3 internally")
    N, D = x.shape
    N2, K = alpha.shape
    K2, KS, D2 = basis.shape
    if N != N2:
        raise ValueError(f"x and alpha token counts differ: {N} vs {N2}")
    if (K, D) != (K2, D2):
        raise ValueError(f"alpha/basis dimensions disagree: K,D={(K, D)} vs {(K2, D2)}")
    if min(N, D, K, KS) <= 0:
        raise ValueError("total tokens, D, K, and KS must all be positive")
    if not cu_seqlens.is_cuda or cu_seqlens.device != x.device:
        raise ValueError("cu_seqlens must be a CUDA tensor on the same device as x")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be rank-1 with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive Python int")
    return N, cu_seqlens.numel() - 1, D, K, KS


class _MoCFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha: torch.Tensor,
        basis: torch.Tensor,
        activation: str | None,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        cu_seqlens_cpu: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x.contiguous()
        alpha = alpha.contiguous()
        basis = basis.contiguous()
        is_varlen = cu_seqlens is not None

        if is_varlen:
            if max_seqlen is None:
                raise ValueError("max_seqlen is required when cu_seqlens is provided")
            cu_seqlens = cu_seqlens.contiguous()
            if cu_seqlens_cpu is not None:
                cu_seqlens_cpu = cu_seqlens_cpu.contiguous()
            N, n_seqs, D, K, KS = check_packed_inputs(
                x, alpha, basis, cu_seqlens, max_seqlen,
            )
            if chunk_indices is None:
                chunk_indices = ops.prepare_chunk_indices(
                    cu_seqlens,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
            else:
                chunk_indices = chunk_indices.contiguous()
            B = T = None
        else:
            if max_seqlen is not None:
                raise ValueError("max_seqlen must be None when cu_seqlens is None")
            if cu_seqlens_cpu is not None or chunk_indices is not None:
                raise ValueError("cu_seqlens_cpu and chunk_indices require cu_seqlens")
            B, T, D, K, KS = check_fixed_inputs(x, alpha, basis)
            N = n_seqs = None

        act = activation_flag(activation)
        out = ops.launch_forward(
            x,
            alpha,
            basis,
            activation_flag=act,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        if is_varlen:
            ctx.save_for_backward(x, alpha, basis, cu_seqlens)
        else:
            ctx.save_for_backward(x, alpha, basis)

        ctx.activation_flag = act
        ctx.is_varlen = is_varlen
        ctx.chunk_indices = chunk_indices
        ctx.shape = (B, T, N, n_seqs, D, K, KS)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy: torch.Tensor):
        activation = ctx.activation_flag
        is_varlen = ctx.is_varlen
        chunk_indices = ctx.chunk_indices

        if is_varlen:
            x, alpha, basis, cu_seqlens = ctx.saved_tensors
        else:
            x, alpha, basis = ctx.saved_tensors
            cu_seqlens = None

        dy = dy.contiguous()
        need_x = ctx.needs_input_grad[0]
        preact = dy

        if activation:
            preact = ops.launch_preact(
                x,
                alpha,
                basis,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        dx, dalpha, dbasis = ops.launch_backward(
            x,
            alpha,
            basis,
            dy,
            preact,
            activation_flag=activation,
            need_x=need_x,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return dx, dalpha, dbasis, None, None, None, None, None


def _canonicalize_packed_input(
    x: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if x.ndim == 2 and alpha.ndim == 2:
        return x, alpha, False
    if x.ndim == 3 and alpha.ndim == 3 and x.shape[0] == 1 and alpha.shape[0] == 1:
        return x.squeeze(0), alpha.squeeze(0), True
    raise ValueError("varlen x/alpha must be [N,D]/[N,K] or [1,N,D]/[1,N,K]")


def _check_cache(x: torch.Tensor, basis: torch.Tensor, cache: torch.Tensor) -> None:
    B, _, D = x.shape
    KS = basis.shape[1]
    if cache.shape != (B, D, KS):
        raise ValueError(f"cache must have shape {(B, D, KS)}, got {tuple(cache.shape)}")
    if cache.device != x.device or cache.dtype != x.dtype:
        raise ValueError("cache must have the same device and dtype as x")


def _final_state(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    kernel_size: int,
) -> torch.Tensor:
    B, T, D = x.shape
    if T >= kernel_size:
        return x[:, -kernel_size:].transpose(1, 2).contiguous()
    prefix = x.new_zeros(B, D, kernel_size - T) if cache is None else cache[:, :, T:]
    return torch.cat((prefix, x.transpose(1, 2)), dim=-1)


def _fixed_moc(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    check_fixed_inputs(x, alpha, basis)
    return _MoCFunction.apply(
        x, alpha, basis, activation, None, None, None, None,
    )


def _step_moc(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    cache: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    _, T, _, _, _ = check_fixed_inputs(x, alpha, basis)
    if T != 1:
        raise ValueError(f"step expects sequence length 1, got {T}")
    _check_cache(x, basis, cache)
    return ops.launch_update(
        x.squeeze(1).contiguous(),
        alpha.squeeze(1).contiguous(),
        basis.contiguous(),
        cache,
        activation_flag=activation_flag(activation),
    ).unsqueeze(1)


def moc_triton(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    cache: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    activation_flag(activation)

    if cu_seqlens is None:
        if cu_seqlens_cpu is not None or chunk_indices is not None:
            raise ValueError("cu_seqlens_cpu and chunk_indices require cu_seqlens")
        B, T, D, K, KS = check_fixed_inputs(x, alpha, basis)
        if cache is not None:
            _check_cache(x, basis, cache)
        if cache is not None and T == 1:
            out = _step_moc(
                x,
                alpha,
                basis,
                cache,
                activation,
            )
            return out, (cache if output_final_state else None)

        if cache is None:
            out = _fixed_moc(
                x,
                alpha,
                basis,
                activation,
            )
        else:
            history = cache.transpose(1, 2)
            alpha_prefix = alpha.new_zeros(B, KS, K)
            out = _fixed_moc(
                torch.cat((history, x), dim=1),
                torch.cat((alpha_prefix, alpha), dim=1),
                basis,
                activation,
            )[:, KS:]
        final_state = _final_state(x, cache, KS) if output_final_state else None
        return out, final_state

    if cache is not None or output_final_state:
        raise ValueError("cache and output_final_state are only supported for fixed-length inputs")
    if max_seqlen is None:
        raise ValueError("max_seqlen is required for varlen inputs")

    x_packed, alpha_packed, restore = _canonicalize_packed_input(x, alpha)
    N, _, D, _, _ = check_packed_inputs(
        x_packed,
        alpha_packed,
        basis,
        cu_seqlens,
        max_seqlen,
    )
    out = _MoCFunction.apply(
        x_packed,
        alpha_packed,
        basis,
        activation,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_cpu,
        chunk_indices,
    )

    return (out.unsqueeze(0) if restore else out), None


class MoC(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        kernel_size: int,
        k: int,
        activation: str | None = None,
        init_std: float | None = None,
        router_activation: str = "softmax",
    ) -> None:
        super().__init__()
        activation_flag(activation)
        check_router_activation(router_activation)

        self.dim = dim
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.k = k
        self.activation = activation
        self.router_activation = router_activation

        self.conv_bases = nn.Parameter(torch.empty(k, kernel_size, dim))
        self.router = nn.Parameter(torch.zeros(k, z_dim))

        if init_std is None:
            init_std = moc_fan_in_std(k, kernel_size)

        nn.init.uniform_(
            self.conv_bases,
            a=-init_std * math.sqrt(3),
            b=init_std * math.sqrt(3),
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        router_logits = F.linear(z, self.router)

        if self.router_activation == "softmax":
            alpha = torch.softmax(router_logits, dim=-1)
        else:
            alpha = torch.exp(router_logits)

        return moc_triton(
            x,
            alpha,
            self.conv_bases,
            cache=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )

    def step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"step expects x with shape [B, 1, D], got {tuple(x.shape)}")
        if z.ndim != 3 or z.shape[:2] != x.shape[:2]:
            raise ValueError(f"step expects z with shape [B, 1, Z], got {tuple(z.shape)}")
        if cache is None:
            cache = x.new_zeros(x.shape[0], self.dim, self.kernel_size)
        return self.forward(
            x,
            z,
            cache=cache,
            output_final_state=output_final_state,
        )

    @property
    def state_size(self) -> int:
        return self.dim * self.kernel_size


__all__ = ["MoC", "moc_triton"]
