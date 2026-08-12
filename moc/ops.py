from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from .kernels import (
    _moc_bwd_dalpha_dbasis_fixed_kernel,
    _moc_bwd_dalpha_dbasis_varlen_kernel,
    _moc_bwd_dx_fixed_kernel,
    _moc_bwd_dx_varlen_kernel,
    _moc_fwd_fixed_kernel,
    _moc_fwd_varlen_kernel,
    _moc_update_kernel,
)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_block_k_fwd_dx(k: int) -> int:
    return max(16, _next_power_of_2(k))


def _pick_block_k_bwd_reduction(k: int) -> int:
    return max(16, _next_power_of_2(k))


VARLEN_BLOCK_T = 64
_CHUNK_CACHE_SIZE = 4
_chunk_index_cache: list[tuple[torch.Tensor, int, int, torch.Tensor | None, int | None, torch.Tensor]] = []


def _autotune_bucket(x: int) -> int:
    return _next_power_of_2(max(1, x))


def _tensor_version(x: torch.Tensor | None) -> int | None:
    return None if x is None else x._version


def clear_chunk_index_cache() -> None:
    _chunk_index_cache.clear()


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int = VARLEN_BLOCK_T,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive Python int")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be rank-1 with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64")

    if cu_seqlens_cpu is not None:
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("cu_seqlens_cpu must be a CPU tensor")
        if cu_seqlens_cpu.ndim != 1 or cu_seqlens_cpu.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens_cpu must match cu_seqlens shape")
        if cu_seqlens_cpu.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens_cpu must have dtype torch.int32 or torch.int64")

    cu_version = _tensor_version(cu_seqlens)
    cpu_version = _tensor_version(cu_seqlens_cpu)
    for i, entry in enumerate(_chunk_index_cache):
        cached_cu, cached_version, cached_size, cached_cpu, cached_cpu_version, cached_chunks = entry
        if (
            cached_cu is cu_seqlens
            and cached_version == cu_version
            and cached_size == chunk_size
            and cached_cpu is cu_seqlens_cpu
            and cached_cpu_version == cpu_version
        ):
            if i:
                _chunk_index_cache.insert(0, _chunk_index_cache.pop(i))
            return cached_chunks

    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    lengths = src[1:] - src[:-1]
    chunk_counts = torch.div(lengths + chunk_size - 1, chunk_size, rounding_mode="floor")
    seq_ids = torch.repeat_interleave(
        torch.arange(chunk_counts.numel(), device=src.device, dtype=chunk_counts.dtype),
        chunk_counts,
    )
    seq_starts = F.pad(chunk_counts.cumsum(0), (1, 0))[:-1]
    intra_chunk = (
        torch.arange(seq_ids.shape[0], device=src.device, dtype=chunk_counts.dtype)
        - seq_starts[seq_ids]
    )
    chunk_indices = torch.stack((seq_ids, intra_chunk), dim=1).to(
        device=cu_seqlens.device,
        dtype=torch.int32,
    ).contiguous()

    _chunk_index_cache.insert(
        0,
        (cu_seqlens, cu_version, chunk_size, cu_seqlens_cpu, cpu_version, chunk_indices),
    )
    del _chunk_index_cache[_CHUNK_CACHE_SIZE:]
    return chunk_indices

def _fixed_strides(x: torch.Tensor, alpha: torch.Tensor, out: torch.Tensor):
    return (*x.stride(), *alpha.stride(), *out.stride())


def _varlen_strides(x: torch.Tensor, alpha: torch.Tensor, out: torch.Tensor):
    return (
        0, x.stride(0), x.stride(1),
        0, alpha.stride(0), alpha.stride(1),
        0, out.stride(0), out.stride(1),
    )


def _check_chunk_indices(chunk_indices: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    if not chunk_indices.is_cuda or chunk_indices.device != cu_seqlens.device:
        raise ValueError("chunk_indices must be a CUDA tensor on the same device as cu_seqlens")
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2]")
    if chunk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("chunk_indices must have dtype torch.int32 or torch.int64")
    if not chunk_indices.is_contiguous():
        raise ValueError("chunk_indices must be contiguous")


def launch_forward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    *,
    activation_flag: int,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    block_t: int = VARLEN_BLOCK_T,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    K, KS, _ = basis.shape
    bf16_compute = x.dtype == torch.bfloat16
    out = torch.empty_like(x)

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, block_t, cu_seqlens_cpu)
        _check_chunk_indices(chunk_indices, cu_seqlens)
        N, D = x.shape
        B = 0
        T = 0
        n_chunks = chunk_indices.shape[0]
        n_chunks_bucket = _autotune_bucket(n_chunks)
        strides = _varlen_strides(x, alpha, out)
        fwd_kernel = _moc_fwd_varlen_kernel
    else:
        B, T, D = x.shape
        n_chunks = 0
        n_chunks_bucket = 1
        chunk_indices = x
        cu_seqlens = x
        strides = _fixed_strides(x, alpha, out)
        fwd_kernel = _moc_fwd_fixed_kernel

    stride_x_b, stride_x_t, stride_x_d, stride_a_b, stride_a_t, stride_a_k, stride_o_b, stride_o_t, stride_o_d = strides

    if is_varlen:
        def grid(meta):
            return (n_chunks, 1, triton.cdiv(D, meta["BLOCK_D"]))
        fwd_kernel[grid](
            x, alpha, basis, out, cu_seqlens, chunk_indices,
            B, T, D, n_chunks_bucket,
            stride_x_b, stride_x_t, stride_x_d,
            stride_a_b, stride_a_t, stride_a_k,
            *basis.stride(),
            stride_o_b, stride_o_t, stride_o_d,
            BLOCK_T=block_t,
            BLOCK_K=_pick_block_k_fwd_dx(K), K=K, KS=KS,
            ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=True,
        )
    else:
        def grid(meta):
            return (B, triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(D, meta["BLOCK_D"]))
        fwd_kernel[grid](
            x, alpha, basis, out, cu_seqlens, chunk_indices,
            B, T, D, n_chunks_bucket,
            stride_x_b, stride_x_t, stride_x_d,
            stride_a_b, stride_a_t, stride_a_k,
            *basis.stride(),
            stride_o_b, stride_o_t, stride_o_d,
            BLOCK_K=_pick_block_k_fwd_dx(K), K=K, KS=KS,
            ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=False,
        )
    return out


def launch_preact(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    block_t: int = VARLEN_BLOCK_T,
) -> torch.Tensor:
    out = launch_forward(
        x, alpha, basis,
        activation_flag=0,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices, block_t=block_t,
    )
    return out


def launch_backward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    moc_dy: torch.Tensor,
    preact: torch.Tensor,
    *,
    activation_flag: int,
    need_x: bool,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    block_t: int = VARLEN_BLOCK_T,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    is_varlen = cu_seqlens is not None
    K, KS, _ = basis.shape
    bf16_compute = x.dtype == torch.bfloat16

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, block_t, cu_seqlens_cpu)
        _check_chunk_indices(chunk_indices, cu_seqlens)
        N, D = x.shape
        B = 0
        T = 0
        n_chunks = chunk_indices.shape[0]
        n_chunks_bucket = _autotune_bucket(n_chunks)
        stride_x_b, stride_x_t, stride_x_d = 0, x.stride(0), x.stride(1)
        stride_dy_b, stride_dy_t, stride_dy_d = 0, moc_dy.stride(0), moc_dy.stride(1)
        stride_a_b, stride_a_t, stride_a_k = 0, alpha.stride(0), alpha.stride(1)
        stride_z_b, stride_z_t, stride_z_d = 0, preact.stride(0), preact.stride(1)
        dx_kernel = _moc_bwd_dx_varlen_kernel
        reduction_kernel = _moc_bwd_dalpha_dbasis_varlen_kernel
    else:
        B, T, D = x.shape
        n_chunks = 0
        n_chunks_bucket = 1
        chunk_indices = x
        cu_seqlens = x
        stride_x_b, stride_x_t, stride_x_d = x.stride()
        stride_dy_b, stride_dy_t, stride_dy_d = moc_dy.stride()
        stride_a_b, stride_a_t, stride_a_k = alpha.stride()
        stride_z_b, stride_z_t, stride_z_d = preact.stride()
        dx_kernel = _moc_bwd_dx_fixed_kernel
        reduction_kernel = _moc_bwd_dalpha_dbasis_fixed_kernel

    dx = None
    if need_x:
        dx = torch.empty_like(x)
        if is_varlen:
            stride_dx_b, stride_dx_t, stride_dx_d = 0, dx.stride(0), dx.stride(1)
            def grid(meta):
                return (n_chunks, 1, triton.cdiv(D, meta["BLOCK_D"]))
            dx_kernel[grid](
                moc_dy, alpha, basis, dx, preact, cu_seqlens, chunk_indices,
                B, T, D, n_chunks_bucket,
                stride_dy_b, stride_dy_t, stride_dy_d,
                stride_a_b, stride_a_t, stride_a_k,
                *basis.stride(),
                stride_dx_b, stride_dx_t, stride_dx_d,
                stride_z_b, stride_z_t, stride_z_d,
                BLOCK_T=block_t,
                BLOCK_K=_pick_block_k_fwd_dx(K), K=K, KS=KS,
                ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=True,
            )
        else:
            stride_dx_b, stride_dx_t, stride_dx_d = dx.stride()
            def grid(meta):
                return (B, triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(D, meta["BLOCK_D"]))
            dx_kernel[grid](
                moc_dy, alpha, basis, dx, preact, cu_seqlens, chunk_indices,
                B, T, D, n_chunks_bucket,
                stride_dy_b, stride_dy_t, stride_dy_d,
                stride_a_b, stride_a_t, stride_a_k,
                *basis.stride(),
                stride_dx_b, stride_dx_t, stride_dx_d,
                stride_z_b, stride_z_t, stride_z_d,
                BLOCK_K=_pick_block_k_fwd_dx(K), K=K, KS=KS,
                ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=False,
            )

    dalpha = torch.empty_like(alpha)
    dbasis_accum = torch.zeros((K, KS, D), device=basis.device, dtype=torch.float32)
    dbasis_strides = dbasis_accum.stride()

    if is_varlen:
        stride_da_b, stride_da_t, stride_da_k = 0, dalpha.stride(0), dalpha.stride(1)
        def grid(meta):
            return (n_chunks, 1)
        reduction_kernel[grid](
            x, moc_dy, alpha, basis, dalpha, dbasis_accum, preact, cu_seqlens, chunk_indices,
            B, T, D, n_chunks_bucket,
            stride_x_b, stride_x_t, stride_x_d,
            stride_dy_b, stride_dy_t, stride_dy_d,
            stride_a_b, stride_a_t, stride_a_k,
            *basis.stride(),
            stride_da_b, stride_da_t, stride_da_k,
            *dbasis_strides,
            stride_z_b, stride_z_t, stride_z_d,
            BLOCK_T=block_t,
            BLOCK_K=_pick_block_k_bwd_reduction(K), K=K, KS=KS,
            ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=True,
        )
    else:
        stride_da_b, stride_da_t, stride_da_k = dalpha.stride()
        def grid(meta):
            return (B, triton.cdiv(T, meta["BLOCK_T"]))
        reduction_kernel[grid](
            x, moc_dy, alpha, basis, dalpha, dbasis_accum, preact, cu_seqlens, chunk_indices,
            B, T, D, n_chunks_bucket,
            stride_x_b, stride_x_t, stride_x_d,
            stride_dy_b, stride_dy_t, stride_dy_d,
            stride_a_b, stride_a_t, stride_a_k,
            *basis.stride(),
            stride_da_b, stride_da_t, stride_da_k,
            *dbasis_strides,
            stride_z_b, stride_z_t, stride_z_d,
            BLOCK_K=_pick_block_k_bwd_reduction(K), K=K, KS=KS,
            ACTIVATION=activation_flag, BF16_COMPUTE=bf16_compute, IS_VARLEN=False,
        )

    dbasis = dbasis_accum.to(basis.dtype)
    return dx, dalpha, dbasis


def launch_update(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    cache: torch.Tensor,
    *,
    activation_flag: int,
) -> torch.Tensor:
    B, D = x.shape
    K, KS, _ = basis.shape
    out = torch.empty_like(x)
    block_d = min(128, max(16, _next_power_of_2(D)))
    _moc_update_kernel[(B, triton.cdiv(D, block_d))](
        x, alpha, basis, cache, out, B, D,
        *x.stride(), *alpha.stride(), *basis.stride(), *cache.stride(), *out.stride(),
        BLOCK_D=block_d,
        BLOCK_K=_pick_block_k_fwd_dx(K), K=K, KS=KS,
        ACTIVATION=activation_flag,
        BF16_COMPUTE=x.dtype == torch.bfloat16,
        num_warps=4 if block_d >= 64 else 2,
    )
    return out

