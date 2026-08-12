from __future__ import annotations

import triton
import triton.language as tl

FWD_DX_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_D": block_d, "BLOCK_T": block_t}, num_warps=num_warps, num_stages=num_stages)
    for block_d in (32, 64, 128)
    for block_t in (32, 64)
    for num_warps in (1, 2, 4, 8)
    for num_stages in (2, 6)
]

DALPHA_DBASIS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_D": block_d, "BLOCK_T": block_t}, num_warps=num_warps, num_stages=num_stages)
    for block_d in (32, 64, 128)
    for block_t in (32, 64, 128)
    for num_warps in (2, 4, 8)
    for num_stages in (3,)
]

VARLEN_FWD_DX_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_D": block_d}, num_warps=num_warps, num_stages=num_stages)
    for block_d in (64, 128, 256)
    for num_warps in (2, 4, 8)
    for num_stages in (2, 6)
]

VARLEN_DALPHA_DBASIS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_D": block_d}, num_warps=num_warps, num_stages=num_stages)
    for block_d in (32, 64)
    for num_warps in (2, 4, 8)
    for num_stages in (3,)
]

@triton.jit
def _moc_fwd_kernel(
    x_ptr, alpha_ptr, basis_ptr, out_ptr, cu_ptr, chunk_ptr,
    B, T, D, N_CHUNKS_BUCKET,
    stride_x_b, stride_x_t, stride_x_d,
    stride_a_b, stride_a_t, stride_a_k,
    stride_w_k, stride_w_s, stride_w_d,
    stride_o_b, stride_o_t, stride_o_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    KS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BF16_COMPUTE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    if IS_VARLEN:
        pid_seq = tl.load(chunk_ptr + pid_b * 2).to(tl.int32)
        pid_chunk = tl.load(chunk_ptr + pid_b * 2 + 1).to(tl.int32)
        seq_start = tl.load(cu_ptr + pid_seq).to(tl.int64)
        seq_end = tl.load(cu_ptr + pid_seq + 1).to(tl.int64)
        seq_len = seq_end - seq_start
        batch_offset = 0
    else:
        pid_seq = pid_b
        pid_chunk = pid_t
        seq_start = 0
        seq_len = T
        batch_offset = pid_seq

    if IS_VARLEN:
        offs_t = pid_chunk.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = seq_start + offs_t
    else:
        offs_t = pid_chunk * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = offs_t
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    tmask = offs_t < seq_len
    dmask = offs_d < D
    kmask = offs_k < K

    a = tl.load(
        alpha_ptr
        + batch_offset * stride_a_b
        + token_t[:, None] * stride_a_t
        + offs_k[None, :] * stride_a_k,
        mask=tmask[:, None] & kmask[None, :],
        other=0.0,
    )
    if BF16_COMPUTE:
        a = a.to(tl.bfloat16)
    else:
        a = a.to(tl.float32)

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for s in tl.static_range(0, KS):
        x_local_t = offs_t - (KS - 1 - s)
        x_token_t = seq_start + x_local_t
        xmask = (
            (x_local_t[:, None] >= 0)
            & (x_local_t[:, None] < seq_len)
            & dmask[None, :]
        )

        x = tl.load(
            x_ptr
            + batch_offset * stride_x_b
            + x_token_t[:, None] * stride_x_t
            + offs_d[None, :] * stride_x_d,
            mask=xmask,
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            basis_ptr
            + offs_k[:, None] * stride_w_k
            + s * stride_w_s
            + offs_d[None, :] * stride_w_d,
            mask=kmask[:, None] & dmask[None, :],
            other=0.0,
        )
        if BF16_COMPUTE:
            w = w.to(tl.bfloat16)
        else:
            w = w.to(tl.float32)

        acc += x * tl.dot(a, w, input_precision="tf32")

    if ACTIVATION:
        acc *= tl.sigmoid(acc)

    tl.store(
        out_ptr
        + batch_offset * stride_o_b
        + token_t[:, None] * stride_o_t
        + offs_d[None, :] * stride_o_d,
        acc,
        mask=tmask[:, None] & dmask[None, :],
    )


_moc_fwd_fixed_kernel = triton.autotune(
    configs=FWD_DX_AUTOTUNE_CONFIGS,
    key=["B", "T", "D", "K", "KS"],
)(_moc_fwd_kernel)

_moc_fwd_varlen_kernel = triton.autotune(
    configs=VARLEN_FWD_DX_AUTOTUNE_CONFIGS,
    key=["N_CHUNKS_BUCKET", "D", "K", "KS", "BLOCK_T"],
)(_moc_fwd_kernel)


@triton.jit
def _moc_bwd_dx_kernel(
    dy_ptr, alpha_ptr, basis_ptr, dx_ptr, preact_ptr, cu_ptr, chunk_ptr,
    B, T, D, N_CHUNKS_BUCKET,
    stride_dy_b, stride_dy_t, stride_dy_d,
    stride_a_b, stride_a_t, stride_a_k,
    stride_w_k, stride_w_s, stride_w_d,
    stride_dx_b, stride_dx_t, stride_dx_d,
    stride_z_b, stride_z_t, stride_z_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    KS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BF16_COMPUTE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    if IS_VARLEN:
        pid_seq = tl.load(chunk_ptr + pid_b * 2).to(tl.int32)
        pid_chunk = tl.load(chunk_ptr + pid_b * 2 + 1).to(tl.int32)
        seq_start = tl.load(cu_ptr + pid_seq).to(tl.int64)
        seq_end = tl.load(cu_ptr + pid_seq + 1).to(tl.int64)
        seq_len = seq_end - seq_start
        batch_offset = 0
    else:
        pid_seq = pid_b
        pid_chunk = pid_t
        seq_start = 0
        seq_len = T
        batch_offset = pid_seq

    if IS_VARLEN:
        offs_t = pid_chunk.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = seq_start + offs_t
    else:
        offs_t = pid_chunk * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = offs_t
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    tmask = offs_t < seq_len
    dmask = offs_d < D
    kmask = offs_k < K

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for s in tl.static_range(0, KS):
        out_local_t = offs_t + (KS - 1 - s)
        out_token_t = seq_start + out_local_t
        out_mask = out_local_t < seq_len
        td_mask = out_mask[:, None] & dmask[None, :]

        dz = tl.load(
            dy_ptr
            + batch_offset * stride_dy_b
            + out_token_t[:, None] * stride_dy_t
            + offs_d[None, :] * stride_dy_d,
            mask=td_mask,
            other=0.0,
        ).to(tl.float32)

        if ACTIVATION:
            z = tl.load(
                preact_ptr
                + batch_offset * stride_z_b
                + out_token_t[:, None] * stride_z_t
                + offs_d[None, :] * stride_z_d,
                mask=td_mask,
                other=0.0,
            ).to(tl.float32)
            sig = tl.sigmoid(z)
            dz *= sig * (1.0 + z * (1.0 - sig))

        a = tl.load(
            alpha_ptr
            + batch_offset * stride_a_b
            + out_token_t[:, None] * stride_a_t
            + offs_k[None, :] * stride_a_k,
            mask=out_mask[:, None] & kmask[None, :],
            other=0.0,
        )

        w = tl.load(
            basis_ptr
            + offs_k[:, None] * stride_w_k
            + s * stride_w_s
            + offs_d[None, :] * stride_w_d,
            mask=kmask[:, None] & dmask[None, :],
            other=0.0,
        )

        if BF16_COMPUTE:
            a = a.to(tl.bfloat16)
            w = w.to(tl.bfloat16)
        else:
            a = a.to(tl.float32)
            w = w.to(tl.float32)

        acc += dz * tl.dot(a, w, input_precision="tf32")

    tl.store(
        dx_ptr
        + batch_offset * stride_dx_b
        + token_t[:, None] * stride_dx_t
        + offs_d[None, :] * stride_dx_d,
        acc,
        mask=tmask[:, None] & dmask[None, :],
    )


_moc_bwd_dx_fixed_kernel = triton.autotune(
    configs=FWD_DX_AUTOTUNE_CONFIGS,
    key=["B", "T", "D", "K", "KS"],
)(_moc_bwd_dx_kernel)

_moc_bwd_dx_varlen_kernel = triton.autotune(
    configs=VARLEN_FWD_DX_AUTOTUNE_CONFIGS,
    key=["N_CHUNKS_BUCKET", "D", "K", "KS", "BLOCK_T"],
)(_moc_bwd_dx_kernel)


@triton.jit
def _moc_bwd_dalpha_dbasis_dot_kernel(
    x_ptr, dy_ptr, alpha_ptr, basis_ptr,
    dalpha_ptr, dbasis_accum_ptr, preact_ptr, cu_ptr, chunk_ptr,
    B, T, D, N_CHUNKS_BUCKET,
    stride_x_b, stride_x_t, stride_x_d,
    stride_dy_b, stride_dy_t, stride_dy_d,
    stride_a_b, stride_a_t, stride_a_k,
    stride_w_k, stride_w_s, stride_w_d,
    stride_da_b, stride_da_t, stride_da_k,
    stride_db_k, stride_db_s, stride_db_d,
    stride_z_b, stride_z_t, stride_z_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    KS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BF16_COMPUTE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    if IS_VARLEN:
        pid_seq = tl.load(chunk_ptr + pid_b * 2).to(tl.int32)
        pid_chunk = tl.load(chunk_ptr + pid_b * 2 + 1).to(tl.int32)
        seq_start = tl.load(cu_ptr + pid_seq).to(tl.int64)
        seq_end = tl.load(cu_ptr + pid_seq + 1).to(tl.int64)
        seq_len = seq_end - seq_start
        batch_offset = 0
    else:
        pid_seq = pid_b
        pid_chunk = pid_t
        seq_start = 0
        seq_len = T
        batch_offset = pid_seq

    if IS_VARLEN:
        offs_t = pid_chunk.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = seq_start + offs_t
    else:
        offs_t = pid_chunk * BLOCK_T + tl.arange(0, BLOCK_T)
        token_t = offs_t
    offs_k = tl.arange(0, BLOCK_K)

    tmask = offs_t < seq_len
    kmask = offs_k < K

    a = tl.load(
        alpha_ptr
        + batch_offset * stride_a_b
        + token_t[:, None] * stride_a_t
        + offs_k[None, :] * stride_a_k,
        mask=tmask[:, None] & kmask[None, :],
        other=0.0,
    )
    if BF16_COMPUTE:
        a = a.to(tl.bfloat16)
    else:
        a = a.to(tl.float32)

    dalpha_acc = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)

    for d0 in tl.range(0, D, BLOCK_D, num_stages=1):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        dmask = offs_d < D
        td_mask = tmask[:, None] & dmask[None, :]

        dz = tl.load(
            dy_ptr
            + batch_offset * stride_dy_b
            + token_t[:, None] * stride_dy_t
            + offs_d[None, :] * stride_dy_d,
            mask=td_mask,
            other=0.0,
        ).to(tl.float32)

        if ACTIVATION:
            z = tl.load(
                preact_ptr
                + batch_offset * stride_z_b
                + token_t[:, None] * stride_z_t
                + offs_d[None, :] * stride_z_d,
                mask=td_mask,
                other=0.0,
            ).to(tl.float32)
            sig = tl.sigmoid(z)
            dz *= sig * (1.0 + z * (1.0 - sig))

        for s in tl.static_range(0, KS):
            x_local_t = offs_t - (KS - 1 - s)
            x_token_t = seq_start + x_local_t
            xmask = (
                (x_local_t[:, None] >= 0)
                & (x_local_t[:, None] < seq_len)
                & dmask[None, :]
            )

            x = tl.load(
                x_ptr
                + batch_offset * stride_x_b
                + x_token_t[:, None] * stride_x_t
                + offs_d[None, :] * stride_x_d,
                mask=xmask,
                other=0.0,
            ).to(tl.float32)

            m = dz * x
            if BF16_COMPUTE:
                m = m.to(tl.bfloat16)

            w = tl.load(
                basis_ptr
                + offs_k[:, None] * stride_w_k
                + s * stride_w_s
                + offs_d[None, :] * stride_w_d,
                mask=kmask[:, None] & dmask[None, :],
                other=0.0,
            )
            if BF16_COMPUTE:
                w = w.to(tl.bfloat16)
            else:
                w = w.to(tl.float32)
            dalpha_acc += tl.dot(m, tl.trans(w), input_precision="tf32")

            dbasis_partial = tl.dot(tl.trans(a), m, input_precision="tf32")
            tl.atomic_add(
                dbasis_accum_ptr
                + offs_k[:, None] * stride_db_k
                + s * stride_db_s
                + offs_d[None, :] * stride_db_d,
                dbasis_partial,
                mask=kmask[:, None] & dmask[None, :],
                sem="relaxed",
            )

    tl.store(
        dalpha_ptr
        + batch_offset * stride_da_b
        + token_t[:, None] * stride_da_t
        + offs_k[None, :] * stride_da_k,
        dalpha_acc,
        mask=tmask[:, None] & kmask[None, :],
    )


_moc_bwd_dalpha_dbasis_fixed_kernel = triton.autotune(
    configs=DALPHA_DBASIS_AUTOTUNE_CONFIGS,
    key=["B", "T", "D", "K", "KS"],
    reset_to_zero=["dbasis_accum_ptr"],
)(_moc_bwd_dalpha_dbasis_dot_kernel)

_moc_bwd_dalpha_dbasis_varlen_kernel = triton.autotune(
    configs=VARLEN_DALPHA_DBASIS_AUTOTUNE_CONFIGS,
    key=["N_CHUNKS_BUCKET", "D", "K", "KS", "BLOCK_T"],
    reset_to_zero=["dbasis_accum_ptr"],
)(_moc_bwd_dalpha_dbasis_dot_kernel)


@triton.jit
def _moc_update_kernel(
    x_ptr, alpha_ptr, basis_ptr, cache_ptr, out_ptr,
    B, D,
    stride_x_b, stride_x_d,
    stride_a_b, stride_a_k,
    stride_w_k, stride_w_s, stride_w_d,
    stride_c_b, stride_c_d, stride_c_s,
    stride_o_b, stride_o_d,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    KS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BF16_COMPUTE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    dmask = offs_d < D
    kmask = offs_k < K

    a = tl.load(
        alpha_ptr
        + pid_b * stride_a_b
        + offs_k[:, None] * stride_a_k,
        mask=kmask[:, None],
        other=0.0,
    )
    if BF16_COMPUTE:
        a = a.to(tl.bfloat16)
    else:
        a = a.to(tl.float32)

    x = tl.load(
        x_ptr
        + pid_b * stride_x_b
        + offs_d[None, :] * stride_x_d,
        mask=dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((1, BLOCK_D), dtype=tl.float32)

    for s in tl.static_range(0, KS):
        if s == KS - 1:
            state = x
        else:
            state = tl.load(
                cache_ptr
                + pid_b * stride_c_b
                + offs_d[None, :] * stride_c_d
                + (s + 1) * stride_c_s,
                mask=dmask[None, :],
                other=0.0,
            ).to(tl.float32)

        w = tl.load(
            basis_ptr
            + offs_k[:, None] * stride_w_k
            + s * stride_w_s
            + offs_d[None, :] * stride_w_d,
            mask=kmask[:, None] & dmask[None, :],
            other=0.0,
        )
        if BF16_COMPUTE:
            w = w.to(tl.bfloat16)
        else:
            w = w.to(tl.float32)
        acc += state * tl.sum(a * w, axis=0, dtype=tl.float32)[None, :]

    if ACTIVATION:
        acc *= tl.sigmoid(acc)

    tl.store(
        out_ptr
        + pid_b * stride_o_b
        + offs_d[None, :] * stride_o_d,
        acc,
        mask=dmask[None, :],
    )

    for s in tl.static_range(0, KS - 1):
        state = tl.load(
            cache_ptr
            + pid_b * stride_c_b
            + offs_d[None, :] * stride_c_d
            + (s + 1) * stride_c_s,
            mask=dmask[None, :],
            other=0.0,
        )
        tl.store(
            cache_ptr
            + pid_b * stride_c_b
            + offs_d[None, :] * stride_c_d
            + s * stride_c_s,
            state,
            mask=dmask[None, :],
        )

    tl.store(
        cache_ptr
        + pid_b * stride_c_b
        + offs_d[None, :] * stride_c_d
        + (KS - 1) * stride_c_s,
        x,
        mask=dmask[None, :],
    )


