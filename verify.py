from __future__ import annotations

import torch

from moc import moc_triton
from moc.reference import moc_ref, moc_varlen_ref


DTYPES = [torch.bfloat16, torch.float32]
KERNEL_SIZES = [2, 4]
MODES = [
    ("plain", None),
    ("silu", "silu"),
]

B, T, D, K = 2, 64, 64, 8
VARLEN_LENGTHS = [31, 17, 5, 23]
STEP_T = 8

TOL = {
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (1e-1, 1e-1),
}


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = TOL[dtype]
    try:
        torch.testing.assert_close(
            actual.detach().float().cpu(),
            expected.detach().float().cpu(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"{name} failed for {dtype}: {exc}") from exc


def _kernel_inputs(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    x_dtype: torch.dtype,
    alpha_dtype: torch.dtype,
    basis_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        x.to(x_dtype).detach().requires_grad_(),
        alpha.to(alpha_dtype).detach().requires_grad_(),
        basis.to(basis_dtype).detach().requires_grad_(),
    )


def _reference_inputs(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        x.detach().float().cpu().requires_grad_(),
        alpha.detach().to(compute_dtype).float().cpu().requires_grad_(),
        basis.detach().to(compute_dtype).float().cpu().requires_grad_(),
    )


def _check_backward(
    name: str,
    y_kernel: torch.Tensor,
    y_ref: torch.Tensor,
    kernel_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dy: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    y_ref.backward(dy.detach().float().cpu())
    y_kernel.backward(dy)

    _assert_close(f"{name} forward", y_kernel, y_ref, dtype)
    for grad_name, kernel_tensor, ref_tensor in zip(("dx", "dalpha", "dbasis"), kernel_inputs, ref_inputs):
        _assert_close(f"{name} {grad_name}", kernel_tensor.grad, ref_tensor.grad, dtype)


def _check_fixed(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    dy: torch.Tensor,
    dtype: torch.dtype,
    alpha_dtype: torch.dtype,
    basis_dtype: torch.dtype,
    activation: str | None,
) -> None:
    kx, ka, kb = _kernel_inputs(x, alpha, basis, dtype, alpha_dtype, basis_dtype)
    rx, ra, rb = _reference_inputs(kx, ka, kb, dtype)

    y_ref = moc_ref(
        rx,
        ra,
        rb,
        activation=activation,
    )
    y_kernel, _ = moc_triton(
        kx,
        ka,
        kb,
        activation=activation,
    )

    _check_backward(
        "fixed",
        y_kernel,
        y_ref,
        (kx, ka, kb),
        (rx, ra, rb),
        dy.to(dtype),
        dtype,
    )


def _check_varlen(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    dy: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dtype: torch.dtype,
    alpha_dtype: torch.dtype,
    basis_dtype: torch.dtype,
    activation: str | None,
) -> None:
    kx, ka, kb = _kernel_inputs(x, alpha, basis, dtype, alpha_dtype, basis_dtype)
    rx, ra, rb = _reference_inputs(kx, ka, kb, dtype)

    y_ref = moc_varlen_ref(
        rx,
        ra,
        rb,
        cu_seqlens.cpu(),
        activation=activation,
    )
    y_kernel, _ = moc_triton(
        kx,
        ka,
        kb,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        activation=activation,
    )

    _check_backward(
        "varlen",
        y_kernel,
        y_ref,
        (kx, ka, kb),
        (rx, ra, rb),
        dy.to(dtype),
        dtype,
    )

    y_batched, _ = moc_triton(
        kx.detach().unsqueeze(0),
        ka.detach().unsqueeze(0),
        kb.detach(),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        activation=activation,
    )
    _assert_close("varlen batched form", y_batched, y_ref.unsqueeze(0), dtype)


def _check_step(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    dtype: torch.dtype,
    alpha_dtype: torch.dtype,
    basis_dtype: torch.dtype,
    activation: str | None,
) -> None:
    kx = x[:, :STEP_T].to(dtype)
    ka = alpha[:, :STEP_T].to(alpha_dtype)
    kb = basis.to(basis_dtype)

    rx = kx.float().cpu()
    ra = ka.to(dtype).float().cpu()
    rb = kb.to(dtype).float().cpu()
    y_ref = moc_ref(
        rx,
        ra,
        rb,
        activation=activation,
    )

    cache = torch.zeros(B, D, basis.shape[1], device="cuda", dtype=dtype)
    outputs = []
    for t in range(STEP_T):
        y, cache = moc_triton(
            kx[:, t:t + 1],
            ka[:, t:t + 1],
            kb,
            cache=cache,
            output_final_state=True,
            activation=activation,
        )
        outputs.append(y)

    y_kernel = torch.cat(outputs, dim=1)
    expected_state = kx[:, -basis.shape[1]:].transpose(1, 2).contiguous()

    _assert_close("step forward", y_kernel, y_ref, dtype)
    _assert_close("step final state", cache, expected_state, dtype)


def _mixed_storage_dtypes(dtype: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    if dtype == torch.bfloat16:
        return torch.float32, torch.float16
    return torch.bfloat16, torch.float16


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("verification requires CUDA")

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, T, D, device="cuda", generator=g) * 0.5
    alpha = torch.softmax(torch.randn(B, T, K, device="cuda", generator=g), dim=-1)
    dy = torch.randn(B, T, D, device="cuda", generator=g) * 0.1

    offsets = [0]
    for length in VARLEN_LENGTHS:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    max_seqlen = max(VARLEN_LENGTHS)
    N = offsets[-1]

    x_varlen = torch.randn(N, D, device="cuda", generator=g) * 0.5
    alpha_varlen = torch.softmax(torch.randn(N, K, device="cuda", generator=g), dim=-1)
    dy_varlen = torch.randn(N, D, device="cuda", generator=g) * 0.1

    n = 0
    for dtype in DTYPES:
        for ks in KERNEL_SIZES:
            basis = torch.randn(K, ks, D, device="cuda", generator=g) * 0.1
            for mode_name, activation in MODES:
                _check_fixed(
                    x, alpha, basis, dy,
                    dtype, dtype, dtype,
                    activation,
                )
                _check_varlen(
                    x_varlen, alpha_varlen, basis, dy_varlen,
                    cu_seqlens, max_seqlen,
                    dtype, dtype, dtype,
                    activation,
                )
                _check_step(
                    x, alpha, basis,
                    dtype, dtype, dtype,
                    activation,
                )
                n += 3
                print(f"passed {dtype} KS={ks} mode={mode_name}")

        basis = torch.randn(K, 4, D, device="cuda", generator=g) * 0.1
        alpha_dtype, basis_dtype = _mixed_storage_dtypes(dtype)
        _check_fixed(
            x, alpha, basis, dy,
            dtype, alpha_dtype, basis_dtype,
            None,
        )
        _check_varlen(
            x_varlen, alpha_varlen, basis, dy_varlen,
            cu_seqlens, max_seqlen,
            dtype, alpha_dtype, basis_dtype,
            None,
        )
        _check_step(
            x, alpha, basis,
            dtype, alpha_dtype, basis_dtype,
            None,
        )
        n += 3
        print(f"passed {dtype} mixed alpha={alpha_dtype} basis={basis_dtype}")

    print(f"All {n} checks passed")


if __name__ == "__main__":
    main()
