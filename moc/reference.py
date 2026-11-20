from __future__ import annotations

import torch
import torch.nn.functional as F


def moc_routing_ref(
    router_logits: torch.Tensor,
    routing_activation: str = "exp",
    routing_norm: str = "l2",
) -> torch.Tensor:
    """Convert router logits to MoC mixture coefficients.

    Matches the routing choices in ``MoC``:
      - routing_activation: ``"exp"`` or ``"softplus"``
      - routing_norm: ``"l2"`` (default), ``"l1"``, or ``"none"``
    """
    routing_activation = routing_activation.lower()
    routing_norm = routing_norm.lower()

    if routing_activation == "exp":
        r = torch.exp(router_logits)
    elif routing_activation == "softplus":
        r = F.softplus(router_logits)
    else:
        raise ValueError(f"unsupported routing activation: {routing_activation}")

    if routing_norm == "none":
        return r
    if routing_norm == "l1":
        return F.normalize(r, p=1, dim=-1)
    if routing_norm == "l2":
        return F.normalize(r, p=2, dim=-1)
    raise ValueError(f"unsupported routing norm: {routing_norm}")


def _postprocess(
    out: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    if activation is not None:
        if activation not in ("silu", "swish"):
            raise ValueError(f"unsupported activation: {activation}")
        out = F.silu(out)
    return out


def moc_ref(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    activation: str | None = None,
) -> torch.Tensor:
    # x:     [B, T, D]
    # alpha: [B, T, K] (already routed, e.g. via moc_routing_ref)
    # basis: [K, KS, D]
    ks = basis.shape[1]
    x_pad = F.pad(x, (0, 0, ks - 1, 0))
    windows = x_pad.unfold(dimension=1, size=ks, step=1)  # [B, T, D, KS]
    dynamic_weight = torch.einsum("btk,ksd->btds", alpha, basis)
    out = (windows * dynamic_weight).sum(dim=-1)
    return _postprocess(out, activation)


def moc_varlen_ref(
    x: torch.Tensor,
    alpha: torch.Tensor,
    basis: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str | None = None,
) -> torch.Tensor:
    restore_batch_dim = x.ndim == 3
    if restore_batch_dim:
        x = x.squeeze(0)
        alpha = alpha.squeeze(0)

    boundaries = cu_seqlens.detach().cpu().tolist()
    chunks = [
        moc_ref(
            x[start:end].unsqueeze(0),
            alpha[start:end].unsqueeze(0),
            basis,
            activation=activation,
        ).squeeze(0)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    out = torch.cat(chunks, dim=0)
    return out.unsqueeze(0) if restore_batch_dim else out


__all__ = ["moc_routing_ref", "moc_ref", "moc_varlen_ref"]
