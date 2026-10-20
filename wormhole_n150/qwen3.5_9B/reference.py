# SPDX-License-Identifier: Apache-2.0
"""Plain PyTorch gated delta rule, same math as HF Qwen3-Next `torch_recurrent_gated_delta_rule`."""

import torch


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None, use_qk_l2norm=True):
    """q, k: [B, T, Hk, dk]  v: [B, T, Hv, dv]  g, beta: [B, T, Hv]

    Returns (o [B, T, Hv, dv], final_state [B, Hv, dk, dv]), all in fp32.
    """
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    B, T, Hk, dk = q.shape
    Hv, dv = v.shape[2], v.shape[3]
    if Hv != Hk:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)
    q = q * dk**-0.5

    S = torch.zeros(B, Hv, dk, dv) if initial_state is None else initial_state.float().clone()
    o = torch.zeros(B, T, Hv, dv)
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        S = S * g[:, t].exp()[..., None, None]
        kv = (S * k_t[..., None]).sum(-2)
        delta = (v_t - kv) * beta[:, t][..., None]
        S = S + k_t[..., None] * delta[..., None, :]
        o[:, t] = (S * q_t[..., None]).sum(-2)
    return o, S
