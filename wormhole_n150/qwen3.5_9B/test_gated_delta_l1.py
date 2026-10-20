# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn

from gated_delta_l1 import GatedDeltaRuleL1, plan_split
from reference import recurrent_gated_delta_rule


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def make_inputs(B, T, Hk, Hv, dk, dv, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, Hk, dk, generator=gen)
    k = torch.randn(B, T, Hk, dk, generator=gen)
    v = torch.randn(B, T, Hv, dv, generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, generator=gen) + 2.0)  # decay in (0, 1)
    beta = torch.sigmoid(torch.randn(B, T, Hv, generator=gen))
    return q, k, v, g, beta


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def check(name, got, want, min_pcc=0.999, max_rel=2e-2):
    p = pcc(got, want)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"{name}: pcc={p:.6f} rel_err={rel:.2e}")
    assert p > min_pcc and rel < max_rel, name


def test_plan():
    # Qwen3.5/3.6 GDN: 32 value heads -> 2 V-column slices per head over the 8x8 grid.
    p = plan_split(32, 128, 128, 64)
    assert (p.nsplit, p.units_per_core, p.num_cores) == (2, 1, 64)
    p = plan_split(16, 128, 128, 64)
    assert (p.nsplit, p.units_per_core, p.num_cores) == (4, 1, 64)
    p = plan_split(128, 128, 128, 64)
    assert (p.nsplit, p.units_per_core, p.num_cores) == (1, 2, 64)


@pytest.mark.parametrize(
    "B, Hk, Hv, T",
    [
        (1, 16, 32, 1),  # Qwen3.5-9B / Qwen3.6 GDN layer, one decode token
        (1, 16, 32, 45),  # multi-token, crosses a 32-token block boundary
        (1, 8, 16, 7),  # nsplit = 4
        (4, 16, 32, 3),  # batch 4 -> 128 heads, 2 heads per core
    ],
)
def test_matches_reference(device, B, Hk, Hv, T):
    dk = dv = 128
    q, k, v, g, beta = make_inputs(B, T, Hk, Hv, dk, dv)
    init = torch.randn(B, Hv, dk, dv) * 0.1
    want_o, want_s = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=init)

    op = GatedDeltaRuleL1(device, B * Hv, dk, dv, initial_state=init.reshape(B * Hv, dk, dv))
    got_o = op(q, k, v, g, beta)
    check("output", got_o, want_o)
    check("state", op.get_state(), want_s.reshape(B * Hv, dk, dv))


def test_long_sequence_slow_decay(device):
    """Decay close to 1 is where a biased state update compounds; keep it under 1%."""
    B, Hk, Hv, dk, dv, T = 1, 16, 32, 128, 128, 1024
    q, k, v, _, beta = make_inputs(B, T, Hk, Hv, dk, dv, seed=3)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, generator=torch.Generator().manual_seed(4)) + 5.0)
    want_o, want_s = recurrent_gated_delta_rule(q, k, v, g, beta)

    op = GatedDeltaRuleL1(device, B * Hv, dk, dv)
    check("long output", op(q, k, v, g, beta), want_o, max_rel=1e-2)
    check("long state", op.get_state(), want_s.reshape(B * Hv, dk, dv), max_rel=1e-2)


def test_decode_loop_keeps_state_in_l1(device):
    """Many single-token calls: the state never leaves L1 between them."""
    B, Hk, Hv, dk, dv, steps = 1, 16, 32, 128, 128, 64
    q, k, v, g, beta = make_inputs(B, steps, Hk, Hv, dk, dv, seed=1)
    want_o, want_s = recurrent_gated_delta_rule(q, k, v, g, beta)

    op = GatedDeltaRuleL1(device, B * Hv, dk, dv)
    state_addr = op.state.buffer_address()
    outs = [op(q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1], g[:, t : t + 1], beta[:, t : t + 1]) for t in range(steps)]
    assert op.state.buffer_address() == state_addr
    check("decode output", torch.cat(outs, dim=1), want_o)
    check("decode state", op.get_state(), want_s.reshape(B * Hv, dk, dv))
