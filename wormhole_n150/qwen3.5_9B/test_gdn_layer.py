# SPDX-License-Identifier: Apache-2.0
"""Full GDN layer on device vs the Hugging Face Qwen3_5GatedDeltaNet (torch fallback path)."""
import pytest
import torch
import ttnn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from gdn_layer import Qwen35GatedDeltaNetTT


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def make_hf_layer(seed=0):
    torch.manual_seed(seed)
    layer = Qwen3_5GatedDeltaNet(Qwen3_5TextConfig(), layer_idx=0).eval()
    with torch.no_grad():
        layer.norm.weight.uniform_(0.5, 1.5)
        layer.dt_bias.uniform_(-1.0, 1.0)
        # Round the matmul weights to bf16 so the comparison measures the kernel, not weight quantisation.
        for lin in (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_a, layer.in_proj_b, layer.out_proj):
            lin.weight.copy_(lin.weight.bfloat16().float())
    return layer


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def check(name, got, want, min_pcc=0.9999, max_rel=1.5e-2):
    p = pcc(got, want)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"{name}: pcc={p:.6f} rel_err={rel:.2e}")
    assert p > min_pcc and rel < max_rel, name


@pytest.mark.parametrize("B, T", [(1, 1), (1, 40), (4, 9)])
def test_layer_matches_hf(device, B, T):
    hf = make_hf_layer()
    x = torch.randn(B, T, hf.hidden_size).bfloat16().float()
    with torch.no_grad():
        want = hf(x)
    tt = Qwen35GatedDeltaNetTT(device, hf, batch=B)
    check(f"layer B={B} T={T}", tt(x), want)


def test_layer_decode_continues_from_l1_state(device):
    """Prefill 20 tokens, then decode 12 tokens one call at a time; compare with HF on all 32."""
    hf = make_hf_layer(seed=1)
    B, T0, steps = 1, 20, 12
    x = torch.randn(B, T0 + steps, hf.hidden_size).bfloat16().float()
    with torch.no_grad():
        want = hf(x)
    tt = Qwen35GatedDeltaNetTT(device, hf, batch=B)
    outs = [tt(x[:, :T0])] + [tt(x[:, t : t + 1]) for t in range(T0, T0 + steps)]
    got = torch.cat(outs, dim=1)
    check("prefill part", got[:, :T0], want[:, :T0])
    check("decode part", got[:, T0:], want[:, T0:])


def test_long_prefill_is_chunked(device):
    """300 tokens -> 3 launches of 128 rows; state and conv history carry across launches."""
    hf = make_hf_layer(seed=2)
    x = torch.randn(1, 300, hf.hidden_size).bfloat16().float()
    with torch.no_grad():
        want = hf(x)
    tt = Qwen35GatedDeltaNetTT(device, hf, batch=1)
    check("long prefill", tt(x), want)
