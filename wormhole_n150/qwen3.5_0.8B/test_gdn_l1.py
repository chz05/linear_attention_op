# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-0.8B GDN layers, decode only, all weights in L1, vs Hugging Face Qwen3_5GatedDeltaNet."""
import pytest
import torch
import ttnn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from gdn_l1 import Qwen35_08B_GDNStackL1

QWEN35_08B = dict(
    hidden_size=1024, linear_num_key_heads=16, linear_num_value_heads=16,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def quantize(w, dtype, device):
    """The exact values the device sees for a weight stored as `dtype` (tile layout)."""
    if dtype == ttnn.bfloat16:
        return w.bfloat16().float()
    return ttnn.to_torch(ttnn.from_torch(w.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)).float()


def make_hf_layers(n, seed=0, weight_dtype=ttnn.bfloat16, device=None):
    """Random 0.8B GDN layers whose projection weights are already rounded to `weight_dtype`,
    so comparisons measure the kernel rather than weight quantisation."""
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(**QWEN35_08B)
    layers = []
    for i in range(n):
        L = Qwen3_5GatedDeltaNet(cfg, layer_idx=i).eval()
        with torch.no_grad():
            L.norm.weight.uniform_(0.5, 1.5)
            L.dt_bias.uniform_(-1.0, 1.0)
            L.norm.weight.copy_(L.norm.weight.bfloat16().float())
            for p in (L.in_proj_qkv, L.in_proj_z, L.in_proj_a, L.in_proj_b):
                # the device stores W^T sliced by core; quantisation blocks run along rows of W^T
                p.weight.copy_(quantize(p.weight.T.contiguous(), weight_dtype, device).T)
            # out_proj: the device quantises W_o^T with the RMSNorm weight folded into its rows
            nw = L.norm.weight.repeat(L.out_proj.weight.shape[1] // L.norm.weight.numel())[:, None]
            folded = quantize((L.out_proj.weight.T * nw).contiguous(), weight_dtype, device)
            L.out_proj.weight.copy_((folded / nw).T)
            L.conv1d.weight.copy_(L.conv1d.weight.bfloat16().float())
        layers.append(L)
    return layers


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def check(name, got, want, min_pcc=0.9999, max_rel=1.5e-2):
    p = pcc(got, want)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"{name}: pcc={p:.6f} rel_err={rel:.2e}")
    assert p > min_pcc and rel < max_rel, name


def run_decode(device, n_layers, B, weight_dtype, steps=8, seed=1, weights_in_dram=False):
    """n_layers layers resident in L1 at once, decoding `steps` tokens one call at a time from zero state.
    Every layer gets the same inputs as its HF counterpart (run on the whole sequence, which is the
    same math) and keeps its own L1 state."""
    hf = make_hf_layers(n_layers, seed=seed, weight_dtype=weight_dtype, device=device)
    x = torch.randn(B, steps, 1024).bfloat16().float()
    with torch.no_grad():
        want = [L(x) for L in hf]
    tt = Qwen35_08B_GDNStackL1(device, hf, batch=B, weight_dtype=weight_dtype, weights_in_dram=weights_in_dram)
    got = [[] for _ in range(n_layers)]
    for t in range(steps):
        for i in range(n_layers):
            got[i].append(tt(x[:, t], layers=[i]))
    return [torch.stack(g, dim=1) for g in got], want


@pytest.mark.parametrize("B", [1, 3, 4, 8, 16])
def test_decode_one_layer_bf16(device, B):
    got, want = run_decode(device, 1, B, ttnn.bfloat16)
    check(f"decode 1 layer bf16 B={B}", got[0], want[0])


@pytest.mark.parametrize("n_layers, B, dtype", [(2, 1, ttnn.bfloat16), (3, 1, ttnn.bfloat8_b), (3, 4, ttnn.bfloat8_b), (6, 1, ttnn.bfloat4_b)])
def test_decode_resident_layers(device, n_layers, B, dtype):
    got, want = run_decode(device, n_layers, B, dtype)
    for i in range(n_layers):
        check(f"decode {n_layers} resident layers {dtype} B={B}, layer {i}", got[i], want[i])


def run_shrinking(device, n_layers, weight_dtype, lengths, seed=2, rounds=1):
    """A batch of len(lengths) sequences decoding together; sequence s finishes after lengths[s] tokens, and the
    batch shrinks as sequences finish (retire), down to 0. Each step's rows must match HF run on that sequence's
    own tokens. With rounds > 1 the stack is reset (full batch, zero states) and runs it again. Returns the batch
    sizes seen, one per step."""
    steps = max(lengths)
    hf = make_hf_layers(n_layers, seed=seed, weight_dtype=weight_dtype, device=device)
    x = torch.randn(len(lengths), steps, 1024).bfloat16().float()
    with torch.no_grad():
        want = [L(x) for L in hf]
    tt = Qwen35_08B_GDNStackL1(device, hf, batch=len(lengths), weight_dtype=weight_dtype)
    for n in range(rounds):
        if n > 0:
            tt.reset_state()
        sizes = []
        for t in range(steps + 1):
            done = [r for r, s in enumerate(tt.seq_ids) if lengths[s] == t]
            tt.retire(done[::-1] if t % 2 else done)  # any order
            assert tt.seq_ids == [s for s in range(len(lengths)) if lengths[s] > t]
            ids = tt.seq_ids
            if not ids:
                assert tt(torch.zeros(0, 1024)).shape == (0, 1024)
                break
            sizes.append(len(ids))
            for i in range(n_layers):
                check(f"round {n} step {t} B={len(ids)} layer {i}", tt(x[ids, t], layers=[i]), want[i][ids, t])
        assert tt.batch == 0
    return sizes


def test_decode_shrinking_batch(device):
    """16 -> 10 -> 8 -> 5 -> 3 -> 2 -> 1 -> 0, with random sequences finishing at each step down."""
    sizes = [16, 16, 10, 10, 8, 8, 5, 3, 3, 2, 1, 1]
    order = torch.randperm(16, generator=torch.Generator().manual_seed(5)).tolist()  # finishing order
    lengths = [0] * 16
    for t, b in enumerate(sizes + [0]):
        for s in order[b:]:
            lengths[s] = lengths[s] or t
    assert run_shrinking(device, 1, ttnn.bfloat8_b, lengths) == sizes


def test_decode_shrinking_batch_to_zero(device):
    """16 for 3 steps, one sequence finishes (15), then the other 15 all finish at once (15 -> 0); then the stack is
    reset to a fresh batch of 16 and the same happens again."""
    lengths = [8] * 16
    lengths[6] = 3
    assert run_shrinking(device, 1, ttnn.bfloat8_b, lengths, rounds=2) == [16] * 3 + [15] * 5


@pytest.mark.parametrize("n_layers, dtype, seed", [(1, ttnn.bfloat8_b, 11), (2, ttnn.bfloat4_b, 12)])
def test_decode_shrinking_batch_random(device, n_layers, dtype, seed):
    """Every sequence of a batch of 16 finishes after a random number of tokens."""
    lengths = torch.randint(1, 25, (16,), generator=torch.Generator().manual_seed(seed)).tolist()
    sizes = run_shrinking(device, n_layers, dtype, lengths, seed=seed)
    print("batch per step:", sizes)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_decode_weights_in_dram(device, dtype):
    """Comparison mode: same kernel, weights streamed from DRAM every step."""
    got, want = run_decode(device, 2, 2, dtype, weights_in_dram=True)
    for i in range(2):
        check(f"decode weights in DRAM {dtype}, layer {i}", got[i], want[i])
