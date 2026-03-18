import argparse
from math import inf

from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.flashattn_combine import FlashAttnCombine
from software_model.layernorm import FusedLayerNorm
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from test.flashattn.utils import get_model_shape as get_fa_shape
from test.flashattn.utils import get_output_dtype as get_fa_output_dtype
from test.layernorm.utils import get_output_dtype as get_ln_output_dtype
from test.matmul.utils import get_model_shape as get_matmul_shapes
from test.matmul.utils import get_output_dtype as get_matmul_output_dtype
from test.utils import test_model_dict

"""LLM inference latency simulator.

Combines software_model operators to simulate Transformer inference.
Layer: QKV_proj -> FlashAttn -> O_proj -> LayerNorm -> GateUp_proj -> Down_proj -> LayerNorm

Usage:
  python -m simulate.main --device Orin --model Qwen3_1_7B --precision fp16 --phase prefill --seq_len 512
  python -m simulate.main --device Orin --model Qwen3_8B --precision int8 --phase decode --seq_len 1024 --spec_tokens 64
"""


def _act_dtype_from_precision(precision: str):
    """Base activation dtype from precision. Orin: fp16/int8/int4. Thor: fp8/fp4."""
    if precision == "int4":
        return data_type_dict["fp16"]  # Marlin
    return data_type_dict[precision]


def run_matmul(M, K, N, act_dt, wt_dt, mid_dt, out_dt, device, pcb, l2_prev=None):
    """Run [M,K]x[K,N] matmul. Falls back to roofline if M or K too small."""
    mm = Matmul(act_dt, wt_dt, mid_dt, out_dt, device=device)
    mm(Tensor([M, K], dtype=act_dt), Tensor([K, N], dtype=wt_dt))
    min_m = 16
    if M >= min_m and K >= 256:
        lat = (
            mm.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
            + pcb.compute_module.launch_latency.matmul
        )
    else:
        print("Warning: M or K too small, using roofline model")
        lat = mm.roofline_model(pcb)
    return lat, mm


def run_layer(
    model_name,
    model_cfg,
    precision,
    pcb,
    device,
    M,
    seq_len_q,
    seq_len_kv,
    is_prefill,
    is_causal,
):
    """Simulate in one pass: QKV -> FA -> O -> LN -> GateUp -> Down -> LN -> QKV.
    Returns (lat_first, lat_rest):
      lat_first = first QKV through post_ffn LN (layer 1)
      lat_rest = FA through second QKV (layer 2+)
    """
    K_shapes, N_shapes = get_matmul_shapes(model_name)
    d, nq, nkv = get_fa_shape(model_name)
    H = model_cfg["hidden_size"]

    base_act = _act_dtype_from_precision(precision)
    fp32 = data_type_dict["fp32"]
    int32 = data_type_dict["int32"]
    wt_dt = (
        data_type_dict["int8"]
        if precision == "int8"
        else (data_type_dict["int4"] if precision == "int4" else base_act)
    )
    mid_dt = int32 if precision == "int8" else fp32

    # Activation flow: qkv_out -> FlashAttn -> attn_out -> o_out -> ln -> up_act -> up_out -> dn_act -> dn_out -> ln
    qkv_act, qkv_out = (
        base_act,
        get_matmul_output_dtype(base_act, "qkv_proj", is_test=False),
    )
    qkv_dt = qkv_out  # FlashAttn receives QKV proj output
    attn_out_dt = get_fa_output_dtype(base_act, is_test=False)
    o_act, o_out = (
        attn_out_dt,
        get_matmul_output_dtype(attn_out_dt, "o_proj", is_test=False),
    )
    ln_out_dt = get_ln_output_dtype(
        base_act, is_test=False
    )  # LN output = up_proj input
    up_act, up_out = (
        ln_out_dt,
        get_matmul_output_dtype(ln_out_dt, "up_proj", is_test=False),
    )
    dn_act, dn_out = up_out, get_matmul_output_dtype(up_out, "down_proj", is_test=False)

    l2_prev = None
    lat_first = 0.0
    lat_rest = 0.0
    breakdown = {}

    def _log(op_name, lat_ms):
        breakdown[op_name] = lat_ms
        print(f"    {op_name:<18} {lat_ms:>8.4f} ms")

    # 1. QKV proj (L2 cold)
    print(
        f"  [1/8] QKV_proj  [M={M}, K={K_shapes['qkv_proj']}, N={N_shapes['qkv_proj']}] (L2 cold)"
    )
    lat, mm = run_matmul(
        M,
        K_shapes["qkv_proj"],
        N_shapes["qkv_proj"],
        qkv_act,
        wt_dt,
        mid_dt,
        qkv_out,
        device,
        pcb,
        l2_prev,
    )
    l2_prev = getattr(mm, "l2_status", None)
    lat_first += lat
    _log("qkv_proj", lat * 1000)

    # 2. FlashAttn
    print(
        f"  [2/8] FlashAttn  [seq_q={seq_len_q}, seq_kv={seq_len_kv}, heads={nq}/{nkv}, causal={is_causal}]"
    )
    num_splits_list = [1] if is_prefill else [1, 2, 4]
    lat_fa_combine = inf
    for num_splits in num_splits_list:
        temp_out = fp32 if num_splits > 1 else attn_out_dt
        fa = FlashAttn(
            qkv_dt, fp32, temp_out, is_prefill, is_causal, num_splits, device=device
        )
        fa(
            Tensor([seq_len_q, nq, d], qkv_dt),
            Tensor([seq_len_kv, nkv, d], qkv_dt),
            Tensor([seq_len_kv, nkv, d], qkv_dt),
        )
        launch = (
            pcb.compute_module.launch_latency.flashattn_prefill
            if is_prefill
            else pcb.compute_module.launch_latency.flashattn_decode
        )
        lat_fa_combine_temp = (
            fa.compile_and_simulate(pcb, L2Cache_previous=l2_prev) + launch
        )

        if num_splits > 1:
            combine = FlashAttnCombine(fp32, attn_out_dt)
            combine(Tensor([seq_len_q, nq * d, num_splits], fp32))
            lat_fa_combine_temp += (
                combine.compile_and_simulate(pcb, L2Cache_previous=fa.l2_status)
                + pcb.compute_module.launch_latency.flashattn_combine
            )
        if lat_fa_combine_temp < lat_fa_combine:
            lat_fa_combine = lat_fa_combine_temp
            l2_prev = combine.l2_status if num_splits > 1 else fa.l2_status

    lat_first += lat_fa_combine
    lat_rest += lat_fa_combine
    _log("flash_attn", lat_fa_combine * 1000)

    # 3. O proj
    print(f"  [3/8] O_proj  [M={M}, K={K_shapes['o_proj']}, N={N_shapes['o_proj']}]")
    lat, mm = run_matmul(
        M,
        K_shapes["o_proj"],
        N_shapes["o_proj"],
        o_act,
        wt_dt,
        mid_dt,
        o_out,
        device,
        pcb,
        l2_prev,
    )
    l2_prev = getattr(mm, "l2_status", None)
    lat_first += lat
    lat_rest += lat
    _log("o_proj", lat * 1000)

    # 4. Post-attn LayerNorm
    print(f"  [4/8] PostAttn_LayerNorm  [M={M}, N={H}]")
    ln = FusedLayerNorm(o_out, up_act)
    ln(Tensor([M, H], o_out), Tensor([M, H], o_out))
    lat = (
        ln.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
        + pcb.compute_module.launch_latency.layernorm
    )
    l2_prev = ln.l2_status
    lat_first += lat
    lat_rest += lat
    _log("post_attn_ln", lat * 1000)

    # 5. Gate+Up proj
    print(
        f"  [5/8] GateUp_proj  [M={M}, K={K_shapes['up_proj']}, N={N_shapes['up_proj']}]"
    )
    lat, mm = run_matmul(
        M,
        K_shapes["up_proj"],
        N_shapes["up_proj"],
        up_act,
        wt_dt,
        mid_dt,
        up_out,
        device,
        pcb,
        l2_prev,
    )
    l2_prev = getattr(mm, "l2_status", None)
    lat_first += lat
    lat_rest += lat
    _log("gate_up_proj", lat * 1000)

    # 6. Down proj
    print(
        f"  [6/8] Down_proj  [M={M}, K={K_shapes['down_proj']}, N={N_shapes['down_proj']}]"
    )
    lat, mm = run_matmul(
        M,
        K_shapes["down_proj"],
        N_shapes["down_proj"],
        dn_act,
        wt_dt,
        mid_dt,
        dn_out,
        device,
        pcb,
        l2_prev,
    )
    l2_prev = getattr(mm, "l2_status", None)
    lat_first += lat
    lat_rest += lat
    _log("down_proj", lat * 1000)

    # 7. Post-FFN LayerNorm
    print(f"  [7/8] PostFFN_LayerNorm  [M={M}, N={H}]")
    ln = FusedLayerNorm(dn_out, ln_out_dt)
    ln(Tensor([M, H], dn_out), Tensor([M, H], dn_out))
    lat = (
        ln.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
        + pcb.compute_module.launch_latency.layernorm
    )
    l2_prev = ln.l2_status
    lat_first += lat
    lat_rest += lat
    _log("post_ffn_ln", lat * 1000)

    # 8. QKV proj (L2 warm, receives from post_ffn_ln)
    print(
        f"  [8/8] QKV_proj  [M={M}, K={K_shapes['qkv_proj']}, N={N_shapes['qkv_proj']}] (L2 warm)"
    )
    lat, mm = run_matmul(
        M,
        K_shapes["qkv_proj"],
        N_shapes["qkv_proj"],
        qkv_act,
        wt_dt,
        mid_dt,
        qkv_out,
        device,
        pcb,
        l2_prev,
    )
    lat_rest += lat
    _log("qkv_proj_2", lat * 1000)

    if breakdown:
        print(
            f"  --- lat_first: {lat_first * 1000:.4f} ms, lat_rest: {lat_rest * 1000:.4f} ms ---"
        )
        # Summary table: use qkv_proj warm value, show latency and share
        summary_ops = [
            ("qkv_proj", breakdown.get("qkv_proj_2", breakdown.get("qkv_proj", 0))),
            ("flash_attn", breakdown.get("flash_attn", 0)),
            ("flashattn_combine", breakdown.get("flashattn_combine", 0)),
            ("o_proj", breakdown.get("o_proj", 0)),
            ("post_attn_ln", breakdown.get("post_attn_ln", 0)),
            ("gate_up_proj", breakdown.get("gate_up_proj", 0)),
            ("down_proj", breakdown.get("down_proj", 0)),
            ("post_ffn_ln", breakdown.get("post_ffn_ln", 0)),
        ]
        total_ms = lat_rest * 1000
        print("  Operator latency summary (qkv_proj=warm):")
        print(f"  {'Operator':<20} {'Latency(ms)':>12} {'Share':>8}")
        print("  " + "-" * 42)
        for name, lat_ms in summary_ops:
            if name == "flashattn_combine" and lat_ms <= 0:
                continue
            pct = (lat_ms / total_ms * 100) if total_ms > 0 else 0
            print(f"  {name:<20} {lat_ms:>12.4f} {pct:>7.1f}%")

    return lat_first, lat_rest


def resolve_dims(args, is_prefill):
    if is_prefill:
        return args.batch_size * args.seq_len, args.seq_len, args.seq_len
    return args.batch_size * args.spec_tokens, args.spec_tokens, args.seq_len


def main():
    parser = argparse.ArgumentParser(description="LLM inference latency simulator")
    parser.add_argument("--device", default="Orin", choices=["Orin", "Thor"])
    parser.add_argument(
        "--model", default="Qwen3_1_7B", choices=list(test_model_dict.keys())
    )
    parser.add_argument(
        "--precision", default="fp16", choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--spec_tokens", type=int, default=64, choices=[64, 128])
    args = parser.parse_args()

    is_prefill = args.phase == "prefill"
    if not is_prefill and args.is_causal:
        parser.error("decode does not support --is_causal")
    if args.batch_size < 1 or args.seq_len < 1:
        parser.error("batch_size and seq_len must be >= 1")
    if not is_prefill:
        min_m = 64 if args.device == "Thor" else 32
        if args.batch_size * args.spec_tokens < min_m:
            parser.error(f"decode: batch_size * spec_tokens must be >= {min_m}")
    if args.device not in device_dict:
        parser.error(f"Unknown device {args.device}")
    if args.device == "Orin" and args.precision not in ("fp16", "int8", "int4"):
        parser.error("Orin supports fp16/int8/int4 only")
    if args.device == "Thor" and args.precision not in ("fp8", "fp4"):
        parser.error("Thor supports fp8/fp4 only")

    pcb = device_dict[args.device]

    M, seq_len_q, seq_len_kv = resolve_dims(args, is_prefill)
    print("=" * 56)
    print("  LLM Inference Latency Simulator")
    print("=" * 56)
    print(f"  device:      {args.device}")
    print(f"  model:       {args.model}")
    print(f"  precision:   {args.precision}")
    print(f"  phase:       {args.phase}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  seq_len:     {args.seq_len}")
    if not is_prefill:
        print(f"  spec_tokens: {args.spec_tokens}")
    model_cfg = test_model_dict[args.model]
    num_layers = model_cfg["num_layers"]
    print(f"  num_layers:  {num_layers}")
    print(f"  M:           {M}  (matmul batch dim)")
    print(f"  seq_len_q:   {seq_len_q}")
    print(f"  seq_len_kv:  {seq_len_kv}")
    print("-" * 56)
    print("  Simulating (QKV->FA->O->LN->GateUp->Down->LN->QKV)...")
    lat_first, lat_rest = run_layer(
        args.model,
        test_model_dict[args.model],
        args.precision,
        pcb,
        args.device,
        M,
        seq_len_q,
        seq_len_kv,
        is_prefill,
        args.is_causal if is_prefill else False,
    )
    total = lat_first + (num_layers - 1) * lat_rest
    print("-" * 56)
    print(f"  Layer 1 (cold):  {lat_first * 1000:.4f} ms")
    print(f"  Layer 2+ (warm): {lat_rest * 1000:.4f} ms")
    print(f"  Total ({num_layers} layers): {total * 1000:.4f} ms")
    print("=" * 56)


if __name__ == "__main__":
    main()
