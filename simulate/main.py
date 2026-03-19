import argparse
from math import inf

from hardware_model.device import device_dict
from power_model.power_model import (
    calculate_flashattn_power,
    calculate_layernorm_power,
    calculate_matmul_power,
)
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


def _comm_tx_allgather_reducescatter(total_bytes: float, degree: int) -> float:
    """Ring: per-rank Tx for Allgather/ReduceScatter."""
    return (degree - 1) * total_bytes / degree


def _comm_tx_alltoall_transpose(per_rank_bytes: float, degree: int) -> float:
    """Ring: per-rank Tx for AlltoAll (transpose)."""
    return (degree - 1) * per_rank_bytes / degree


def _compute_comm_time(
    tx_bytes: float, bandwidth_gbps: float, latency_us: float
) -> float:
    """Comm time (s) = bytes/bandwidth + latency."""
    if bandwidth_gbps <= 0:
        return 0.0
    return tx_bytes / (bandwidth_gbps * 1e9) + latency_us * 1e-6


def _compute_non_overlapped_comm(
    comm_breakdown: dict[str, float],
    bandwidth_gbps: float,
    latency_us: float,
    lat_breakdown_ms: dict[str, float],
    overlap_config: dict[str, str],
) -> tuple[float, dict[str, float]]:
    """Non-overlapped comm latency (s) and per-collective breakdown (ms)."""
    if bandwidth_gbps <= 0:
        return 0.0, {}
    total_s = 0.0
    breakdown_ms = {}
    for name, tx_bytes in comm_breakdown.items():
        comm_time = _compute_comm_time(tx_bytes, bandwidth_gbps, latency_us)
        overlap_op = overlap_config.get(name)
        overlap_s = lat_breakdown_ms[overlap_op] / 1000 if overlap_op else 0.0
        non_overlapped_s = max(0.0, comm_time - overlap_s)
        total_s += non_overlapped_s
        if non_overlapped_s > 0:
            breakdown_ms[name] = non_overlapped_s * 1000
    return total_s, breakdown_ms


def print_layer_summary(
    lat_breakdown,
    power_breakdown,
    comm_breakdown,
    first_ops,
    rest_ops,
    comm_bandwidth_gbps=0.0,
    comm_latency_us=0.0,
):
    """Compute lat_first/lat_rest (incl. non-overlapped comm), print breakdown, return (lat_first, lat_rest, power_avg)."""
    lat_first = sum(lat_breakdown[op] for op in first_ops) / 1000
    lat_rest = sum(lat_breakdown[op] for op in rest_ops) / 1000

    comm_non_overlapped_ms = {}
    if comm_bandwidth_gbps > 0 and comm_breakdown:
        overlap_config = {
            "reducescatter_before_postattn_ln": "o_proj",
            "allgather_after_postattn_ln": "gate_up_proj",
            "reducescatter_before_postffn_ln": "down_proj",
            "allgather_after_postffn_ln": "qkv_proj_2",
            "alltoall_before_fa": "qkv_proj",
            "alltoall_after_fa": "o_proj",
        }
        non_overlapped_s, comm_non_overlapped_ms = _compute_non_overlapped_comm(
            comm_breakdown,
            comm_bandwidth_gbps,
            comm_latency_us,
            lat_breakdown,
            overlap_config,
        )
        lat_first += non_overlapped_s
        lat_rest += non_overlapped_s

    # Latency summary
    print(
        f"  --- lat_first: {lat_first * 1000:.4f} ms, lat_rest: {lat_rest * 1000:.4f} ms ---"
    )
    summary_ops = [
        ("qkv_proj", lat_breakdown["qkv_proj_2"]),  # warm path
        ("flash_attn", lat_breakdown["flash_attn"]),
        ("o_proj", lat_breakdown["o_proj"]),
        ("post_attn_ln", lat_breakdown["post_attn_ln"]),
        ("gate_up_proj", lat_breakdown["gate_up_proj"]),
        ("down_proj", lat_breakdown["down_proj"]),
        ("post_ffn_ln", lat_breakdown["post_ffn_ln"]),
    ]
    total_ms = sum(lat_breakdown[op] for op in rest_ops) + sum(
        comm_non_overlapped_ms.values()
    )
    print("  Operator latency summary (qkv_proj=warm):")
    print(f"  {'Operator':<35} {'Latency(ms)':>12} {'Share':>8}")
    print("  " + "-" * 57)
    for name, lat_ms in summary_ops:
        pct = (lat_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"  {name:<35} {lat_ms:>12.4f} {pct:>7.1f}%")
    for name, lat_ms in sorted(comm_non_overlapped_ms.items()):
        pct = (lat_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"  {name:<35} {lat_ms:>12.4f} {pct:>7.1f}%")

    # Power/Energy summary
    total_energy_mJ = sum(
        power_breakdown[name] * lat_breakdown[name] for name in rest_ops
    )
    total_runtime_s = lat_rest
    power_avg = total_energy_mJ / 1000 / total_runtime_s if total_runtime_s > 0 else 0
    print("  Power/Energy summary (mJ), avg power (W) = energy / runtime:")
    print(f"  {'Operator':<20} {'Power(W)':>10} {'Energy(mJ)':>12} {'Share':>8}")
    print("  " + "-" * 52)
    for name in power_breakdown:
        power_w = power_breakdown[name]
        energy_mJ = power_w * lat_breakdown[name]
        pct = (energy_mJ / total_energy_mJ * 100) if total_energy_mJ > 0 else 0
        print(f"  {name:<20} {power_w:>10.2f} {energy_mJ:>12.2f} {pct:>7.1f}%")
    print(f"  {'Total':<20} {power_avg:>10.2f} {total_energy_mJ:>12.2f}")

    if comm_breakdown:
        total_comm_tx = sum(comm_breakdown.values())
        print("  Communication (Ring, per-rank Tx bytes):")
        print(f"  {'Collective':<35} {'Tx(bytes)':>14} {'Share':>8}")
        print("  " + "-" * 60)
        for name, tx_bytes in sorted(comm_breakdown.items()):
            pct = (tx_bytes / total_comm_tx * 100) if total_comm_tx > 0 else 0
            print(f"  {name:<35} {tx_bytes:>14.0f} {pct:>7.1f}%")
        print(f"  {'Total per layer':<35} {total_comm_tx:>14.0f}")

    return lat_first, lat_rest, power_avg


def run_matmul(M, K, N, act_dt, wt_dt, mid_dt, out_dt, device, pcb, l2_prev=None):
    """Run [M,K]x[K,N] matmul. Falls back to roofline if M or K too small."""
    if K < 256:
        assert False, "too small"
    if device == "Orin":
        M = max(32, M) if wt_dt.name == "int4" else max(64, M)
    elif device == "Thor":
        M = max(64, M)
    else:
        assert False, "invalid device"
    mm = Matmul(act_dt, wt_dt, mid_dt, out_dt, device=device)
    mm(Tensor([M, K], dtype=act_dt), Tensor([K, N], dtype=wt_dt))
    lat = (
        mm.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
        + pcb.compute_module.launch_latency.matmul
    )
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
    parallelism="none",
    degree=1,
    comm_bandwidth_gbps=0.0,
    comm_latency_us=0.0,
):
    """Simulate in one pass: QKV -> FA -> O -> LN -> GateUp -> Down -> LN -> QKV.
    Returns (lat_first, lat_rest):
      lat_first = first QKV through post_ffn LN (layer 1)
      lat_rest = FA through second QKV (layer 2+)
    """
    K_shapes, N_shapes = get_matmul_shapes(model_name)
    d, nq, nkv = get_fa_shape(model_name)
    H = model_cfg["hidden_size"]
    M_ln = M // degree if (parallelism == "TP" and degree > 1) else M
    # Apply parallelism to shapes
    # CP: M/seq_len already divided in main()
    # TP: qkv_proj/up_proj -> N//=degree; o_proj/down_proj -> K//=degree
    # CP: matmul M already divided in main()
    if parallelism == "TP" and degree > 1:
        N_shapes = {
            op_name: (
                N_shapes[op_name] // degree
                if op_name in ("qkv_proj", "up_proj")
                else N_shapes[op_name]
            )
            for op_name in N_shapes
        }
        K_shapes = {
            op_name: (
                K_shapes[op_name] // degree
                if op_name not in ("qkv_proj", "up_proj")
                else K_shapes[op_name]
            )
            for op_name in K_shapes
        }
    # FlashAttn: both TP and CP: num_heads_q //= degree, num_heads_kv //= degree
    if parallelism != "none" and degree > 1:
        nq = nq // degree
        nkv = nkv // degree

    base_act = (
        data_type_dict["fp16"] if precision == "int4" else data_type_dict[precision]
    )
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
    assert ln_out_dt == qkv_act
    up_act, up_out = (
        ln_out_dt,
        get_matmul_output_dtype(ln_out_dt, "up_proj", is_test=False),
    )
    dn_act, dn_out = up_out, get_matmul_output_dtype(up_out, "down_proj", is_test=False)

    l2_prev = None
    lat_breakdown = {}
    power_breakdown = {}
    comm_breakdown = {}  # per-rank Tx bytes for each collective
    mode = "prefill" if is_prefill else "decode"
    fa_precision = "fp16" if precision in ("int4", "int8", "fp16") else "fp8"

    def record_op(op_name, lat_s, power_fn):
        lat_breakdown[op_name] = lat_s * 1000
        r = power_fn()
        power_breakdown[op_name] = r["power_breakdown_watts"]["total"]
        print(
            f"    {op_name:<18} {lat_s * 1000:>8.4f} ms {power_breakdown[op_name]:>8.2f} W"
        )

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
    l2_prev = mm.l2_status
    record_op(
        "qkv_proj",
        lat,
        lambda: calculate_matmul_power(
            device, precision, mm.mem_access_size, mm.fma_count, lat
        ),
    )
    # CP: AlltoAll before FlashAttn
    if parallelism == "CP" and degree > 1:
        # Q+K+V before FA
        comm_breakdown["alltoall_before_fa"] = _comm_tx_alltoall_transpose(
            seq_len_q * N_shapes["qkv_proj"] * qkv_dt.word_size, degree
        )
        assert N_shapes["qkv_proj"] == (nq * d + 2 * nkv * d) * degree
    else:
        assert N_shapes["qkv_proj"] == nq * d + 2 * nkv * d

    # 2. FlashAttn
    print(
        f"  [2/8] FlashAttn  [seq_q={seq_len_q}, seq_kv={seq_len_kv}, heads={nq}/{nkv}, causal={is_causal}]"
    )
    num_splits_list = [1] if is_prefill else [1, 2, 4]
    lat_fa_combine = inf
    best_fa = None
    best_l2_prev = None
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
            best_l2_prev = combine.l2_status if num_splits > 1 else fa.l2_status
            best_fa = fa
    l2_prev = best_l2_prev
    record_op(
        "flash_attn",
        lat_fa_combine,
        lambda: calculate_flashattn_power(
            device,
            fa_precision,
            best_fa.mem_access_size,
            best_fa.fma_count,
            lat_fa_combine,
        ),
    )

    # CP: AlltoAll after FlashAttn
    if parallelism == "CP" and degree > 1:
        # attn output after FA
        comm_breakdown["alltoall_after_fa"] = _comm_tx_alltoall_transpose(
            seq_len_q * K_shapes["o_proj"] * attn_out_dt.word_size, degree
        )
        assert degree * nq * d == K_shapes["o_proj"]
    else:
        assert nq * d == K_shapes["o_proj"]

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
    l2_prev = mm.l2_status
    record_op(
        "o_proj",
        lat,
        lambda: calculate_matmul_power(
            device, precision, mm.mem_access_size, mm.fma_count, lat
        ),
    )

    # TP: ReduceScatter before PostAttn_LayerNorm
    if parallelism == "TP" and degree > 1:
        comm_breakdown["reducescatter_before_postattn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * o_out.word_size, degree)
        )
    assert N_shapes["o_proj"] == H

    # 4. Post-attn LayerNorm
    print(f"  [4/8] PostAttn_LayerNorm  [M={M}, N={H}]")
    ln = FusedLayerNorm(o_out, up_act)
    ln(Tensor([M_ln, H], o_out), Tensor([M_ln, H], o_out))
    lat = (
        ln.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
        + pcb.compute_module.launch_latency.layernorm
    )
    l2_prev = ln.l2_status
    record_op(
        "post_attn_ln",
        lat,
        lambda: calculate_layernorm_power(device, mode, ln.mem_access_size, lat),
    )

    # TP: Allgather after PostAttn_LayerNorm
    if parallelism == "TP" and degree > 1:
        comm_breakdown["allgather_after_postattn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * up_act.word_size, degree)
        )
    assert H == K_shapes["up_proj"]

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
    l2_prev = mm.l2_status
    record_op(
        "gate_up_proj",
        lat,
        lambda: calculate_matmul_power(
            device, precision, mm.mem_access_size, mm.fma_count, lat
        ),
    )

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
    l2_prev = mm.l2_status
    record_op(
        "down_proj",
        lat,
        lambda: calculate_matmul_power(
            device, precision, mm.mem_access_size, mm.fma_count, lat
        ),
    )

    # TP: ReduceScatter before PostFFN_LayerNorm
    if parallelism == "TP" and degree > 1:
        comm_breakdown["reducescatter_before_postffn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * dn_out.word_size, degree)
        )

    # 7. Post-FFN LayerNorm
    print(f"  [7/8] PostFFN_LayerNorm  [M={M}, N={H}]")
    ln = FusedLayerNorm(dn_out, ln_out_dt)
    ln(Tensor([M_ln, H], dn_out), Tensor([M_ln, H], dn_out))
    lat = (
        ln.compile_and_simulate(pcb, L2Cache_previous=l2_prev)
        + pcb.compute_module.launch_latency.layernorm
    )
    l2_prev = ln.l2_status
    record_op(
        "post_ffn_ln",
        lat,
        lambda: calculate_layernorm_power(device, mode, ln.mem_access_size, lat),
    )

    # TP: Allgather after PostFFN_LayerNorm
    if parallelism == "TP" and degree > 1:
        comm_breakdown["allgather_after_postffn_ln"] = _comm_tx_allgather_reducescatter(
            M * H * ln_out_dt.word_size, degree
        )

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
    record_op(
        "qkv_proj_2",
        lat,
        lambda: calculate_matmul_power(
            device, precision, mm.mem_access_size, mm.fma_count, lat
        ),
    )

    first_ops = [
        "qkv_proj",
        "flash_attn",
        "o_proj",
        "post_attn_ln",
        "gate_up_proj",
        "down_proj",
        "post_ffn_ln",
    ]
    rest_ops = [
        "flash_attn",
        "o_proj",
        "post_attn_ln",
        "gate_up_proj",
        "down_proj",
        "post_ffn_ln",
        "qkv_proj_2",
    ]
    lat_first, lat_rest, power_avg = print_layer_summary(
        lat_breakdown,
        power_breakdown,
        comm_breakdown,
        first_ops,
        rest_ops,
        comm_bandwidth_gbps=comm_bandwidth_gbps,
        comm_latency_us=comm_latency_us,
    )
    return lat_first, lat_rest, power_avg


def resolve_dims(args, is_prefill):
    if is_prefill:
        return args.seq_len, args.seq_len, args.seq_len
    return args.spec_tokens, args.spec_tokens, args.seq_len


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
    parser.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--spec_tokens", type=int, default=64, choices=[64, 128])
    parser.add_argument(
        "--parallelism",
        default="none",
        choices=["TP", "CP", "none"],
        help="Parallelism type: TP (Tensor), CP (Context), or none",
    )
    parser.add_argument(
        "--parallelism_degree",
        type=int,
        default=2,
        choices=[2, 4],
        help="Parallelism degree (2 or 4), only used when --parallelism is not none",
    )
    parser.add_argument(
        "--comm_bandwidth",
        type=float,
        default=0.0,
        help="Comm bandwidth (GB/s), >0 to add non-overlapped comm latency",
    )
    parser.add_argument(
        "--comm_latency",
        type=float,
        default=0.0,
        help="Comm latency (us) per collective call",
    )
    args = parser.parse_args()

    is_prefill = args.phase == "prefill"
    if not is_prefill and args.is_causal:
        parser.error("decode does not support --is_causal")
    if args.seq_len < 1:
        parser.error("seq_len must be >= 1")
    if not is_prefill:
        min_m = 64 if args.device == "Thor" else 32
        if args.spec_tokens < min_m:
            parser.error(f"decode: spec_tokens must be >= {min_m}")
    if args.device not in device_dict:
        parser.error(f"Unknown device {args.device}")
    if args.device == "Orin" and args.precision not in ("fp16", "int8", "int4"):
        parser.error("Orin supports fp16/int8/int4 only")
    if args.device == "Thor" and args.precision not in ("fp8", "fp4"):
        parser.error("Thor supports fp8/fp4 only")

    degree = 1 if args.parallelism == "none" else args.parallelism_degree

    pcb = device_dict[args.device]

    M, seq_len_q, seq_len_kv = resolve_dims(args, is_prefill)
    if args.parallelism == "CP" and degree > 1:
        M = M // degree
        # seq_len_q, seq_len_kv 不变，仅 FlashAttn 的 num_heads 缩减
    print("=" * 56)
    print("  LLM Inference Latency Simulator")
    print("=" * 56)
    print(f"  device:      {args.device}")
    print(f"  model:       {args.model}")
    print(f"  precision:   {args.precision}")
    print(f"  phase:       {args.phase}")
    print(f"  seq_len:     {args.seq_len}")
    if not is_prefill:
        print(f"  spec_tokens: {args.spec_tokens}")
    model_cfg = test_model_dict[args.model]
    num_layers = model_cfg["num_layers"]
    print(f"  num_layers:  {num_layers}")
    print(f"  parallelism: {args.parallelism} (degree={degree})")
    print(f"  M:           {M}  (matmul batch dim)")
    print(f"  seq_len_q:   {seq_len_q}")
    print(f"  seq_len_kv:  {seq_len_kv}")
    print("-" * 56)
    print("  Simulating (QKV->FA->O->LN->GateUp->Down->LN->QKV)...")
    lat_first, lat_rest, power_avg = run_layer(
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
        parallelism=args.parallelism,
        degree=degree,
        comm_bandwidth_gbps=args.comm_bandwidth,
        comm_latency_us=args.comm_latency,
    )
    total = lat_first + (num_layers - 1) * lat_rest
    print("-" * 56)
    print(f"  Layer 1 (cold):  {lat_first * 1000:.4f} ms")
    print(f"  Layer 2+ (warm): {lat_rest * 1000:.4f} ms")
    print(f"  Total ({num_layers} layers): {total * 1000:.4f} ms")
    if power_avg is not None:
        print(f"  Est. avg power:   {power_avg:.2f} W")
    print("=" * 56)


if __name__ == "__main__":
    main()
