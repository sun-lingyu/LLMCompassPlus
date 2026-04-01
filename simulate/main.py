import argparse
import json
import os
from math import inf
from typing import Any, Dict, Tuple

from filelock import FileLock

from hardware_model.device import device_dict
from icnt_model.icnt_model import (
    calculate_pcie_dynamic_energy,
    calculate_ucie_dynamic_energy,
    get_nearest_pcie_lane,
    get_nearest_ucie_configurations,
    load_json_data,
)
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

file_dir = os.path.dirname(os.path.abspath(__file__))

LAYER_COMPUTE_CACHE_PATH = os.path.join(file_dir, "layer_compute_cache.json")

LAYER_CACHE_META_KEYS = (
    "device",
    "model",
    "precision",
    "phase",
    "spec_tokens",
    "seq_len",
    "is_causal",
    "parallelism",
    "degree",
)

LAYER_COMPUTE_OP_NAMES = (
    "qkv_proj",
    "flash_attn",
    "o_proj",
    "post_attn_ln",
    "gate_up_proj",
    "down_proj",
    "post_ffn_ln",
    "qkv_proj_2",
)


def layer_cache_key_tuple_from_record(r: dict) -> tuple:
    return (
        r["device"],
        r["model"],
        r["precision"],
        r["phase"],
        int(r["spec_tokens"]),
        int(r["seq_len"]),
        bool(r["is_causal"]),
        r["parallelism"],
        int(r["degree"]),
    )


def load_layer_compute_cache(
    path: str,
) -> Dict[Tuple[Any, ...], Dict[str, dict]]:
    """Load JSON list of layer records; return (records, key -> {op_name: {lat, dram_access, power}})."""
    assert os.path.exists(path)
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return [], {}
    assert isinstance(data, list)
    cache_dict: Dict[Tuple[Any, ...], Dict[str, dict]] = {}
    for r in data:
        assert isinstance(r, dict)
        kt = layer_cache_key_tuple_from_record(r)
        ops = {}
        for op in LAYER_COMPUTE_OP_NAMES:
            assert op in r
            entry = r[op]
            ops[op] = {
                "lat_ms": float(entry["lat_ms"]),
                "dram_access_byte": float(entry["dram_access_byte"]),
                "power_w": float(entry["power_w"]),
            }
        assert kt not in cache_dict, f"Duplicate key {kt}"  # no duplicate keys
        cache_dict[kt] = ops
    return cache_dict


def update_layer_cache_record(flat_row: dict, path: str) -> None:
    """Merge flat_row (meta + per-op metrics) into records by cache key and write JSON."""
    new_r = {mk: flat_row[mk] for mk in LAYER_CACHE_META_KEYS}
    for op in LAYER_COMPUTE_OP_NAMES:
        if op in flat_row:
            new_r[op] = flat_row[op]
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
        data.append(new_r)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def _launch_lat_ms_for_compute_op(op_name: str, pcb, is_prefill: bool) -> float:
    if op_name in (
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "qkv_proj_2",
    ):
        return pcb.compute_module.launch_latency.matmul * 1000
    if op_name == "flash_attn":
        return (
            pcb.compute_module.launch_latency.flashattn_prefill
            if is_prefill
            else pcb.compute_module.launch_latency.flashattn_decode
        ) * 1000
    if op_name in ("post_attn_ln", "post_ffn_ln"):
        return pcb.compute_module.launch_latency.layernorm * 1000
    assert False, f"unknown op_name {op_name}"


def _fill_comm_breakdown_for_layer(
    parallelism: str,
    degree: int,
    M: int,
    N_shapes: dict,
    K_shapes: dict,
    qkv_dt,
    nq: int,
    nkv: int,
    d: int,
    seq_len_q: int,
    attn_out_dt,
    H: int,
    o_out,
    up_act,
    dn_out,
    ln_out_dt,
) -> dict[str, float]:
    """Fill comm_breakdown from shapes only (same logic as interleaved in run_layer)."""
    comm_breakdown: dict[str, float] = {}
    if parallelism == "CP" and degree > 1:
        comm_breakdown["alltoall_before_fa"] = _comm_tx_alltoall_transpose(
            M * N_shapes["qkv_proj"] * qkv_dt.word_size, degree
        )
        assert N_shapes["qkv_proj"] == (nq * d + 2 * nkv * d) * degree
    else:
        assert N_shapes["qkv_proj"] == nq * d + 2 * nkv * d

    if parallelism == "CP" and degree > 1:
        comm_breakdown["alltoall_after_fa"] = _comm_tx_alltoall_transpose(
            seq_len_q * nq * d * attn_out_dt.word_size, degree
        )
        assert degree * nq * d == K_shapes["o_proj"]
    else:
        assert nq * d == K_shapes["o_proj"]

    if parallelism == "TP" and degree > 1:
        comm_breakdown["reducescatter_before_postattn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * o_out.word_size, degree)
        )
    assert N_shapes["o_proj"] == H

    if parallelism == "TP" and degree > 1:
        comm_breakdown["allgather_after_postattn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * up_act.word_size, degree)
        )
    assert H == K_shapes["up_proj"]

    if parallelism == "TP" and degree > 1:
        comm_breakdown["reducescatter_before_postffn_ln"] = (
            _comm_tx_allgather_reducescatter(M * H * dn_out.word_size, degree)
        )

    if parallelism == "TP" and degree > 1:
        comm_breakdown["allgather_after_postffn_ln"] = _comm_tx_allgather_reducescatter(
            M * H * ln_out_dt.word_size, degree
        )

    return comm_breakdown


def _comm_tx_allgather_reducescatter(total_bytes: float, degree: int) -> float:
    """Ring: per-rank Tx for Allgather/ReduceScatter."""
    return (degree - 1) * total_bytes / degree


def _comm_tx_alltoall_transpose(per_rank_bytes: float, degree: int) -> float:
    """Ring: per-rank Tx for AlltoAll (transpose)."""
    if degree == 2:
        # unidirection ring
        # Calculation:
        # Suppose there are p processors and each processor holds p messages of size m.
        # Each processor sends (p-1) round of messages.
        # The message size of each round is m(p-1), m(p-2), ..., m.
        # Therefore, the total TX of each processor is mp(p-1)/2.
        # In our notation, degree = p, per_rank_bytes = mp.
        # See https://en.wikipedia.org/wiki/All-to-all_%28parallel_pattern%29
        return (degree - 1) * per_rank_bytes / 2
        # If use bidirection ring for degree = 2 case, total TX = degree * per_rank_bytes / 4 = per_rank_bytes / 2.
        # So the code is also correct for bidirection ring.
    elif degree == 4:
        # bidirection ring
        # Calculation:
        # For bidirection ring, Each processor sends messages to both left and right neighbors.
        # When p is even:
        # The message size for left neighbors is mp/2, m(p/2-1), ..., m.
        # The message size for right neighbors is m(p/2-1), m(p/2-2), ..., m.
        # Therefore, the total TX of each processor is mp^2/4
        return degree * per_rank_bytes / 4


def _compute_non_overlapped_comm(
    comm_breakdown: dict[str, float],
    bandwidth_gbps: float,
    latency_ns: float,
    lat_breakdown_ms: dict[str, float],
    dram_access_breakdown_bytes: dict[str, float],
    launch_lat_breakdown_ms: dict[str, float],
    overlap_config: dict[str, str],
    io_module,
    no_contention: bool,
    comm_bandwidth_efficiency: float = 0.9,
) -> tuple[float, float, dict[str, float]]:
    """Non-overlapped comm latency (s) and per-collective breakdown (ms).

    Models: (1) comm does not start until after the overlap op's launch latency;
    (2) during effective compute (total op time minus launch), DRAM contention:
    comm uses min(link Gbps, DRAM leftover after compute's average DRAM BW);
    (3) after the overlap op finishes, comm uses full link bandwidth_gbps.
    """
    assert bandwidth_gbps > 0
    eff_bandwidth_gbps = bandwidth_gbps * comm_bandwidth_efficiency
    dram_peak_gbps = io_module.bandwidth * io_module.bandwidth_efficiency * 8 / 1e9
    t_lat_s = latency_ns * 1e-9
    total_comm_tx_time_ms = 0.0
    total_non_overlapped_ms = 0.0
    breakdown_ms = {}

    for name, tx_bytes in comm_breakdown.items():
        overlap_op = overlap_config[name]
        T_comp_s = lat_breakdown_ms[overlap_op] / 1000.0
        Launch_s = launch_lat_breakdown_ms[overlap_op] / 1000.0
        assert Launch_s >= 0 and T_comp_s > Launch_s
        T_comp_eff_s = T_comp_s - Launch_s
        dram_access_size_byte = float(dram_access_breakdown_bytes[overlap_op])
        assert dram_access_size_byte > 0

        overlap_mem_bw_gbps = dram_access_size_byte / T_comp_eff_s * 8 / 1e9
        assert dram_peak_gbps > overlap_mem_bw_gbps, (
            f"{overlap_op}| T_comp_s: {T_comp_s}, Launch_s: {Launch_s} dram_peak_gbps: {dram_peak_gbps}, overlap_mem_bw_gbps: {overlap_mem_bw_gbps}"
        )

        if no_contention:
            bw_ov_gbps = eff_bandwidth_gbps
        else:
            bw_ov_gbps = min(
                eff_bandwidth_gbps,
                (dram_peak_gbps - overlap_mem_bw_gbps)
                / 2,  # Remaining BW is used for Tx + Rx, so divide by 2
            )
            if dram_peak_gbps - overlap_mem_bw_gbps < eff_bandwidth_gbps:
                print(
                    f"WARNING: Comm & Comp overlap: Mem BW contention dominates! Effective BW is {bw_ov_gbps / 8:.2f} GB/s"
                )

        tx_giga_bytes = tx_bytes / 1e9
        cap_giga_bytes = bw_ov_gbps * T_comp_eff_s / 8
        if tx_giga_bytes <= cap_giga_bytes:
            T_comm_eff_s = tx_giga_bytes * 8 / bw_ov_gbps
        else:
            rem_giga_bytes = tx_giga_bytes - cap_giga_bytes
            T_comm_eff_s = T_comp_eff_s + rem_giga_bytes * 8 / eff_bandwidth_gbps
        T_comm_s = t_lat_s + T_comm_eff_s
        non_overlapped_ms = max(0, T_comm_s - T_comp_eff_s) * 1000
        if non_overlapped_ms > 0:
            breakdown_ms[name] = non_overlapped_ms

        total_comm_tx_time_ms += T_comm_s * 1000
        total_non_overlapped_ms += non_overlapped_ms
    return total_comm_tx_time_ms, total_non_overlapped_ms, breakdown_ms


def print_layer_summary_comp(
    lat_breakdown,
    power_breakdown,
    op_names,
):
    """Compute layer latency (compute only), print breakdown, return (layer_lat_ms, power_avg)."""
    layer_lat_ms = sum(lat_breakdown[op] for op in op_names)

    # Latency summary
    summary_ops = [
        ("qkv_proj", lat_breakdown["qkv_proj_2"]),  # warm path
        ("flash_attn", lat_breakdown["flash_attn"]),
        ("o_proj", lat_breakdown["o_proj"]),
        ("post_attn_ln", lat_breakdown["post_attn_ln"]),
        ("gate_up_proj", lat_breakdown["gate_up_proj"]),
        ("down_proj", lat_breakdown["down_proj"]),
        ("post_ffn_ln", lat_breakdown["post_ffn_ln"]),
    ]
    print("  " + "=" * 57)
    print("  Operator latency summary:")
    print(f"  {'Operator':<20} {'Latency(ms)':>12} {'Share':>8}")
    print("  " + "-" * 55)
    for name, lat_ms in summary_ops:
        pct = (lat_ms / layer_lat_ms * 100) if layer_lat_ms > 0 else 0
        print(f"  {name:<20} {lat_ms:>12.4f} {pct:>7.1f}%")
    print(f"  {'Total':<20} {layer_lat_ms:>12.4f}")

    # Power/Energy summary
    total_energy_mJ = sum(
        power_breakdown[name] * lat_breakdown[name] for name in op_names
    )
    power_avg = total_energy_mJ / layer_lat_ms
    print("  " + "=" * 57)
    print("  Operator Power/Energy summary:")
    print(f"  {'Operator':<20} {'Power(W)':>10} {'Energy(mJ)':>12} {'Share':>8}")
    print("  " + "-" * 55)
    for name in power_breakdown:
        power_w = power_breakdown[name]
        energy_mJ = power_w * lat_breakdown[name]
        pct = (energy_mJ / total_energy_mJ * 100) if total_energy_mJ > 0 else 0
        print(f"  {name:<20} {power_w:>10.2f} {energy_mJ:>12.2f} {pct:>7.1f}%")
    print(f"  {'Avg/Total':<20} {power_avg:>10.2f} {total_energy_mJ:>12.2f}")

    return layer_lat_ms, power_avg


def print_layer_summary_comm(
    lat_breakdown,
    comm_breakdown,
    comm_bandwidth_gbps,
    device,
    dram_access_breakdown_bytes,
    launch_lat_breakdown_ms,
    pcb,
    icnt_type,
    no_contention,
    ucie_spec_path=f"{file_dir}/../icnt_model/configs/UCIE.json",
    pcie_spec_path=f"{file_dir}/../icnt_model/configs/PCIE.json",
):
    if icnt_type.startswith("ucie"):
        ucie_spec = load_json_data(ucie_spec_path)
        comm_latency_ns = ucie_spec["latency"]
    elif icnt_type == "pcie":  # pcie
        pcie_spec = load_json_data(pcie_spec_path)
        comm_latency_ns = pcie_spec["latency"]
    else:
        assert False

    total_comm_tx_bytes = sum(comm_breakdown.values())
    print("  " + "=" * 57)
    print("  Communication size summary:")
    print(f"  {'Operator':<35} {'Size(MB)':>12} {'Share':>8}")
    for name, size_byte in sorted(comm_breakdown.items()):
        pct = size_byte / total_comm_tx_bytes * 100
        print(f"  *{name:<35} {size_byte / 1024**2:>12.4f} {pct:>7.1f}%")
    print(f"  {'Total':<35} {total_comm_tx_bytes / 1024**2:>9.4f} MB")

    overlap_config = {
        "reducescatter_before_postattn_ln": "o_proj",
        "allgather_after_postattn_ln": "gate_up_proj",
        "reducescatter_before_postffn_ln": "down_proj",
        "allgather_after_postffn_ln": "qkv_proj_2",
        "alltoall_before_fa": "qkv_proj",
        "alltoall_after_fa": "o_proj",
    }
    total_comm_tx_time_ms, non_overlapped_ms, comm_non_overlapped_ms = (
        _compute_non_overlapped_comm(
            comm_breakdown,
            comm_bandwidth_gbps,
            comm_latency_ns,
            lat_breakdown,
            dram_access_breakdown_bytes,
            launch_lat_breakdown_ms,
            overlap_config,
            pcb.io_module,
            no_contention,
        )
    )

    print("  " + "=" * 57)
    print(f"  {icnt_type.upper()} Communication latency summary:")
    print(
        f"  {'Operator':<35} {'Latency(ms)':>12} {'Non-ov(ms)':>10} {'Non-ov Share':>15}"
    )
    for name, tx_bytes in comm_breakdown.items():
        lat_ideal_ms = (
            comm_latency_ns / 1e6 + tx_bytes * 8 / 1024**3 / comm_bandwidth_gbps * 1000
        )
        lat_non_overlapped_ms = comm_non_overlapped_ms.get(name, 0)
        pct_non_overlapped = (
            lat_non_overlapped_ms / non_overlapped_ms * 100
            if non_overlapped_ms > 0
            else 0
        )
        print(
            f"  {name:<35} {lat_ideal_ms:>12.4f} {lat_non_overlapped_ms:>10.4f} {pct_non_overlapped:>14.1f}%"
        )
    print(f"  {'Total Non-overlapped Latency':<35} {non_overlapped_ms:>9.4f} ms")

    print("  " + "=" * 57)
    power_results = {}
    voltage = 0.7 if device == "Orin" else 0.5
    if icnt_type.startswith("ucie"):
        ucie_nearest_configs = get_nearest_ucie_configurations(
            comm_bandwidth_gbps, ucie_spec
        )
        if icnt_type.endswith("std"):
            pkg = "standard"
        elif icnt_type.endswith("adv"):
            pkg = "advanced"
        else:
            assert False
        candidates = ucie_nearest_configs[pkg]
        for candidate in candidates:
            module_count, lane_count, rate_gt = (
                candidate["module_count"],
                candidate["lane_count"],
                candidate["rate_gt"],
            )
            capacity_gbps = module_count * lane_count * rate_gt
            energy_mj = calculate_ucie_dynamic_energy(
                pkg,
                rate_gt,
                f"{voltage}V",
                total_comm_tx_bytes,
                ucie_spec,
            )
            power_avg = energy_mj / total_comm_tx_time_ms
            print(
                f"  {pkg:<10} {module_count} × {lane_count} lane(s) x {rate_gt:>2} GT/s = {capacity_gbps:.1f} Gbps: {power_avg:>5.2f} W {energy_mj:>5.2f} mJ"
            )
            power_results[(pkg, module_count, lane_count, rate_gt)] = energy_mj
    elif icnt_type == "pcie":
        pcie_lanes = get_nearest_pcie_lane(comm_bandwidth_gbps, pcie_spec)
        capacity_gbps = pcie_lanes * pcie_spec["rate_gt"]
        energy_mj = calculate_pcie_dynamic_energy(total_comm_tx_bytes, pcie_spec)
        power_avg = energy_mj / total_comm_tx_time_ms
        print(
            f"  {pcie_lanes} x {pcie_spec['rate_gt']} GT/s = {capacity_gbps:.1f} Gbps: {power_avg:>5.2f} W {energy_mj:>5.2f} mJ"
        )
        # pcie_switch_dynamic_power = pcie_spec["switch_power_k"] * pcie_lanes
        # print(f"  Additional PCIe switch power: {pcie_switch_dynamic_power:.2f} W")
        # energy_mj += pcie_switch_dynamic_power * total_comm_tx_time_ms
        power_results[pcie_lanes] = energy_mj
    else:
        assert False

    return non_overlapped_ms, power_results


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
    parallelism,
    degree=1,
    comm_bandwidth_gbps=0.0,
    icnt_type="pcie",
    no_contention=False,
    spec_tokens=64,
    seq_len=512,
    layer_cache_dict=None,
    cache_path=None,
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
    if degree > 1:
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

    comm_breakdown = _fill_comm_breakdown_for_layer(
        parallelism,
        degree,
        M,
        N_shapes,
        K_shapes,
        qkv_dt,
        nq,
        nkv,
        d,
        seq_len_q,
        attn_out_dt,
        H,
        o_out,
        up_act,
        dn_out,
        ln_out_dt,
    )

    l2_prev = None
    lat_breakdown = {}
    power_breakdown = {}
    dram_access_breakdown_bytes = {}
    launch_lat_breakdown_ms = {}
    mode = "prefill" if is_prefill else "decode"
    fa_precision = "fp16" if precision in ("int4", "int8", "fp16") else "fp8"
    phase = mode
    cache_key = (
        device,
        model_name,
        precision,
        phase,
        int(spec_tokens),
        int(seq_len),
        bool(is_causal),
        parallelism,
        int(degree),
    )
    layer_cache_row = {
        "device": device,
        "model": model_name,
        "precision": precision,
        "phase": phase,
        "spec_tokens": int(spec_tokens),
        "seq_len": int(seq_len),
        "is_causal": bool(is_causal),
        "parallelism": parallelism,
        "degree": int(degree),
    }

    use_cached_layer = False
    if cache_path and layer_cache_dict is not None and layer_cache_dict is not None:
        ops_cached = layer_cache_dict.get(cache_key)
        if ops_cached:
            use_cached_layer = True
            for op in LAYER_COMPUTE_OP_NAMES:
                e = ops_cached[op]
                lat_breakdown[op] = float(e["lat_ms"])
                dram_access_breakdown_bytes[op] = float(e["dram_access_byte"])
                power_breakdown[op] = float(e["power_w"])
                launch_lat_breakdown_ms[op] = _launch_lat_ms_for_compute_op(
                    op, pcb, is_prefill
                )
            print("  (compute from layer_compute_cache.json)")
            for op in LAYER_COMPUTE_OP_NAMES:
                print(
                    f"    {op:<18} {lat_breakdown[op]:>8.4f} ms {power_breakdown[op]:>8.2f} W  [cache]"
                )

    def record_op(
        op_name,
        lat_s,
        power_fn,
        *,
        mem_access_bytes: float,
        launch_latency_s: float,
    ):
        lat_breakdown[op_name] = lat_s * 1000
        dram_access_breakdown_bytes[op_name] = float(mem_access_bytes)
        launch_lat_breakdown_ms[op_name] = launch_latency_s * 1000
        r = power_fn()
        power_breakdown[op_name] = r["power_breakdown_watts"]["total"]
        print(
            f"    {op_name:<18} {lat_s * 1000:>8.4f} ms {power_breakdown[op_name]:>8.2f} W"
        )
        if cache_path and layer_cache_dict is not None:
            layer_cache_row[op_name] = {
                "lat_ms": lat_breakdown[op_name],
                "dram_access_byte": dram_access_breakdown_bytes[op_name],
                "power_w": power_breakdown[op_name],
            }

    if not use_cached_layer:
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
            mem_access_bytes=mm.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.matmul,
        )
        # 2. FlashAttn
        print(
            f"  [2/8] FlashAttn  [seq_q={seq_len_q}, seq_kv={seq_len_kv}, heads={nq}/{nkv}, causal={is_causal}]"
        )
        num_splits_list = [1] if is_prefill else [1, 2, 4]
        lat_fa_combine = inf
        best_fa = None
        best_l2_prev = None
        best_fa_dram_bytes = 0.0
        best_fa_launch_s = 0.0
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
            launch_fa = (
                pcb.compute_module.launch_latency.flashattn_prefill
                if is_prefill
                else pcb.compute_module.launch_latency.flashattn_decode
            )
            lat_fa_combine_temp = (
                fa.compile_and_simulate(pcb, L2Cache_previous=l2_prev) + launch_fa
            )
            dram_bytes_this = fa.mem_access_size
            launch_s_this = launch_fa

            if num_splits > 1:
                combine = FlashAttnCombine(fp32, attn_out_dt)
                combine(Tensor([seq_len_q, nq * d, num_splits], fp32))
                lat_fa_combine_temp += (
                    combine.compile_and_simulate(pcb, L2Cache_previous=fa.l2_status)
                    + pcb.compute_module.launch_latency.flashattn_combine
                )
                dram_bytes_this += combine.mem_access_size
                launch_s_this += pcb.compute_module.launch_latency.flashattn_combine
            if lat_fa_combine_temp < lat_fa_combine:
                lat_fa_combine = lat_fa_combine_temp
                best_l2_prev = combine.l2_status if num_splits > 1 else fa.l2_status
                best_fa = fa
                best_fa_dram_bytes = dram_bytes_this
                best_fa_launch_s = launch_s_this
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
            mem_access_bytes=best_fa_dram_bytes,
            launch_latency_s=best_fa_launch_s,
        )

        # 3. O proj
        print(
            f"  [3/8] O_proj  [M={M}, K={K_shapes['o_proj']}, N={N_shapes['o_proj']}]"
        )
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
            mem_access_bytes=mm.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.matmul,
        )

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
            mem_access_bytes=ln.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.layernorm,
        )

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
            mem_access_bytes=mm.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.matmul,
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
            mem_access_bytes=mm.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.matmul,
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
            mem_access_bytes=ln.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.layernorm,
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
            mem_access_bytes=mm.mem_access_size,
            launch_latency_s=pcb.compute_module.launch_latency.matmul,
        )

        if cache_path and layer_cache_dict is not None:
            update_layer_cache_record(layer_cache_row, cache_path)
        if layer_cache_dict is not None:
            layer_cache_dict[cache_key] = {
                op: layer_cache_row[op]
                for op in LAYER_COMPUTE_OP_NAMES
                if op in layer_cache_row
            }

    print()
    print("  Layer summary:")
    op_names = [
        "flash_attn",
        "o_proj",
        "post_attn_ln",
        "gate_up_proj",
        "down_proj",
        "post_ffn_ln",
        "qkv_proj_2",
    ]
    results = {}
    lat, power_avg = print_layer_summary_comp(
        lat_breakdown,
        power_breakdown,
        op_names,
    )
    results["lat_comp"] = lat
    results["power_avg_comp"] = power_avg
    if degree > 1:
        lat_comm, power_results_comm = print_layer_summary_comm(
            lat_breakdown,
            comm_breakdown,
            comm_bandwidth_gbps,
            device,
            dram_access_breakdown_bytes,
            launch_lat_breakdown_ms,
            pcb,
            icnt_type,
            no_contention,
        )
        results["lat_comm"] = lat_comm
        results["power_results_comm"] = power_results_comm
    print("  " + "=" * 57)
    return results


def print_final_results(num_layers, results, icnt_type, has_comm_results):
    lat_comp = results["lat_comp"]
    power_avg_comp = results["power_avg_comp"]
    if not has_comm_results:
        lat_total = num_layers * lat_comp
        print(f"  Layer: {lat_comp:.4f} ms")
        print(f"  Total ({num_layers} layers): {lat_total:.4f} ms")
        print(f"  Est. avg power:   {power_avg_comp:.2f} W")
        print("=" * 57)
        return

    lat_comm = results["lat_comm"]
    power_results_comm = results["power_results_comm"]
    print()
    print("  Overall summary:")
    print("  " + "+" * 57)
    print(f"  Layer [Compute]: {lat_comp:.4f} ms")
    print(f"  Total [Compute] ({num_layers} layers): {num_layers * lat_comp:.4f} ms")
    print(f"  Layer [Compute + Communication]: {lat_comp + lat_comm:.4f} ms")
    print(
        f"  Total [Compute + Communication] ({num_layers} layers): {num_layers * (lat_comp + lat_comm):.4f} ms"
    )

    # Comm results
    print("  " + "+" * 57)
    print(f"  Est. avg power [Compute]:   {power_avg_comp:.2f} W")
    print("  Est. avg power [Communication]:")
    if icnt_type.startswith("ucie"):
        for (
            pkg,
            module_count,
            lane_count,
            rate_gt,
            energy_mj_comm,
        ) in power_results_comm.items():
            capacity_gbps = module_count * lane_count * rate_gt
            power_avg = energy_mj_comm / lat_comm
            print(
                f"  {pkg:<10} {module_count} × {lane_count} lane(s) x {rate_gt:>2} GT/s = {capacity_gbps:.1f} Gbps : {power_avg:.2f} W"
            )
    elif icnt_type == "pcie":
        pcie_spec = load_json_data(f"{file_dir}/../icnt_model/configs/PCIE.json")
        # pcie_switch_static_power = pcie_spec["switch_power_intercept"]
        assert len(power_results_comm) == 1
        for pcie_lanes, energy_mj_comm in power_results_comm.items():
            capacity_gbps = pcie_lanes * pcie_spec["rate_gt"]
            power_avg = (power_avg_comp * lat_comp + energy_mj_comm) / (
                lat_comp + lat_comm
            )
            print(
                f"    {pcie_lanes} lane(s) {capacity_gbps:.1f} Gbps: {power_avg:.2f} W"
            )
            # print(f"  Additional PCIe switch power: {pcie_switch_static_power:.2f} W")
    else:
        assert False
    print("  " + "+" * 57)


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
        default="TP",
        choices=["TP", "CP"],
        help="Parallelism type: TP (Tensor), CP (Context)",
    )
    parser.add_argument(
        "--parallelism_degree",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Parallelism degree (1, 2 or 4)",
    )
    parser.add_argument(
        "--comm_bandwidth",
        type=float,
        default=0.0,
        help="Comm bandwidth (GB/s), >0 to add non-overlapped comm latency",
    )
    parser.add_argument(
        "--icnt_type",
        type=str,
        default="pcie",
        choices=["pcie", "ucie_std", "ucie_adv"],
        help="ICNT type: pcie (PCIe), ucie_std (UCIE standard), ucie_adv (UCIE advanced)",
    )
    parser.add_argument(
        "--no_contention",
        action="store_true",
        help="Do not factor in comm&comp mem bw contention",
    )
    args = parser.parse_args()

    is_prefill = args.phase == "prefill"
    if args.model == "InternVision" and args.is_causal:
        parser.error("InternVision should not use --is_causal")
        # This check can be removed if necessary
    if args.model != "InternVision" and not args.is_causal:
        parser.error("Only InternVision should be non-causal")
        # This check can be removed if necessary
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

    degree = args.parallelism_degree
    if degree > 1:
        assert args.comm_bandwidth > 0
    else:
        assert args.comm_bandwidth == 0

    pcb = device_dict[args.device]

    layer_cache_dict = load_layer_compute_cache(LAYER_COMPUTE_CACHE_PATH)

    seq_len_kv = args.seq_len
    M = seq_len_q = args.seq_len if is_prefill else args.spec_tokens

    if args.parallelism == "CP" and degree > 1:
        M = M // degree
    print("=" * 57)
    print("  LLM Inference Latency Simulator")
    print("=" * 57)
    print(f"  device:      {args.device}")
    print(f"  model:       {args.model}")
    print(f"  precision:   {args.precision}")
    print(f"  para degree: {degree}")
    if degree > 1:
        print(f"  parallelism: {args.parallelism}")
        print(f"  comm_bw:     {args.comm_bandwidth} GB/s")
        print(f"  icnt_type:   {args.icnt_type}")
        print(f"  no_contention: {args.no_contention}")
    print(f"  phase:       {args.phase}")
    print(f"  causal:      {args.is_causal}")
    print(f"  seq_len:     {args.seq_len}")
    if not is_prefill:
        print(f"  spec_tokens: {args.spec_tokens}")
    model_cfg = test_model_dict[args.model]
    num_layers = model_cfg["num_layers"]
    print(f"  num_layers:  {num_layers}")
    print(f"  M:           {M}")
    print(f"  seq_len_q:   {seq_len_q}")
    print(f"  seq_len_kv:  {seq_len_kv}")
    print("-" * 57)
    print("  Simulating (QKV->FA->O->LN->GateUp->Down->LN->QKV)...")
    if degree > 1:
        if args.icnt_type in ("pcie", "ucie_std"):
            effective_comm_bandwidth = args.comm_bandwidth * 2
            # always bidirection ring with two ports
        elif args.icnt_type == "ucie_adv":
            effective_comm_bandwidth = (
                args.comm_bandwidth * 2 if degree == 4 else args.comm_bandwidth
            )
            # unidirection ring when degree = 2, bidirection ring when degree = 4
        else:
            assert False
    else:
        effective_comm_bandwidth = args.comm_bandwidth
    results = run_layer(
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
        comm_bandwidth_gbps=effective_comm_bandwidth * 8,  # GB/s to Gbps
        icnt_type=args.icnt_type,
        no_contention=args.no_contention,
        spec_tokens=args.spec_tokens,
        seq_len=args.seq_len,
        layer_cache_dict=layer_cache_dict,
        cache_path=LAYER_COMPUTE_CACHE_PATH,
    )
    print_final_results(num_layers, results, args.icnt_type, degree > 1)


if __name__ == "__main__":
    main()
