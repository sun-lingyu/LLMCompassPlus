#!/usr/bin/env python3
"""
Design Space Exploration (DSE) for ViT+LLM inference on embedded GPUs.

Searches over SM count and L2 cache size to find hardware configurations
satisfying latency and area constraints.  Memory bandwidth is halved when
the SoC area constraint is 100 mm².

Inference configurations
------------------------
Robo : 2 × InternVision (seq 1024, prefill)
       + Qwen3_4B (seq 512, prefill only)

AD   : 12 × InternVision (seq 1024, prefill)
       + Qwen3_4B (seq 1024, prefill)
       + 7 × Qwen3_4B decode (spec_tokens 64, seq_kv 1024)

Usage
-----
python -m dse.dse \\
    --area 400 --base_hw Orin --inference_config Robo --precision fp16_int4

python -m dse.dse \\
    --area 100 --base_hw Thor --inference_config AD --precision fp8 \\
    --latency_limit 120
"""

import argparse
import copy
import io
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from typing import Dict, Tuple

# ── sys.path: make project root importable ─────────────────────────────────────
_DSE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DSE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from area_model.area_model import calculate_gpu_area  # noqa: E402
from hardware_model.device import _create_device_from_config  # noqa: E402
from simulate.main import (  # noqa: E402
    load_layer_compute_cache,
    run_layer,
    update_layer_cache_record,
)
from test.utils import test_model_dict  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────────────
_AREA_MODEL_DIR = os.path.join(_PROJECT_ROOT, "area_model")
_HW_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "hardware_model", "configs")
DSE_CACHE_PATH = os.path.join(_DSE_DIR, "dse_layer_cache.json")

# ── Model constants ─────────────────────────────────────────────────────────────
_VIT_MODEL = "InternVision"
_LLM_MODEL = "Qwen3_4B"
_VIT_NUM_LAYERS: int = test_model_dict[_VIT_MODEL]["num_layers"]  # 24
_LLM_NUM_LAYERS: int = test_model_dict[_LLM_MODEL]["num_layers"]  # 36

# ── Area budget ─────────────────────────────────────────────────────────────────
GPU_AREA_FRACTION = 0.4  # GPU ≤ 40% of total SoC area

# ── SM / L2 search grids ────────────────────────────────────────────────────────
ORIN_SM_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
ORIN_L2_MB_CANDIDATES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

THOR_SM_CANDIDATES = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
THOR_L2_MB_CANDIDATES = [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0]

# ── Precision rules ─────────────────────────────────────────────────────────────
# ViT precision is fixed per platform (not controlled by CLI)
_VIT_PRECISION: Dict[str, str] = {"Orin": "fp16", "Thor": "fp8"}

# Allowed LLM precision values per platform
LLM_PRECISION_CHOICES: Dict[str, list] = {
    "Orin": ["fp16_int4", "int8"],  # fp16_int4 → fp16 prefill / int4 decode
    "Thor": ["fp8", "fp4"],
}

# ── LLM parallelism (2-SoC UCIe interconnect) ────────────────────────────────
# ViT runs degree=1 on each SoC independently (2 or 12 ViTs split evenly).
# LLM runs across both SoCs with degree=2.
_LLM_DEGREE = 2
_LLM_ICNT_TYPE = "ucie_std"
_LLM_NO_CONTENTION = True
# 128 GB/s link → convert to Gbps for run_layer (comm_bandwidth_gbps parameter)
# main.py applies ×8 (GB/s→Gbps); for degree=2 no ring-factor adjustment needed.
_LLM_COMM_BW_GBPS: float = 128 * 8  # 1024 Gbps
# CP chosen over TP when CP latency is within this factor of TP latency
_CP_VS_TP_THRESHOLD = 1.05


# ══════════════════════════════════════════════════════════════════════════════
# Hardware helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_dse_device_name(
    base_hw: str, sm_count: int, l2_mb: float, halve_bw: bool
) -> str:
    """Unique string that encodes the full hardware configuration for cache keying."""
    name = f"{base_hw}_sm{sm_count}_l2_{l2_mb}mb"
    if halve_bw:
        name += "_halfbw"
    return name


@contextmanager
def _area_model_cwd():
    """
    Temporarily change cwd to area_model/ so that calculate_gpu_area can
    resolve its relative config path  ("configs/<hw>_die_mm.json").
    """
    original = os.getcwd()
    try:
        os.chdir(_AREA_MODEL_DIR)
        yield
    finally:
        os.chdir(original)


def _create_modified_device(
    base_hw: str, sm_count: int, l2_mb: float, halve_bw: bool = False
):
    """
    Load the base JSON config, deep-copy it, apply SM / L2 / BW overrides,
    and return a Device object.
    """
    config_path = os.path.join(_HW_CONFIG_DIR, f"{base_hw}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = copy.deepcopy(config)
    config["compute_module"]["core_count"] = sm_count
    config["compute_module"]["l2_size"] = int(l2_mb * 1024 * 1024)
    if halve_bw:
        config["io_module"]["bandwidth"] = config["io_module"]["bandwidth"] // 2
    return _create_device_from_config(config)


def _check_area_constraint(
    base_hw: str, sm_count: int, l2_mb: float, soc_area_mm2: float
) -> Tuple[bool, float]:
    """
    Returns (passes, gpu_area_mm2).
    Passes when GPU area ≤ soc_area_mm2 × GPU_AREA_FRACTION.
    Suppresses the print inside calculate_gpu_area.
    """
    with _area_model_cwd(), redirect_stdout(io.StringIO()):
        result = calculate_gpu_area(base_hw, sm_count, l2_mb)
    gpu_area: float = result["area_breakdown_mm2"]["total"]
    return gpu_area <= soc_area_mm2 * GPU_AREA_FRACTION, gpu_area


# ══════════════════════════════════════════════════════════════════════════════
# DSE layer cache helpers
# ══════════════════════════════════════════════════════════════════════════════


def _load_dse_cache() -> dict:
    """
    Load the DSE cache from *DSE_CACHE_PATH*.
    Creates an empty JSON list file on first use.
    """
    if not os.path.exists(DSE_CACHE_PATH):
        with open(DSE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
    return load_layer_compute_cache(DSE_CACHE_PATH)


def _build_run_layer_cache(
    dse_cache: dict,
    dse_device_name: str,
    original_device: str,
) -> dict:
    """
    Filter DSE cache entries that match *dse_device_name* and re-key them
    with *original_device* so that run_layer can find them by its own key.

    Translation:
        (dse_device_name, model, prec, …)  →  (original_device, model, prec, …)
    """
    result: dict = {}
    for (
        dev,
        model,
        prec,
        phase,
        spec_t,
        seq_l,
        is_c,
        para,
        deg,
    ), ops in dse_cache.items():
        if dev == dse_device_name:
            run_key = (
                original_device,
                model,
                prec,
                phase,
                spec_t,
                seq_l,
                is_c,
                para,
                deg,
            )
            result[run_key] = ops
    return result


def _flush_new_cache_entries(
    layer_cache_before: set,
    layer_cache_after: dict,
    original_device: str,
    dse_device_name: str,
    dse_cache: dict,
) -> None:
    """
    Identify entries added to *layer_cache_after* that were not in
    *layer_cache_before*, translate their device key back to *dse_device_name*,
    append them to the DSE cache file, and insert them into *dse_cache*.
    """
    for key, ops in layer_cache_after.items():
        if key in layer_cache_before:
            continue
        dev, model, prec, phase, spec_t, seq_l, is_c, para, deg = key
        assert dev == original_device, f"Unexpected device in layer_cache_dict: {dev!r}"
        dse_key = (dse_device_name, model, prec, phase, spec_t, seq_l, is_c, para, deg)
        if dse_key in dse_cache:
            continue  # already persisted (e.g. from a previous run)
        record: dict = {
            "device": dse_device_name,
            "model": model,
            "precision": prec,
            "phase": phase,
            "spec_tokens": spec_t,
            "seq_len": seq_l,
            "is_causal": is_c,
            "parallelism": para,
            "degree": deg,
        }
        # ops keys are LAYER_COMPUTE_OP_NAMES items
        for op_name, op_data in ops.items():
            record[op_name] = op_data
        update_layer_cache_record(record, DSE_CACHE_PATH)
        dse_cache[dse_key] = ops


# ══════════════════════════════════════════════════════════════════════════════
# Layer simulation wrapper
# ══════════════════════════════════════════════════════════════════════════════


def _simulate_one_layer(
    model_name: str,
    precision: str,
    base_hw: str,
    pcb,
    dse_device_name: str,
    dse_cache: dict,
    *,
    M: int,
    seq_len_q: int,
    seq_len_kv: int,
    is_prefill: bool,
    is_causal: bool,
    spec_tokens: int = 64,
    seq_len: int,
    parallelism: str = "TP",
    degree: int = 1,
    comm_bandwidth_gbps: float = 0.0,
    icnt_type: str = "pcie",
    no_contention: bool = False,
) -> Tuple[float, float]:
    """
    Simulate one transformer layer for the given hardware config.
    Returns (total_lat_ms, avg_power_w) for a single layer, where:
      – total_lat_ms = lat_comp  (degree=1)
                     = lat_comp + lat_comm  (degree>1, non-overlapped comm added)
      – avg_power_w is energy-weighted average across compute and (if degree>1)
        the most-efficient interconnect candidate.

    Handles DSE cache transparently:
      – pre-translates DSE cache entries to original-device keys before the call
      – post-translates new entries back to DSE device keys and flushes to disk
    """
    model_cfg = test_model_dict[model_name]

    layer_cache_dict = _build_run_layer_cache(dse_cache, dse_device_name, base_hw)
    layer_cache_before: set = set(layer_cache_dict.keys())

    # cache_path=None: run_layer will not write to any file.
    # layer_cache_dict being non-None is now sufficient for the cache-read branch
    # (main.py was patched to check `if layer_cache_dict is not None` only).
    with redirect_stdout(io.StringIO()):
        results = run_layer(
            model_name,
            model_cfg,
            precision,
            pcb,
            base_hw,  # original device name for HW-specific logic
            M,
            seq_len_q,
            seq_len_kv,
            is_prefill,
            is_causal,
            parallelism=parallelism,
            degree=degree,
            comm_bandwidth_gbps=comm_bandwidth_gbps,
            icnt_type=icnt_type,
            no_contention=no_contention,
            spec_tokens=spec_tokens,
            seq_len=seq_len,
            layer_cache_dict=layer_cache_dict,
            cache_path=None,  # DSE cache managed separately below
        )

    _flush_new_cache_entries(
        layer_cache_before, layer_cache_dict, base_hw, dse_device_name, dse_cache
    )

    lat_comp: float = results["lat_comp"]
    power_comp: float = results["power_avg_comp"]

    if degree == 1 or "lat_comm" not in results:
        return lat_comp, power_comp

    # degree > 1: add non-overlapped comm latency and interconnect power
    lat_comm: float = results["lat_comm"]
    total_lat = lat_comp + lat_comm

    # Pick the most energy-efficient interconnect candidate (minimum comm energy)
    power_results_comm: dict = results.get("power_results_comm", {})
    if power_results_comm:
        min_comm_energy_mj = min(power_results_comm.values())
        avg_power = (power_comp * lat_comp + min_comm_energy_mj) / total_lat
    else:
        avg_power = power_comp

    return total_lat, avg_power


def _simulate_llm_layer(
    precision: str,
    base_hw: str,
    pcb,
    dse_device_name: str,
    dse_cache: dict,
    *,
    seq_len_q: int,
    seq_len_kv: int,
    is_prefill: bool,
    is_causal: bool,
    spec_tokens: int = 64,
    seq_len: int,
) -> Tuple[float, float, str]:
    """
    Simulate one LLM (Qwen3_4B) layer across both SoCs (degree=2).

    For prefill: runs both TP and CP; adopts CP when its latency is within
    _CP_VS_TP_THRESHOLD × TP latency, otherwise adopts TP.
    For decode:  CP is not supported; always uses TP.

    Returns (total_lat_ms, avg_power_w, chosen_parallelism) where
    chosen_parallelism is "TP" or "CP".
    """
    common = {
        "base_hw": base_hw,
        "pcb": pcb,
        "dse_device_name": dse_device_name,
        "dse_cache": dse_cache,
        "seq_len_kv": seq_len_kv,
        "is_prefill": is_prefill,
        "is_causal": is_causal,
        "spec_tokens": spec_tokens,
        "seq_len": seq_len,
        "degree": _LLM_DEGREE,
        "comm_bandwidth_gbps": _LLM_COMM_BW_GBPS,
        "icnt_type": _LLM_ICNT_TYPE,
        "no_contention": _LLM_NO_CONTENTION,
    }

    # TP: M stays at seq_len_q (sequence not split across ranks)
    tp_lat, tp_pow = _simulate_one_layer(
        _LLM_MODEL,
        precision,
        **common,
        M=seq_len_q,
        seq_len_q=seq_len_q,
        parallelism="TP",
    )

    if not is_prefill:
        # decode: CP unsupported
        return tp_lat, tp_pow, "TP"

    # CP prefill: M is divided by degree (each rank handles a contiguous chunk)
    M_cp = seq_len_q // _LLM_DEGREE
    cp_lat, cp_pow = _simulate_one_layer(
        _LLM_MODEL,
        precision,
        **common,
        M=M_cp,
        seq_len_q=seq_len_q,
        parallelism="CP",
    )

    if cp_lat <= tp_lat * _CP_VS_TP_THRESHOLD:
        return cp_lat, cp_pow, "CP"
    return tp_lat, tp_pow, "TP"


# ══════════════════════════════════════════════════════════════════════════════
# Inference latency / power computation
# ══════════════════════════════════════════════════════════════════════════════


def _segment_stats(
    layer_lat_ms: float,
    layer_pow_w: float,
    num_layers: int,
    num_steps: int = 1,
) -> Tuple[float, float]:
    """
    Compute aggregated (total_lat_ms, total_energy_mJ) for a repeated segment.

    Energy: W × ms = mJ  (consistent SI scaling)
    """
    total_lat = layer_lat_ms * num_layers * num_steps
    total_energy = layer_pow_w * layer_lat_ms * num_layers * num_steps
    return total_lat, total_energy


def compute_inference_latency(
    inference_config: str,
    base_hw: str,
    sm_count: int,
    l2_mb: float,
    halve_bw: bool,
    llm_precision: str,
    dse_cache: dict,
) -> dict:
    """
    Simulate the full inference pipeline for one hardware configuration.

    Returns
    -------
    dict with keys:
        total_lat_ms  – total end-to-end latency (ms)
        avg_power_w   – energy-weighted average power (W)
        breakdown     – per-segment latency breakdown (ms)
    """
    dse_device_name = _make_dse_device_name(base_hw, sm_count, l2_mb, halve_bw)
    pcb = _create_modified_device(base_hw, sm_count, l2_mb, halve_bw)

    vit_precision = _VIT_PRECISION[base_hw]

    # Split LLM precision for prefill vs decode
    if llm_precision == "fp16_int4":
        llm_prefill_prec = "fp16"
        llm_decode_prec = "int4"
    else:
        llm_prefill_prec = llm_precision
        llm_decode_prec = llm_precision

    # ── ViT layer (InternVision, prefill, seq 1024, non-causal, degree=1) ──────
    # Each SoC independently handles its share of ViTs; no cross-SoC comm needed.
    VIT_SEQ = 1024
    vit_layer_lat, vit_layer_pow = _simulate_one_layer(
        _VIT_MODEL,
        vit_precision,
        base_hw,
        pcb,
        dse_device_name,
        dse_cache,
        M=VIT_SEQ,
        seq_len_q=VIT_SEQ,
        seq_len_kv=VIT_SEQ,
        is_prefill=True,
        is_causal=False,
        seq_len=VIT_SEQ,  # spec_tokens uses default 64, degree=1
    )

    # ── Robo: 2 × ViT  +  LLM prefill (seq 512, degree=2) ───────────────────
    if inference_config == "Robo":
        LLM_SEQ = 512

        vit_lat, vit_energy = _segment_stats(
            vit_layer_lat, vit_layer_pow, _VIT_NUM_LAYERS, num_steps=1
        )

        llm_layer_lat, llm_layer_pow, llm_prefill_para = _simulate_llm_layer(
            llm_prefill_prec,
            base_hw,
            pcb,
            dse_device_name,
            dse_cache,
            seq_len_q=LLM_SEQ,
            seq_len_kv=LLM_SEQ,
            is_prefill=True,
            is_causal=True,
            seq_len=LLM_SEQ,
        )
        llm_prefill_lat, llm_prefill_energy = _segment_stats(
            llm_layer_lat, llm_layer_pow, _LLM_NUM_LAYERS
        )

        total_lat = vit_lat + llm_prefill_lat
        total_energy = vit_energy + llm_prefill_energy
        breakdown = {
            "vit_lat_ms": vit_lat,
            "llm_prefill_lat_ms": llm_prefill_lat,
            "llm_prefill_para": llm_prefill_para,
        }

    # ── AD: 12 × ViT  +  LLM prefill (seq 1024)  +  7 × decode ─────────────
    elif inference_config == "AD":
        LLM_SEQ = 1024
        SPEC_TOKENS = 64 if llm_precision != "fp4" else 128
        NUM_DECODE_STEPS = 7  # accept 5 tokens/step × 7 steps = 35 tokens

        vit_lat, vit_energy = _segment_stats(
            vit_layer_lat, vit_layer_pow, _VIT_NUM_LAYERS, num_steps=6
        )

        llm_prefill_layer_lat, llm_prefill_layer_pow, llm_prefill_para = (
            _simulate_llm_layer(
                llm_prefill_prec,
                base_hw,
                pcb,
                dse_device_name,
                dse_cache,
                seq_len_q=LLM_SEQ,
                seq_len_kv=LLM_SEQ,
                is_prefill=True,
                is_causal=True,
                spec_tokens=SPEC_TOKENS,
                seq_len=LLM_SEQ,
            )
        )
        llm_prefill_lat, llm_prefill_energy = _segment_stats(
            llm_prefill_layer_lat, llm_prefill_layer_pow, _LLM_NUM_LAYERS
        )

        # Decode: CP is not supported → always TP via _simulate_llm_layer
        llm_decode_layer_lat, llm_decode_layer_pow, _ = _simulate_llm_layer(
            llm_decode_prec,
            base_hw,
            pcb,
            dse_device_name,
            dse_cache,
            seq_len_q=SPEC_TOKENS,
            seq_len_kv=LLM_SEQ,
            is_prefill=False,
            is_causal=False,
            spec_tokens=SPEC_TOKENS,
            seq_len=LLM_SEQ,
        )
        llm_decode_lat, llm_decode_energy = _segment_stats(
            llm_decode_layer_lat,
            llm_decode_layer_pow,
            _LLM_NUM_LAYERS,
            num_steps=NUM_DECODE_STEPS,
        )

        total_lat = vit_lat + llm_prefill_lat + llm_decode_lat
        total_energy = vit_energy + llm_prefill_energy + llm_decode_energy
        breakdown = {
            "vit_lat_ms": vit_lat,
            "llm_prefill_lat_ms": llm_prefill_lat,
            "llm_prefill_para": llm_prefill_para,
            "llm_decode_lat_ms": llm_decode_lat,
        }

    else:
        raise ValueError(f"Unknown inference_config: {inference_config!r}")

    avg_power = total_energy / total_lat if total_lat > 0 else 0.0
    return {
        "total_lat_ms": total_lat,
        "avg_power_w": avg_power,
        "breakdown": breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DSE grid search
# ══════════════════════════════════════════════════════════════════════════════


def run_dse(
    base_hw: str,
    soc_area_mm2: float,
    inference_config: str,
    llm_precision: str,
    latency_limit_ms: float,
) -> list:
    """
    Sweep over (SM count × L2 size) grid, apply area filter first, then
    simulate latency/power for configs that pass.

    Returns a list of result dicts for all evaluated (area-feasible) configs.
    """
    halve_bw = soc_area_mm2 <= 100.0

    # Sort both dimensions descending so pruning can terminate early.
    # Monotonicity: more SM / larger L2 → strictly lower latency.
    sm_desc = sorted(
        ORIN_SM_CANDIDATES if base_hw == "Orin" else THOR_SM_CANDIDATES,
        reverse=True,
    )
    l2_desc = sorted(
        ORIN_L2_MB_CANDIDATES if base_hw == "Orin" else THOR_L2_MB_CANDIDATES,
        reverse=True,
    )
    total = len(sm_desc) * len(l2_desc)

    print("=" * 72)
    print("  DSE: ViT + LLM Hardware Configuration Search")
    print("=" * 72)
    print(f"  Base HW          : {base_hw}")
    print(
        f"  SoC area         : {soc_area_mm2:.0f} mm²"
        f"  (GPU budget ≤ {soc_area_mm2 * GPU_AREA_FRACTION:.0f} mm²)"
    )
    print(f"  Halve memory BW  : {halve_bw}")
    print(f"  Inference config : {inference_config}")
    print(f"  LLM precision    : {llm_precision}")
    print(f"  Latency limit    : {latency_limit_ms} ms")
    print(
        f"  Grid (max)       : {len(sm_desc)} SM × {len(l2_desc)} L2"
        f" = {total} configs (pruning may reduce this)"
    )
    print("-" * 72)

    dse_cache = _load_dse_cache()
    evaluated: list = []

    # l2_cutoff_idx: inner loop runs over l2_desc[0 : l2_cutoff_idx].
    # When (sm, l2_desc[j]) fails latency, latency also fails for any sm' ≤ sm
    # at l2_desc[j] and all smaller L2 values → tighten cutoff to j.
    l2_cutoff_idx = len(l2_desc)
    idx = 0

    for sm in sm_desc:
        if l2_cutoff_idx == 0:
            # Rule 2: every L2 value already failed latency for a larger SM;
            # no smaller SM can do better.
            print(f"  [pruned] SM={sm} and below: all L2 sizes exceed latency limit")
            break

        new_l2_cutoff = l2_cutoff_idx  # only tightened on a latency failure

        for l2_idx in range(l2_cutoff_idx):
            l2 = l2_desc[l2_idx]
            idx += 1
            tag = f"[{idx:3d}/{total}] SM={sm:2d}  L2={l2:5.1f} MB"

            # ── Step 1: area constraint (fast) ─────────────────────────────
            area_ok, gpu_area = _check_area_constraint(base_hw, sm, l2, soc_area_mm2)
            if not area_ok:
                max_area = soc_area_mm2 * GPU_AREA_FRACTION
                print(
                    f"{tag} | GPU {gpu_area:6.1f} mm²"
                    f" > budget {max_area:.1f} mm²  [area skip]"
                )
                continue  # area failures do NOT update the latency cutoff

            # ── Step 2: latency + power simulation ─────────────────────────
            print(
                f"{tag} | GPU {gpu_area:6.1f} mm²  | simulating …",
                end="",
                flush=True,
            )
            try:
                result = compute_inference_latency(
                    inference_config,
                    base_hw,
                    sm,
                    l2,
                    halve_bw,
                    llm_precision,
                    dse_cache,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            lat = result["total_lat_ms"]
            pwr = result["avg_power_w"]
            lat_ok = lat <= latency_limit_ms
            flag = "PASS" if lat_ok else "fail"
            print(f"  lat={lat:7.2f} ms  pwr={pwr:6.1f} W  [{flag}]")

            breakdown_flat = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in result["breakdown"].items()
            }
            evaluated.append(
                {
                    "sm_count": sm,
                    "l2_mb": l2,
                    "gpu_area_mm2": round(gpu_area, 2),
                    "total_lat_ms": round(lat, 4),
                    "avg_power_w": round(pwr, 2),
                    "lat_ok": lat_ok,
                    **breakdown_flat,
                }
            )

            if not lat_ok:
                # Rule 1: smaller L2 for this SM will also fail → stop inner loop.
                # Rule 2: record this L2 index so the next (smaller) SM skips it
                #         and all smaller L2 values.
                new_l2_cutoff = l2_idx
                print(f"  → latency pruning: SM≤{sm}, L2≤{l2:.1f} MB all skipped")
                break

        l2_cutoff_idx = new_l2_cutoff  # apply Rule 2 for the next SM

    return evaluated


# ══════════════════════════════════════════════════════════════════════════════
# Result reporting
# ══════════════════════════════════════════════════════════════════════════════


def _print_summary(
    results: list,
    inference_config: str,
    latency_limit_ms: float,
) -> None:
    passing = [r for r in results if r["lat_ok"]]
    print()
    print("=" * 72)
    print(
        f"  DSE Summary: {len(passing)} / {len(results)} evaluated configs"
        f" meet the {latency_limit_ms:.0f} ms constraint"
    )
    print("=" * 72)

    if not passing:
        print("  No feasible configurations found.")
        return

    if inference_config == "AD":
        lat_cols = ["vit_lat_ms", "llm_prefill_lat_ms", "llm_decode_lat_ms"]
        lat_hdr = (
            f"  {'ViT(ms)':>10}  {'LLMpre(ms)':>11}  {'Para':>4}  {'LLMdec(ms)':>11}"
        )
    else:
        lat_cols = ["vit_lat_ms", "llm_prefill_lat_ms"]
        lat_hdr = f"  {'ViT(ms)':>10}  {'LLMpre(ms)':>11}  {'Para':>4}"

    main_hdr = (
        f"  {'SM':>4}  {'L2(MB)':>7}  {'GPU(mm²)':>9}  "
        f"{'Total(ms)':>10}  {'Power(W)':>9}"
    )
    print(main_hdr + lat_hdr)
    print("  " + "-" * (len(main_hdr) + len(lat_hdr) - 2))

    for r in sorted(passing, key=lambda x: x["total_lat_ms"]):
        row = (
            f"  {r['sm_count']:>4}  {r['l2_mb']:>7.1f}  {r['gpu_area_mm2']:>9.1f}  "
            f"{r['total_lat_ms']:>10.2f}  {r['avg_power_w']:>9.1f}"
        )
        for col in lat_cols:
            row += f"  {r.get(col, 0.0):>10.2f}"
            # show chosen parallelism right after LLM-prefill latency
            if col == "llm_prefill_lat_ms":
                row += f"  {r.get('llm_prefill_para', '?'):>4}"
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DSE for ViT+LLM inference: search SM count and L2 cache size "
            "under latency and area constraints."
        )
    )
    parser.add_argument(
        "--area",
        type=int,
        required=True,
        choices=[100, 400],
        help="SoC area in mm² (100 or 400).  GPU is budgeted at ≤40%% of this.",
    )
    parser.add_argument(
        "--base_hw",
        required=True,
        choices=["Orin", "Thor"],
        help="Base hardware platform whose config JSON is used as the starting point.",
    )
    parser.add_argument(
        "--inference_config",
        required=True,
        choices=["Robo", "AD"],
        help=(
            "Robo: 2×InternVision (seq 1024) + Qwen3_4B prefill (seq 512).  "
            "AD:   12×InternVision (seq 1024) + Qwen3_4B prefill+decode (seq 1024)."
        ),
    )
    parser.add_argument(
        "--precision",
        required=True,
        help=(
            "LLM precision.  "
            "Orin: fp16_int4 (fp16 prefill / int4 decode) | int8.  "
            "Thor: fp8 | fp4."
        ),
    )
    parser.add_argument(
        "--latency_limit",
        type=float,
        default=105.0,
        help="End-to-end latency constraint in ms (default: 100ms + 5ms margin).",
    )
    args = parser.parse_args()

    valid_precisions = LLM_PRECISION_CHOICES[args.base_hw]
    if args.precision not in valid_precisions:
        parser.error(
            f"For {args.base_hw}, --precision must be one of {valid_precisions}. "
            f"Got: {args.precision!r}"
        )

    results = run_dse(
        base_hw=args.base_hw,
        soc_area_mm2=float(args.area),
        inference_config=args.inference_config,
        llm_precision=args.precision,
        latency_limit_ms=args.latency_limit,
    )

    _print_summary(results, args.inference_config, args.latency_limit)


if __name__ == "__main__":
    main()
