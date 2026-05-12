#!/usr/bin/env python3
"""
Design Space Exploration (DSE) for ViT+LLM inference on embedded GPUs.

Grid search over SM count × L2 cache size for three workload sizes (L/M/S)
sequentially.  A hardware configuration is considered feasible only when it
meets the latency constraint for ALL three sizes simultaneously.

Inference configurations
------------------------
inference_config (CLI)  size   num_ViT  ViT seq  LLM model   LLM seq  decode iters
──────────────────────  ─────  ───────  ───────  ──────────  ───────  ────────────
Robo                    L      2        1024      Qwen3_8B    512      –
                        M      1        1024      Qwen3_4B    256      –
                        S      1        576       Qwen3_1_7B  144      –
AD                      L      12       1024      Qwen3_8B    1024     5
                        M      6        1024      Qwen3_4B    768      5
                        S      6        576       Qwen3_1_7B  512      5

LLM model / degree mapping (fixed)
-----------------------------------
Qwen3_8B   → degree=4  (quad-card,   L size)
Qwen3_4B   → degree=2  (dual-card,   M size)
Qwen3_1_7B → degree=1  (single-card, S size)

Usage
-----
python -m dse.dse \\
    --area 400 --base_hw Orin --inference_config Robo --precision fp16_int4

python -m dse.dse \\
    --base_hw Thor --inference_config AD --precision fp8 \\
    --area 200 --mem_freq 10667 --mem_bitwidth 128 --latency_limit 120
"""

import argparse
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
_VIT_NUM_LAYERS: int = test_model_dict[_VIT_MODEL]["num_layers"]  # 24

# ── Area budget ─────────────────────────────────────────────────────────────────
GPU_AREA_FRACTION = 0.35  # GPU ≤ 35% of total SoC area

# ── SM / L2 search grids ────────────────────────────────────────────────────────
ORIN_SM_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
ORIN_L2_MB_CANDIDATES = [2.0, 4.0, 8.0]

THOR_SM_CANDIDATES = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
THOR_L2_MB_CANDIDATES = [16.0, 32.0, 64.0]

# ── L2 bandwidth sub-unit parameters (per platform) ─────────────────────────
_SM_BW_PER_CYCLE: Dict[str, int] = {"Orin": 32, "Thor": 64}  # B/clk/SM
_FBP_BW_PER_UNIT: int = 256  # B/clk/FBP
_BASE_FBP_COUNT: Dict[str, int] = {"Orin": 2, "Thor": 4}
_BASE_L2_MB: Dict[str, float] = {"Orin": 4.0, "Thor": 32.0}
_L2S_BW_PER_CYCLE: Dict[str, int] = {"Orin": 32, "Thor": 64}  # B/clk/slice
_BASE_L2S_COUNT: Dict[str, int] = {"Orin": 16, "Thor": 24}

# ── Precision rules ─────────────────────────────────────────────────────────────
# ViT precision is fixed per platform (not controlled by CLI)
_VIT_PRECISION: Dict[str, str] = {"Orin": "fp16", "Thor": "fp8"}

# Allowed LLM precision values per platform
LLM_PRECISION_CHOICES: Dict[str, list] = {
    "Orin": ["fp16_int4", "int8"],  # fp16_int4 → fp16 prefill / int4 decode
    "Thor": ["fp8", "fp4"],
}

# ── LLM / ViT interconnect ───────────────────────────────────────────────────
_LLM_ICNT_TYPE = "ucie_std"
_LLM_NO_CONTENTION = True
# Single UCIe link: 128 GB/s → 128 * 8 Gbps.  _comm_bw_for_degree() scales this
# by the number of active links (×1 for degree=2, ×2 for degree=4).
_LINK_BW_GBPS: float = 128 * 8  # 128 * 8 Gbps per link
# CP chosen over TP when CP latency is within this factor of TP latency
_CP_VS_TP_THRESHOLD = 1.05

# LLM model → fixed degree mapping (8B→4-card, 4B→2-card, 1.7B→1-card)
_MODEL_DEFAULT_DEGREE: Dict[str, int] = {
    "Qwen3_8B": 4,
    "Qwen3_4B": 2,
    "Qwen3_1_7B": 1,
}

# ── Inference configuration definitions ─────────────────────────────────────
# Each (inference_config, size) tuple uniquely determines all workload params.
# num_decode_steps=0 means prefill-only (Robo); >0 adds speculative-decode iters.
_INFERENCE_CONFIGS: Dict[str, Dict[str, dict]] = {
    "Robo": {
        "L": {
            "num_vits": 2,
            "vit_seq": 1024,
            "llm_model": "Qwen3_8B",
            "llm_seq": 512,
            "degree": 4,
            "num_decode_steps": 0,
        },
        "M": {
            "num_vits": 1,
            "vit_seq": 1024,
            "llm_model": "Qwen3_4B",
            "llm_seq": 256,
            "degree": 2,
            "num_decode_steps": 0,
        },
        "S": {
            "num_vits": 1,
            "vit_seq": 576,
            "llm_model": "Qwen3_1_7B",
            "llm_seq": 144,
            "degree": 1,
            "num_decode_steps": 0,
        },
    },
    "AD": {
        "L": {
            "num_vits": 12,
            "vit_seq": 1024,
            "llm_model": "Qwen3_8B",
            "llm_seq": 1024,
            "degree": 4,
            "num_decode_steps": 5,
        },
        "M": {
            "num_vits": 6,
            "vit_seq": 1024,
            "llm_model": "Qwen3_4B",
            "llm_seq": 768,
            "degree": 2,
            "num_decode_steps": 5,
        },
        "S": {
            "num_vits": 6,
            "vit_seq": 576,
            "llm_model": "Qwen3_1_7B",
            "llm_seq": 512,
            "degree": 1,
            "num_decode_steps": 5,
        },
    },
}


def _comm_bw_for_degree(degree: int) -> float:
    """Return total comm bandwidth (Gbps) for the given LLM tensor-parallel degree.

    degree=1 → no interconnect (0 Gbps).
    degree=2 → one UCIe link (_LINK_BW_GBPS).
    degree=4 → two UCIe links (2 × _LINK_BW_GBPS).
    """
    if degree <= 1:
        return 0.0
    return _LINK_BW_GBPS * (2 if degree == 4 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# Hardware helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_dse_device_name(
    base_hw: str, sm_count: int, l2_mb: float, mem_freq: int, mem_bitwidth: int
) -> str:
    """Unique string that encodes the full hardware configuration for cache keying."""
    return f"{base_hw}_sm{sm_count}_l2_{l2_mb}mb_{mem_freq}_{mem_bitwidth}bit"


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


def _compute_l2_bw_per_cycle(base_hw: str, sm_count: int, l2_mb: float) -> int:
    """
    Compute l2_bandwidth_per_cycle (B/clk) as the minimum of three limits:
      - SM-side  : SM_BW_PER_CYCLE[hw] × sm_count
      - FBP-side : FBP_BW_PER_UNIT × FBP_COUNT  (FBP_COUNT ∝ l2_mb, min 1)
      - L2S-side : L2S_BW_PER_CYCLE[hw] × L2S_COUNT  (L2S_COUNT ∝ l2_mb)
    """
    base_l2_mb = _BASE_L2_MB[base_hw]
    sm_bw = _SM_BW_PER_CYCLE[base_hw] * sm_count
    FBP_COUNT = max(1.0, _BASE_FBP_COUNT[base_hw] * l2_mb / base_l2_mb)
    assert FBP_COUNT.is_integer(), f"FBP_COUNT is not an integer: {FBP_COUNT}"
    fbp_bw = _FBP_BW_PER_UNIT * FBP_COUNT
    L2S_COUNT = _BASE_L2S_COUNT[base_hw] * l2_mb / base_l2_mb
    assert L2S_COUNT.is_integer(), f"L2S_COUNT is not an integer: {L2S_COUNT}"
    l2s_bw = _L2S_BW_PER_CYCLE[base_hw] * L2S_COUNT
    return int(min(sm_bw, fbp_bw, l2s_bw))


def _create_modified_device(
    base_hw: str, sm_count: int, l2_mb: float, mem_freq: int, mem_bitwidth: int
):
    config_path = os.path.join(_HW_CONFIG_DIR, f"{base_hw}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["compute_module"]["core_count"] = sm_count
    config["compute_module"]["l2_size"] = int(l2_mb * 1024 * 1024)
    config["io_module"]["bandwidth"] = mem_freq * mem_bitwidth * 1e6 / 8
    config["compute_module"]["l2_bandwidth_per_cycle"] = _compute_l2_bw_per_cycle(
        base_hw, sm_count, l2_mb
    )
    return _create_device_from_config(config)


def _check_l2_bw_constraint(
    base_hw: str, sm_count: int, l2_mb: float, mem_freq: int, mem_bitwidth: int
) -> bool:
    """
    Return True when L2 bandwidth (B/s) ≥ memory bandwidth (B/s).
    Configs where L2 is slower than the memory bus are physically unsound.
    """
    config_path = os.path.join(_HW_CONFIG_DIR, f"{base_hw}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    clock_freq: int = config["compute_module"]["clock_freq"]
    l2_bw_bps = _compute_l2_bw_per_cycle(base_hw, sm_count, l2_mb) * clock_freq
    mem_bw_bps = mem_freq * mem_bitwidth * 1e6 / 8
    return l2_bw_bps >= mem_bw_bps


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
    _split_out: dict = None,
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

    _peak_comp: float = results.get("peak_power_comp", power_comp)

    if degree == 1 or "lat_comm" not in results:
        if _split_out is not None:
            _split_out["chip"] = results.get("power_avg_comp_chip", power_comp)
            _split_out["dram"] = results.get("power_avg_comp_dram", 0.0)
            _split_out["peak"] = _peak_comp
        return lat_comp, power_comp

    # degree > 1: add non-overlapped comm latency and interconnect power
    lat_comm: float = results["lat_comm"]
    total_lat = lat_comp + lat_comm

    # Pick the most energy-efficient interconnect candidate (minimum comm energy)
    power_results_comm: dict = results.get("power_results_comm", {})
    if power_results_comm:
        min_comm_energy_mj = min(power_results_comm.values())
        avg_power = (power_comp * lat_comp + min_comm_energy_mj) / total_lat
        if _split_out is not None:
            _chip = results.get("power_avg_comp_chip", power_comp)
            _dram = results.get("power_avg_comp_dram", 0.0)
            # Interconnect energy is SoC-side signaling → attributed to chip
            _split_out["chip"] = (_chip * lat_comp + min_comm_energy_mj) / total_lat
            _split_out["dram"] = _dram * lat_comp / total_lat
            _split_out["peak"] = _peak_comp
    else:
        avg_power = power_comp
        if _split_out is not None:
            _split_out["chip"] = results.get("power_avg_comp_chip", power_comp)
            _split_out["dram"] = results.get("power_avg_comp_dram", 0.0)
            _split_out["peak"] = _peak_comp

    return total_lat, avg_power


def _simulate_llm_layer(
    model_name: str,
    precision: str,
    base_hw: str,
    pcb,
    dse_device_name: str,
    dse_cache: dict,
    *,
    degree: int,
    comm_bw_gbps: float,
    seq_len_q: int,
    seq_len_kv: int,
    is_prefill: bool,
    is_causal: bool,
    spec_tokens: int = 64,
    seq_len: int,
    _split_out: dict = None,
) -> Tuple[float, float, str]:
    """
    Simulate one LLM layer across *degree* SoCs.

    For prefill: runs both TP and CP; adopts CP when its latency is within
    _CP_VS_TP_THRESHOLD × TP latency, otherwise adopts TP.
    For decode:  CP is not supported; always uses TP.
    For degree=1: no interconnect, always TP.

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
        "degree": degree,
        "comm_bandwidth_gbps": comm_bw_gbps,
        "icnt_type": _LLM_ICNT_TYPE,
        "no_contention": _LLM_NO_CONTENTION,
    }

    # TP: M stays at seq_len_q (sequence not split across ranks)
    tp_split: dict = {} if _split_out is not None else None
    tp_lat, tp_pow = _simulate_one_layer(
        model_name,
        precision,
        **common,
        M=seq_len_q,
        seq_len_q=seq_len_q,
        parallelism="TP",
        _split_out=tp_split,
    )

    if not is_prefill or degree <= 1:
        # decode or single-device: CP unsupported / irrelevant
        if _split_out is not None:
            _split_out.update(tp_split)
        return tp_lat, tp_pow, "TP"

    # CP prefill: M is divided by degree (each rank handles a contiguous chunk)
    M_cp = seq_len_q // degree
    cp_split: dict = {} if _split_out is not None else None
    cp_lat, cp_pow = _simulate_one_layer(
        model_name,
        precision,
        **common,
        M=M_cp,
        seq_len_q=seq_len_q,
        parallelism="CP",
        _split_out=cp_split,
    )

    if cp_lat <= tp_lat * _CP_VS_TP_THRESHOLD:
        if _split_out is not None:
            _split_out.update(cp_split)
        return cp_lat, cp_pow, "CP"
    if _split_out is not None:
        _split_out.update(tp_split)
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


def _compute_vit_stats(
    num_vits: int,
    vit_layer_lat: float,
    vit_layer_pow: float,
    vit_precision: str,
    base_hw: str,
    pcb,
    dse_device_name: str,
    dse_cache: dict,
    *,
    degree: int,
    vit_seq: int,
    _chip_layer_pow: float = None,
    _dram_layer_pow: float = None,
    _split_out: dict = None,
) -> Tuple[float, float]:
    """
    Compute total ViT latency and energy when *num_vits* ViTs are distributed
    evenly across *degree* devices.

    - Integer ViTs per device  → simulated at degree=1 (already in vit_layer_lat).
    - Fractional part of 0.5  → one extra ViT simulated at degree=2 TP using the
      single-link interconnect (_LINK_BW_GBPS), added to the total.
    """
    VIT_SEQ = vit_seq
    full_steps = num_vits // degree
    remainder = num_vits % degree
    if remainder == 0:
        has_half = False
        has_quarter = False
    elif degree >= 2 and remainder == degree // 2:
        has_half = True
        has_quarter = False
    elif degree >= 4 and remainder == degree // 4:
        has_half = False
        has_quarter = True
    else:
        raise ValueError(
            f"num_vits={num_vits} on degree={degree} devices (remainder={remainder}). "
            f"Only remainder==0, degree//2 (half), or degree//4 (quarter) are supported."
        )

    total_lat = 0.0
    total_energy = 0.0
    total_chip_energy = 0.0
    total_dram_energy = 0.0
    vit_peak = 0.0
    _track = _split_out is not None

    if full_steps > 0:
        lat, energy = _segment_stats(
            vit_layer_lat, vit_layer_pow, _VIT_NUM_LAYERS, num_steps=full_steps
        )
        total_lat += lat
        total_energy += energy
        if _track and _chip_layer_pow is not None:
            total_chip_energy += _chip_layer_pow * vit_layer_lat * _VIT_NUM_LAYERS * full_steps
            total_dram_energy += (_dram_layer_pow or 0.0) * vit_layer_lat * _VIT_NUM_LAYERS * full_steps
            vit_peak = max(vit_peak, _chip_layer_pow + (_dram_layer_pow or 0.0))

    if has_half:
        # One ViT shared across 2 devices → simulate with degree=2 TP
        _half_split: dict = {} if _track else None
        half_vit_lat, half_vit_pow = _simulate_one_layer(
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
            seq_len=VIT_SEQ,
            degree=2,
            parallelism="TP",
            comm_bandwidth_gbps=_LINK_BW_GBPS,
            icnt_type=_LLM_ICNT_TYPE,
            no_contention=_LLM_NO_CONTENTION,
            _split_out=_half_split,
        )
        lat, energy = _segment_stats(half_vit_lat, half_vit_pow, _VIT_NUM_LAYERS)
        total_lat += lat
        total_energy += energy
        if _track and _half_split:
            total_chip_energy += _half_split.get("chip", half_vit_pow) * half_vit_lat * _VIT_NUM_LAYERS
            total_dram_energy += _half_split.get("dram", 0.0) * half_vit_lat * _VIT_NUM_LAYERS
            vit_peak = max(vit_peak, _half_split.get("peak", half_vit_pow))

    if has_quarter:
        # One ViT shared across 4 devices → simulate with degree=4 TP
        _quarter_split: dict = {} if _track else None
        quarter_vit_lat, quarter_vit_pow = _simulate_one_layer(
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
            seq_len=VIT_SEQ,
            degree=4,
            parallelism="TP",
            comm_bandwidth_gbps=_comm_bw_for_degree(4),
            icnt_type=_LLM_ICNT_TYPE,
            no_contention=_LLM_NO_CONTENTION,
            _split_out=_quarter_split,
        )
        lat, energy = _segment_stats(quarter_vit_lat, quarter_vit_pow, _VIT_NUM_LAYERS)
        total_lat += lat
        total_energy += energy
        if _track and _quarter_split:
            total_chip_energy += _quarter_split.get("chip", quarter_vit_pow) * quarter_vit_lat * _VIT_NUM_LAYERS
            total_dram_energy += _quarter_split.get("dram", 0.0) * quarter_vit_lat * _VIT_NUM_LAYERS
            vit_peak = max(vit_peak, _quarter_split.get("peak", quarter_vit_pow))

    if _track and total_lat > 0:
        _split_out["chip"] = total_chip_energy / total_lat
        _split_out["dram"] = total_dram_energy / total_lat
        _split_out["peak"] = vit_peak

    return total_lat, total_energy


def compute_inference_latency(
    inference_config: str,
    size: str,
    base_hw: str,
    sm_count: int,
    l2_mb: float,
    mem_freq: int,
    mem_bitwidth: int,
    llm_precision: str,
    dse_cache: dict,
) -> dict:
    """
    Simulate the full inference pipeline for one hardware configuration.

    Parameters
    ----------
    inference_config : "Robo" or "AD"
    size             : "L", "M", or "S"

    Returns
    -------
    dict with keys:
        total_lat_ms  – total end-to-end latency (ms)
        avg_power_w   – energy-weighted average power (W)
        breakdown     – per-segment latency breakdown (ms)
    """
    cfg = _INFERENCE_CONFIGS[inference_config][size]
    num_vits: int = cfg["num_vits"]
    VIT_SEQ: int = cfg["vit_seq"]
    llm_model: str = cfg["llm_model"]
    LLM_SEQ: int = cfg["llm_seq"]
    llm_degree: int = cfg["degree"]
    num_decode_steps: int = cfg["num_decode_steps"]

    dse_device_name = _make_dse_device_name(
        base_hw, sm_count, l2_mb, mem_freq, mem_bitwidth
    )
    pcb = _create_modified_device(base_hw, sm_count, l2_mb, mem_freq, mem_bitwidth)

    vit_precision = _VIT_PRECISION[base_hw]
    llm_num_layers: int = test_model_dict[llm_model]["num_layers"]
    llm_comm_bw = _comm_bw_for_degree(llm_degree)

    if llm_precision == "fp16_int4":
        llm_prefill_prec = "fp16"
        llm_decode_prec = "int4"
    else:
        llm_prefill_prec = llm_precision
        llm_decode_prec = llm_precision

    SPEC_TOKENS = 64 if llm_precision != "fp4" else 128

    # ── ViT layer baseline (degree=1 per-device result) ──────────────────────
    _vit_layer_split: dict = {}
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
        seq_len=VIT_SEQ,
        _split_out=_vit_layer_split,
    )

    _llm_common = {
        "model_name": llm_model,
        "base_hw": base_hw,
        "pcb": pcb,
        "dse_device_name": dse_device_name,
        "dse_cache": dse_cache,
        "degree": llm_degree,
        "comm_bw_gbps": llm_comm_bw,
    }

    # ── ViT pipeline ──────────────────────────────────────────────────────────
    _vit_pipeline_split: dict = {}
    vit_lat, vit_energy = _compute_vit_stats(
        num_vits,
        vit_layer_lat,
        vit_layer_pow,
        vit_precision,
        base_hw,
        pcb,
        dse_device_name,
        dse_cache,
        degree=llm_degree,
        vit_seq=VIT_SEQ,
        _chip_layer_pow=_vit_layer_split.get("chip"),
        _dram_layer_pow=_vit_layer_split.get("dram"),
        _split_out=_vit_pipeline_split,
    )

    # ── LLM prefill ───────────────────────────────────────────────────────────
    _prefill_split: dict = {}
    llm_prefill_layer_lat, llm_prefill_layer_pow, llm_prefill_para = (
        _simulate_llm_layer(
            **_llm_common,
            precision=llm_prefill_prec,
            seq_len_q=LLM_SEQ,
            seq_len_kv=LLM_SEQ,
            is_prefill=True,
            is_causal=True,
            spec_tokens=SPEC_TOKENS,
            seq_len=LLM_SEQ,
            _split_out=_prefill_split,
        )
    )
    llm_prefill_lat, llm_prefill_energy = _segment_stats(
        llm_prefill_layer_lat, llm_prefill_layer_pow, llm_num_layers
    )

    total_lat = vit_lat + llm_prefill_lat
    total_energy = vit_energy + llm_prefill_energy
    # Chip/DRAM energy accumulation
    total_chip_energy = (
        _vit_pipeline_split.get("chip", vit_layer_pow) * vit_lat
        + _prefill_split.get("chip", llm_prefill_layer_pow)
        * llm_prefill_layer_lat
        * llm_num_layers
    )
    total_dram_energy = (
        _vit_pipeline_split.get("dram", 0.0) * vit_lat
        + _prefill_split.get("dram", 0.0) * llm_prefill_layer_lat * llm_num_layers
    )
    # Peak power: max single-operator power across all segments (per card)
    peak_power = max(
        _vit_pipeline_split.get("peak", vit_layer_pow),
        _prefill_split.get("peak", llm_prefill_layer_pow),
    )
    breakdown: dict = {
        "vit_lat_ms": vit_lat,
        "llm_prefill_lat_ms": llm_prefill_lat,
        "llm_prefill_para": llm_prefill_para,
    }

    # ── LLM speculative decode (AD only) ─────────────────────────────────────
    if num_decode_steps > 0:
        _decode_split: dict = {}
        llm_decode_layer_lat, llm_decode_layer_pow, _ = _simulate_llm_layer(
            **_llm_common,
            precision=llm_decode_prec,
            seq_len_q=SPEC_TOKENS,
            seq_len_kv=LLM_SEQ,
            is_prefill=False,
            is_causal=False,
            spec_tokens=SPEC_TOKENS,
            seq_len=LLM_SEQ,
            _split_out=_decode_split,
        )
        llm_decode_lat, llm_decode_energy = _segment_stats(
            llm_decode_layer_lat,
            llm_decode_layer_pow,
            llm_num_layers,
            num_steps=num_decode_steps,
        )
        total_lat += llm_decode_lat
        total_energy += llm_decode_energy
        total_chip_energy += (
            _decode_split.get("chip", llm_decode_layer_pow)
            * llm_decode_layer_lat
            * llm_num_layers
            * num_decode_steps
        )
        total_dram_energy += (
            _decode_split.get("dram", 0.0)
            * llm_decode_layer_lat
            * llm_num_layers
            * num_decode_steps
        )
        peak_power = max(peak_power, _decode_split.get("peak", llm_decode_layer_pow))
        breakdown["llm_decode_lat_ms"] = llm_decode_lat

    avg_power = total_energy / total_lat if total_lat > 0 else 0.0
    chip_power = total_chip_energy / total_lat if total_lat > 0 else 0.0
    dram_power = total_dram_energy / total_lat if total_lat > 0 else 0.0
    return {
        "total_lat_ms": total_lat,
        "avg_power_w": avg_power,
        "chip_power_w": chip_power,
        "dram_power_w": dram_power,
        "peak_power_w": peak_power,
        "breakdown": breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DSE grid search
# ══════════════════════════════════════════════════════════════════════════════


def run_dse(
    base_hw: str,
    soc_area_mm2: float,
    inference_config: str,
    size: str,
    llm_precision: str,
    latency_limit_ms: float,
    mem_freq: int,
    mem_bitwidth: int,
    dse_cache: dict,
) -> list:
    """
    Sweep over (SM count × L2 size) grid for one (inference_config, size) pair.

    Applies area filter first, then simulates latency/power for configs that
    pass.  Returns a list of result dicts for all evaluated (area-feasible)
    configs.
    """
    cfg = _INFERENCE_CONFIGS[inference_config][size]
    llm_model: str = cfg["llm_model"]
    llm_degree: int = cfg["degree"]

    sm_desc = sorted(
        ORIN_SM_CANDIDATES if base_hw == "Orin" else THOR_SM_CANDIDATES,
        reverse=True,
    )
    l2_desc = sorted(
        ORIN_L2_MB_CANDIDATES if base_hw == "Orin" else THOR_L2_MB_CANDIDATES,
        reverse=True,
    )
    total = len(sm_desc) * len(l2_desc)

    mem_bw = mem_freq * mem_bitwidth / 8 / 1000  # GB/s

    print("=" * 72)
    print(f"  DSE Grid Search: {inference_config}-{size}")
    print("=" * 72)
    print(f"  Base HW          : {base_hw}")
    print(
        f"  SoC area         : {soc_area_mm2:.0f} mm²"
        f"  (GPU budget ≤ {soc_area_mm2 * GPU_AREA_FRACTION:.0f} mm²)"
    )
    print(
        f"  Memory           : {mem_freq} MT/s  ×  {mem_bitwidth}-bit  ({mem_bw:.1f} GB/s)"
    )
    print(f"  Inference config : {inference_config}-{size}")
    print(f"  LLM model        : {llm_model}  (degree={llm_degree})")
    print(f"  LLM precision    : {llm_precision}")
    print(f"  Latency limit    : {latency_limit_ms} ms")
    print(
        f"  Grid (max)       : {len(sm_desc)} SM × {len(l2_desc)} L2"
        f" = {total} configs (pruning may reduce this)"
    )
    print("-" * 72)

    evaluated: list = []

    # l2_cutoff_idx: inner loop runs over l2_desc[0 : l2_cutoff_idx].
    # When (sm, l2_desc[j]) fails latency, latency also fails for any sm' ≤ sm
    # at l2_desc[j] and all smaller L2 values → tighten cutoff to j.
    l2_cutoff_idx = len(l2_desc)
    idx = 0

    for sm in sm_desc:
        if l2_cutoff_idx == 0:
            print(f"  [pruned] SM={sm} and below: all L2 sizes exceed latency limit")
            break

        new_l2_cutoff = l2_cutoff_idx

        for l2_idx in range(l2_cutoff_idx):
            l2 = l2_desc[l2_idx]
            idx += 1
            tag = f"[{idx:3d}/{total}] SM={sm:2d}  L2={l2:5.1f} MB"

            # ── Step 0: L2 BW vs mem BW (fast) ───────────────────────────────
            if not _check_l2_bw_constraint(base_hw, sm, l2, mem_freq, mem_bitwidth):
                print(f"{tag} | L2 BW < mem BW  [l2bw skip]")
                continue

            # ── Step 1: area constraint (fast) ───────────────────────────────
            area_ok, gpu_area = _check_area_constraint(base_hw, sm, l2, soc_area_mm2)
            if not area_ok:
                max_area = soc_area_mm2 * GPU_AREA_FRACTION
                print(
                    f"{tag} | GPU {gpu_area:6.1f} mm²"
                    f" > budget {max_area:.1f} mm²  [area skip]"
                )
                continue  # area failures do NOT update the latency cutoff

            # ── Step 2: latency + power simulation ───────────────────────────
            print(
                f"{tag} | GPU {gpu_area:6.1f} mm²  | simulating …",
                end="",
                flush=True,
            )
            try:
                result = compute_inference_latency(
                    inference_config,
                    size,
                    base_hw,
                    sm,
                    l2,
                    mem_freq,
                    mem_bitwidth,
                    llm_precision,
                    dse_cache,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            lat = result["total_lat_ms"]
            pwr = result["avg_power_w"]
            chip_pwr = result.get("chip_power_w", pwr)
            dram_pwr = result.get("dram_power_w", 0.0)
            peak_pwr = result.get("peak_power_w", pwr)
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
                    "l2_bw_per_cycle": _compute_l2_bw_per_cycle(base_hw, sm, l2),
                    "gpu_area_mm2": round(gpu_area, 2),
                    "total_lat_ms": round(lat, 4),
                    "avg_power_w": round(pwr, 2),
                    "chip_power_w": round(chip_pwr, 2),
                    "dram_power_w": round(dram_pwr, 2),
                    "peak_power_w": round(peak_pwr, 2),
                    "lat_ok": lat_ok,
                    **breakdown_flat,
                }
            )

            if not lat_ok:
                # Rule 1: smaller L2 for this SM will also fail → stop inner loop.
                # Rule 2: record this L2 index so the next (smaller) SM skips it.
                new_l2_cutoff = l2_idx
                print(f"  → latency pruning: SM≤{sm}, L2≤{l2:.1f} MB all skipped")
                break

        l2_cutoff_idx = new_l2_cutoff

    return evaluated


# ══════════════════════════════════════════════════════════════════════════════
# Result reporting
# ══════════════════════════════════════════════════════════════════════════════


def _print_summary(
    results: list,
    inference_config: str,
    latency_limit_ms: float,
    *,
    label: str = "",
) -> None:
    passing = [r for r in results if r["lat_ok"]]
    print()
    print("=" * 72)
    if label:
        print(f"  {label}")
    print(
        f"  DSE Summary: {len(passing)} / {len(results)} evaluated configs"
        f" meet the {latency_limit_ms:.0f} ms constraint"
    )
    print("=" * 72)

    if not passing:
        print("  No feasible configurations found.")
        return

    has_decode = any("llm_decode_lat_ms" in r for r in passing)
    if has_decode:
        lat_cols = ["vit_lat_ms", "llm_prefill_lat_ms", "llm_decode_lat_ms"]
        lat_hdr = (
            f"  {'ViT(ms)':>10}  {'LLMpre(ms)':>11}  {'Para':>4}  {'LLMdec(ms)':>11}"
        )
    else:
        lat_cols = ["vit_lat_ms", "llm_prefill_lat_ms"]
        lat_hdr = f"  {'ViT(ms)':>10}  {'LLMpre(ms)':>11}  {'Para':>4}"

    main_hdr = (
        f"  {'SM':>4}  {'L2(MB)':>7}  {'L2BW(B/clk)':>12}  {'GPU(mm²)':>9}  "
        f"{'Total(ms)':>10}  {'Power(W)':>9}"
    )
    print(main_hdr + lat_hdr)
    print("  " + "-" * (len(main_hdr) + len(lat_hdr) - 2))

    for r in sorted(passing, key=lambda x: x["total_lat_ms"]):
        row = (
            f"  {r['sm_count']:>4}  {r['l2_mb']:>7.1f}  {r['l2_bw_per_cycle']:>12d}  {r['gpu_area_mm2']:>9.1f}  "
            f"{r['total_lat_ms']:>10.2f}  {r['avg_power_w']:>9.1f}"
        )
        for col in lat_cols:
            row += f"  {r.get(col, 0.0):>10.2f}"
            if col == "llm_prefill_lat_ms":
                row += f"  {r.get('llm_prefill_para', '?'):>4}"
        print(row)


def _print_final_summary(
    results_by_size: Dict[str, list],
    passing_keys: set,
    inference_config: str,
    latency_limit_ms: float,
) -> None:
    """
    Print hardware configs that satisfy the latency constraint for ALL three
    sizes (L+4-card, M+2-card, S+1-card).

    Shows the total latency for each size in separate columns.
    """
    print()
    print("=" * 72)
    print(f"  Final Summary  —  {inference_config}: configs passing ALL sizes")
    print(
        f"  (L + 4-card,  M + 2-card,  S + 1-card,  limit = {latency_limit_ms:.0f} ms each)"
    )
    print("=" * 72)

    if not passing_keys:
        print("  No hardware configuration satisfies all three size constraints.")
        return

    # Build per-size lookup: (sm, l2) → result dict
    lookup: Dict[str, dict] = {s: {} for s in ("L", "M", "S")}
    for size, results in results_by_size.items():
        for r in results:
            lookup[size][(r["sm_count"], r["l2_mb"])] = r

    hdr = (
        f"  {'SM':>4}  {'L2(MB)':>7}  {'L2BW(B/clk)':>12}  {'GPU(mm²)':>9}"
        f"  {'L-Total(ms)':>12}  {'M-Total(ms)':>12}  {'S-Total(ms)':>12}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key in sorted(
        passing_keys, key=lambda k: lookup["L"].get(k, {}).get("total_lat_ms", 0)
    ):
        sm, l2 = key
        r_l = lookup["L"].get(key, {})
        r_m = lookup["M"].get(key, {})
        r_s = lookup["S"].get(key, {})
        lat_l = r_l.get("total_lat_ms", float("nan"))
        lat_m = r_m.get("total_lat_ms", float("nan"))
        lat_s = r_s.get("total_lat_ms", float("nan"))
        gpu_area = r_l.get("gpu_area_mm2", 0.0)
        l2_bw = r_l.get("l2_bw_per_cycle", 0)
        print(
            f"  {sm:>4}  {l2:>7.1f}  {l2_bw:>12d}  {gpu_area:>9.1f}"
            f"  {lat_l:>12.2f}  {lat_m:>12.2f}  {lat_s:>12.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


_MEM_FREQ_CHOICES = [6400, 8533, 9600, 10667, 12800]
_MEM_BITWIDTH_CHOICES = [128, 256]

# LPDDR5/LPDDR5x: [6400, 8533, 9600] ↔ [128, 256]
# LPDDR6:         [10667, 12800]      ↔ [128, 256]
_LPDDR5_FREQS = {6400, 8533, 9600}
_LPDDR6_FREQS = {10667, 12800}
_LPDDR5_BITWIDTHS = {128, 256}
_LPDDR6_BITWIDTHS = {128, 256}

_DEFAULT_MEM: Dict[str, Tuple[int, int]] = {
    "Orin": (6400, 256),
    "Thor": (8533, 256),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DSE for ViT+LLM inference: grid-search SM count and L2 cache size "
            "for all three workload sizes (L/M/S) and report configurations that "
            "satisfy the latency constraint for every size simultaneously."
        )
    )
    parser.add_argument(
        "--area",
        type=int,
        required=True,
        choices=[100, 200, 400],
        help="SoC area in mm² (100, 200 or 400).  GPU is budgeted at ≤35%% of this.",
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
            "Robo: prefill-only pipeline (2–4×ViT + LLM prefill).  "
            "AD:   pipeline with speculative decode (6–12×ViT + prefill + 5×decode).  "
            "Each config is evaluated at three sizes: "
            "L (8B LLM, 4-card), M (4B LLM, 2-card), S (1.7B LLM, 1-card)."
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
        default=110.0,
        help="End-to-end latency constraint in ms (default: 110 ms).",
    )
    parser.add_argument(
        "--mem_freq",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help=(
            "Memory frequency in MT/s.  "
            "LPDDR5/5x: 6400 | 8533 | 9600 (bitwidth must be 128 or 256).  "
            "LPDDR6:    10667 | 12800       (bitwidth must be 128 or 256).  "
            "Default: 6400 for Orin, 8533 for Thor."
        ),
    )
    parser.add_argument(
        "--mem_bitwidth",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        default=None,
        help=(
            "Memory bus width in bits.  "
            "LPDDR5/5x: 128 | 256.  LPDDR6: 128 | 256.  "
            "Default: 256 for both Orin and Thor."
        ),
    )
    args = parser.parse_args()

    # ── Apply defaults for mem_freq / mem_bitwidth based on base_hw ────────────
    default_freq, default_bw = _DEFAULT_MEM[args.base_hw]
    mem_freq: int = args.mem_freq if args.mem_freq is not None else default_freq
    mem_bitwidth: int = (
        args.mem_bitwidth if args.mem_bitwidth is not None else default_bw
    )

    # ── Validate mem_freq / mem_bitwidth combination ────────────────────────────
    if mem_freq in _LPDDR5_FREQS and mem_bitwidth not in _LPDDR5_BITWIDTHS:
        parser.error(
            f"mem_freq={mem_freq} (LPDDR5/5x) is only compatible with "
            f"mem_bitwidth in {sorted(_LPDDR5_BITWIDTHS)}, got {mem_bitwidth}."
        )
    if mem_freq in _LPDDR6_FREQS and mem_bitwidth not in _LPDDR6_BITWIDTHS:
        parser.error(
            f"mem_freq={mem_freq} (LPDDR6) is only compatible with "
            f"mem_bitwidth in {sorted(_LPDDR6_BITWIDTHS)}, got {mem_bitwidth}."
        )

    # ── Validate LLM precision for the chosen platform ─────────────────────────
    valid_precisions = LLM_PRECISION_CHOICES[args.base_hw]
    if args.precision not in valid_precisions:
        parser.error(
            f"For {args.base_hw}, --precision must be one of {valid_precisions}. "
            f"Got: {args.precision!r}"
        )

    # ── Guard against wide-bus config on a small SoC ───────────────────────────
    if args.area < 400 and mem_bitwidth in {256}:
        parser.error(
            f"[ERROR] area={args.area} mm² (< 400) is not compatible with "
            f"mem_bitwidth={mem_bitwidth}-bit: not enough shoreline to accommodate PHYs."
        )

    # ── Shared DSE cache (loaded once, reused across all three size searches) ──
    dse_cache = _load_dse_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Grid search for L (4-card), M (2-card), S (1-card) sequentially.
    # After each size, print its own summary.
    # ══════════════════════════════════════════════════════════════════════════
    results_by_size: Dict[str, list] = {}

    for size in ("L", "M", "S"):
        cfg = _INFERENCE_CONFIGS[args.inference_config][size]
        print(
            f"\n\n{'#' * 72}\n"
            f"#  {args.inference_config}-{size}  |  "
            f"{cfg['llm_model']}  degree={cfg['degree']}  |  "
            f"{cfg['num_vits']}×ViT(seq={cfg['vit_seq']})  "
            f"LLM-seq={cfg['llm_seq']}"
            + (f"  decode×{cfg['num_decode_steps']}" if cfg["num_decode_steps"] else "")
            + f"\n{'#' * 72}"
        )

        size_results = run_dse(
            base_hw=args.base_hw,
            soc_area_mm2=float(args.area),
            inference_config=args.inference_config,
            size=size,
            llm_precision=args.precision,
            latency_limit_ms=args.latency_limit,
            mem_freq=mem_freq,
            mem_bitwidth=mem_bitwidth,
            dse_cache=dse_cache,
        )
        results_by_size[size] = size_results

        _print_summary(
            size_results,
            args.inference_config,
            args.latency_limit,
            label=f"{args.inference_config}-{size}  ({cfg['llm_model']}, degree={cfg['degree']})",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Compute intersection: (SM, L2) pairs that pass in ALL three sizes.
    # ══════════════════════════════════════════════════════════════════════════
    passing_keys: set = None  # type: ignore[assignment]
    for size, results in results_by_size.items():
        size_passing = {(r["sm_count"], r["l2_mb"]) for r in results if r["lat_ok"]}
        if passing_keys is None:
            passing_keys = size_passing
        else:
            passing_keys &= size_passing

    _print_final_summary(
        results_by_size,
        passing_keys or set(),
        args.inference_config,
        args.latency_limit,
    )

    return (
        results_by_size,
        passing_keys,
        {
            "base_hw": args.base_hw,
            "inference_config": args.inference_config,
            "precision": args.precision,
            "area": args.area,
            "mem_freq": mem_freq,
            "mem_bitwidth": mem_bitwidth,
            "latency_limit": args.latency_limit,
        },
    )


if __name__ == "__main__":
    main()
