import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dse.dse import (
    GPU_AREA_FRACTION,
    _check_area_constraint,
    _load_dse_cache,
    compute_inference_latency,
)

BASE_HW = "Thor"
SM_COUNT = 32
L2_MB = 64.0

dse_cache = _load_dse_cache()
result = compute_inference_latency(
    inference_config="AD",
    size="L",
    base_hw=BASE_HW,
    sm_count=SM_COUNT,
    l2_mb=L2_MB,
    mem_freq=8533,
    mem_bitwidth=512,
    llm_precision="fp4",
    dse_cache=dse_cache,
)

# soc_area_mm2=inf skips the area-budget check and just returns the GPU die area.
_, gpu_area = _check_area_constraint(
    base_hw=BASE_HW, sm_count=SM_COUNT, l2_mb=L2_MB, soc_area_mm2=float("inf")
)
required_soc_area = gpu_area / GPU_AREA_FRACTION

print("AD-L | Thor SM=32 L2=64MB mem=8533MT/s x512bit fp4")
print(f"  GPU Area              : {gpu_area:.2f} mm^2")
print(
    f"  Required SoC Area (>=): {required_soc_area:.2f} mm^2 "
    f"(GPU_AREA_FRACTION={GPU_AREA_FRACTION:.0%})"
)
print(f"  Latency                : {result['total_lat_ms']:.2f} ms")
print(f"  Average Power          : {result['avg_power_w']:.2f} W")
print(f"  Peak Power             : {result['peak_power_w']:.2f} W")
print("  Latency Breakdown      :", result["breakdown"])
