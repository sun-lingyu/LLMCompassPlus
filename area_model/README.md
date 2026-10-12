## Current custom-SoC interface policy (2026-10-08)

This section supersedes older memory-only constraint and full-PCIe descriptions below.
Custom designs retain one mean annotated PCIe x8 PHY by default; reference reconstruction
retains all physical reference strips. Extra hardware is configured per die using
`extra_hardware_area_mm2` (default 0), before proportional residual overhead.
D2D area counts PHY only; controller/adapter/buffer area is excluded.
Package validity now uses `constraints.io_phy_shoreline`: memory + D2D + USB +
PCIe + HDMI/DP PHY lengths must fit the configured equivalent-square edge budget.
`phy_edge_count` / `--phy-edge-count` accepts 2 (default, MicroPod) or 4 (monolithic). D2D length
per port is required when package integration has physical ports.
The old `memory_phy_shoreline` output is only a memory diagnostic.
See [MicroPod MCM cost usage](../cost_model/MICROPOD.md) for the complete JSON workflow.

# GPU and SoC area models

Areas are in mm². These are architecture-level estimates, not floorplan or
CPU-performance validation. No DSE integration or cost model is included.

## Usage

Run from the repository root:

```bash
python -m area_model.soc_model --reference
python -m area_model.soc_model --sm 8 --l2-mb 16 --memory-bits 128 --cpu-policy all
python -m unittest discover -s area_model/tests -v
```

Output is JSON with effective configuration, disjoint component areas, modeled
subtotal, unidentified overhead, total die area, reference calibration and scope
warnings. Values are unrounded until presentation. Relative calibration paths are
resolved from the module, not the current working directory.

The default custom scenario has no added inter-SoC ports. It is a standalone
SoC-area scenario, **not a complete MicroPod ring die**. For a reusable ring die,
pass `--d2d-ports 2 --d2d-area-per-port <measured_mm2>`. The supplied per-port area
must cover PHY, controller, adapter and buffers. There is deliberately no guessed
link-area default. Count all manufactured ports, even when inactive in a 1/2-die
product. Existing reference PCIe area remains present; added ports are incremental
and must not duplicate a repurposed existing interface. C2C endpoints can use this
same incremental interface-area input, but their implementation must be stated.

Python API:

```python
from area_model.soc_model import calculate_soc_area, calculate_reference_soc_area

result = calculate_soc_area(8, 16, cpu_policy="balanced", memory_bus_bits=128)
reference = calculate_reference_soc_area()
```

The SoC model supports Orin and Thor via `--hw` / `hw=`. Thor remains the default.
The existing GPU function supports both hardware families with its original area formula. Its file loading no longer changes
or depends on cwd, invalid sizes raise ValueError, and results retain precision.
The standalone GPU Thor reference example now uses 22 physical SMs / 32 MB L2.

## Reference and measurement provenance

`configs/Thor_die_mm.json` remains the GPU calibration. Thor has 22 manufactured
SMs but 20 enabled SMs; L2 is 32 MB as confirmed by the user. Physical SMs determine
GPU silicon area. Active SMs determine the CPU scaling ratio. Custom designs default
to all physical SMs enabled; pass `--active-sm` to represent disabled resources.

`die_shot/Thor_soc_annotated.png` is an unmodified copy of the user-supplied image
from 2026-10-07. `configs/Thor_soc.json` records its SHA-256, image size, crop and
each non-GPU measurement rectangle. Measurements are approximate manual readings
of a 673×417 annotated image, not original layout data. Box coordinates are
half-open `[left, top, right, bottom]`. The die crop excludes outer legends. Its
area uses the inherited 26×16 mm dimensions, which are an existing calibration
assumption, not a new independent measurement.

The two CPU image groups contain 14 labelled physical cores. They do NOT establish
hardware cluster topology. The marked L3 region is measured separately; its
capacity is not inferred. GPU SM/L2/GPC/TPC parameters are retained from the original model.
The command front uses the latest user annotation
`die_shot/Thor_command_front_v2_annotated.png` (1692x1054 pixels).
The stepped outline is approximated by four non-overlapping rectangles:
[109,496,741,562], [109,562,510,585], [706,562,1210,670],
[1084,535,1210,562], within die bounds [43,37,1657,1022].
The lower-left step stops at x=510 to avoid overlap with the existing ISP
measurement. This does not use the enclosing bounding box, which would include
unrelated IP. Pixel areas are scaled using the inherited 26x16 mm die.
Coordinates, source hash and the previous measurement are recorded in
`Thor_die_mm.json`; older reference images are retained.
This block identification is the user's hypothesis, not verified layout data.
The earlier image remains the source for non-GPU regions.
SoC residual overhead is automatically recalibrated from the updated GPU sum;
do not add command-front area separately to the non-GPU component list.
The original GPU compute-complex model excludes video/media, so the explicitly
measured video-codec region is counted once.

## Components and residual overhead

The default custom design retains GPU, CPU/private cache, CPU shared L3, ISP,
Imaging Processor, video codec, LPDDR PHY, CSI, PCIe, USB and Ethernet regions.
Display subsystem and the labelled dedicated HDMI/DP strip are retained by default.
Use `--exclude-display` to remove both, or `include_display=False` in the Python API. Display-PHY removal assumes the labelled strip
is dedicated; revise the rectangles if shared logic is identified.

Let R be ALL identified reference component areas, including display, video and the removable DSP/PVA:

```
alpha = (A_reference_die - sum(R)) / sum(R)
A_custom = (1 + alpha) * sum(retained_and_added_components)
```

`alpha` is overhead relative to the identified subtotal, not a fraction of the
final die. Removing display, DLA or DSP/PVA does not recalibrate alpha. Unidentified fabric,
control, clock/reset and physical whitespace outside identified system components
are covered collectively by the residual. They are not independently measured. No additional utilization
factor is applied because measured rectangles already contain local routing space.
The full-reference 416 mm² closure is an accounting identity, **not validation**.
Use `--overhead-ratio` for sensitivity analysis.

LPDDR PHY area scales with bus width in 16-bit increments at the reference
technology. Rate-dependent PHY changes are not modeled. The package-level memory shoreline check is described below.
The memory controller is not identifiable in the supplied image and is currently
covered by residual overhead, rather than claimed as a separate measurement.
The Python API permits an explicit `memory_controller_area_mm2` only alongside a
caller-recalibrated `overhead_ratio` to avoid double-counting that controller.

## CPU scenarios

For Thor, with r = active_SM_count / 20, compute unrounded cores as:

```
14 * (beta + (1 - beta) * r)
```

`proportional`, `balanced`, `fixed` use beta = 0, 0.5, 1 respectively. Scaled
policies round to nearest `cpu_core_quantum` (default 1), with ties rounded upward
and a minimum of one quantum: max(q, floor(requested/q + 0.5)*q). This is not
Python ties-to-even rounding. Fixed policy retains the reference core count at
default granularity; its legacy upward rounding for an explicit quantum is unchanged. The quantum is a
configurable design assumption, NOT a verified Thor cluster organization. CPU core
and private-cache area and the shared-L3 area scale together by actual rounded
core count / 14. L3 is not rounded independently to integer MB; each core retains
its private cache. Cache capacity/banking and CPU interconnect topology are not modeled.

For 8 active SMs, the three scenarios produce 6, 10, 14 cores. They are sensitivity
scenarios, not guaranteed CPU-performance configurations or rigorous area bounds.
The model does not clip CPU counts at 14 when scaling above 20 active SMs. Override
with `--cpu-cores` if needed; explicit counts must match the specified quantum.
`--cpu-policy all` disallows an explicit core override so comparison is meaningful.

Reference reconstruction explicitly selects all 22 manufactured SMs, 20 active
SMs, 14 CPU cores, 32 MB L2, 256-bit memory, display included and no added D2D ports.
`--reference` rejects custom design flags instead of silently ignoring them.

## Validation scope

Tests check the retained GPU formula, CPU rounding, inactive-SM accounting,
reference closure, display/residual separation, memory-width deltas, manufactured
link-area addition, bad inputs, overlapping measurement rectangles, cwd independence
and machine-readable CLI output. They do not establish physical implementation
accuracy, CPU sufficiency, power, interconnect feasibility or cost advantage.


## Package-level memory PHY shoreline constraint

Select `--integration package` (Python: `integration="package"`) to enforce:

```
L_memory_PHY <= 2 * sqrt(A_SoC)
```

A_SoC is the final physical die area, including residual overhead and all added
interfaces. Equality is legal (floating-point tolerance 1e-12). A failing design
returns `valid: false` and a reason in
`constraints.memory_phy_shoreline`; its area is NOT silently increased. The CLI
still emits JSON successfully for an infeasible design; consumers must inspect
`valid`. Malformed inputs produce a CLI error.

Default integration is `board`, preserving the previous area-only use; the
constraint is then not enforced and `passed` is null. Reference reconstruction
also remains board-level. Integration is explicit and is not inferred from the
number of inter-SoC ports.

The reference memory PHY length is the sum of the vertical extents of both
annotated strips, converted with die_height/image_crop_height:
2 * 283 / 392 * 16 = 23.1020 mm for 256 bits. It scales linearly with memory bus
width at fixed PHY technology/depth; 128 bits uses 11.5510 mm. This is a shoreline
length, not PHY area, and is not multiplied by the residual overhead coefficient.
Provide `--memory-phy-length-mm` (Python: `memory_phy_total_length_mm`) to override
the total length for the target configuration directly; it is not scaled again.

The result reports required/available length, margin, and minimum die area
(L/2)^2. This implements only the requested aggregate two-edge square-die
criterion; it does not certify lane placement, individual-edge fit, corner
keepouts or other I/O placement. `valid` refers only to implemented constraints.

Example:
```bash
python -m area_model.soc_model --sm 8 --l2-mb 16 --memory-bits 128 --cpu-policy all --integration package
```


## Orin calibration

```bash
python -m area_model.soc_model --hw Orin --reference
python -m area_model.soc_model --hw Orin --sm 8 --l2-mb 2 --memory-bits 128 --cpu-policy all --integration package
```

Python: `calculate_soc_area(8, 2, hw="Orin")` or
`calculate_reference_soc_area(hw="Orin")`. An explicit JSON config can also
select the hardware; conflicting `hw` and config selections are rejected.
The CLI defaults L2 to half the selected reference capacity (Orin 2 MB, Thor
16 MB). The API still requires the target L2 capacity explicitly.

Orin uses the unchanged GPU calibration: 16 physical/enabled SMs, 4 MB GPU L2,
and 20.17x22.2 mm total die dimensions (447.774 mm²). Its CPU reference contains
12 labelled cores. CPU policies use r=active_SM/16 and
12*(beta+(1-beta)*r), with the same configurable nearest-core half-up rounding assumption.
At 8 SM, proportional/balanced/fixed therefore give 6/9/12 cores.

`configs/Orin_soc.json` now uses the user's complete IP envelopes from
`die_shot/Orin_soc_regions_annotated.png` (1128x1022). The previous measurement
is retained in `measurement_history`. Every current region has coordinates and
source provenance. Die bounds are [5,5,1122,1017]; image width is 22.2 mm and
height is 20.17 mm (rotated relative to legacy dimension names).

The CPU envelope is tiled by core/support bands and the shared-L3 band; both
scale with CPU count. The 4 MB system SLC is separate and fixed. Full DLA and
ISP/VIC envelopes are measured. Only the dedicated PVA envelope is separately
measured, using [381,1373,1008,1767] on the unobscured original
`Orin_die_annotated.png`, registered to the current image crop. It includes VPU0,
VPU1, VPU Memory 768KB and local logic. Adjacent Security Engine, Safety Island,
system SRAM, external R5 regions and separately labelled DMA form fixed
`system_components`, measured as the previous broad PVA/system envelope minus
the dedicated PVA. These system functions do not scale with GPU, CPU or memory
width. The broad envelope is preserved in history for provenance, not added again. Dedicated PVA boundaries remain manual
die-shot estimates, not precise layout measurements.

The stepped GPU envelope contains the original parameterized compute complex.
Its reference area minus the compute-complex area is recorded as
`gpu_region_other_fixed`. This remains fixed when resizing SMs/L2, a first-order
assumption for unparameterized media/support logic. The complete GPU envelope is
NOT added on top of the compute-complex area; multimedia is not added separately.
The former broad PVA/system box is no longer counted; GPU area is unchanged. CSI PHY
is subtracted from the ISP/VIC envelope where they overlap. Existing PHY boxes
are registered from the earlier die crop, preserving their physical area/length.
Only the remaining uncovered area enters global proportional residual overhead.

DLA is included in full-reference calibration but excluded from custom SoCs by
default. Dedicated PVA is also excluded by default. `--include-dla` / `include_dla=True` can restore
DLA for accounting comparisons; `--exclude-dla` makes its removal explicit.
Removing DLA does not recompute the reference residual ratio.

Codec/display control cannot be independently separated. They are retained in
GPU-region support and remaining system logic rather than claimed as individual measurements. Display I/O PHY is
measured separately. Orin rejects `--exclude-display` because the full display
subsystem cannot be removed accurately with this calibration.

Memory PHY consists of four labelled 64-bit LPDDR5 regions, two horizontal and
two vertical. Total length sums each strip's physical long-edge direction, then
scales by memory width; the same package two-edge square-die constraint applies.
The reference memory controller remains in residual overhead. Inter-SoC link
area still requires an explicit input.

All rectangles and residuals remain first-order estimates. Reference closure is
an accounting identity, not independent accuracy validation.

## Custom designs without a vision accelerator

Custom Thor excludes the labelled DSP region; custom Orin excludes only the
dedicated PVA region. Internal accelerator memory belongs to that removable IP;
identified system support remains in fixed `system_components`; unclassified support remains in residual. Reference calibration and `--reference`
retain both accelerators. The reference residual coefficient is computed before
removal and is not recalibrated for a custom design.

Use `include_vision_accelerator=True` / `--include-vision-accelerator` for an
accounting comparison; default is false. `--exclude-vision-accelerator` makes
removal explicit. Result components retain the `dsp` or `pva` key with zero area
when excluded, while calibration reports its full reference area. The die-area
delta at unchanged other settings is A_accelerator * (1 + alpha), including the
proportional overhead convention. Thor DSP identification is assumed from the
user annotation and is not a verified physical equivalence to Orin PVA.

This GPU-oriented VLA design assumes ISP/VIC and CPU/GPU handle image input and
preprocessing. Accelerator removal is not evidence of zero preprocessing latency
or energy. No preprocessing performance claim is added by the area model.

## Fixed system components

Both hardware models retain `system_components` at their full measured reference
area on every custom die, independently of SMs, CPU cores, L2 and memory width.
Orin uses the previous broad PVA/system envelope minus dedicated PVA, counting
security, safety, system SRAM, DMA and surrounding support together. PVA-local
VPU memory remains exclusively in removable PVA. Thor uses the inner red outline
[132,514,615,576] on `Thor_system_components_annotated.png` (1055x650), with die
bounds [22,16,1033,633], registered to the existing measurement crop. Inner edges
avoid annotation strokes and overlap with DSP. Other Thor measurements are unchanged.

Calibration includes this explicit system area in the identified reference sum,
so it is removed from unidentified residual before computing alpha. Custom area
still follows (1+alpha)*sum(components): the system-component entry stays fixed,
and its proportional overhead contribution is also fixed for a given calibration.
These are aggregate die-shot estimates, not separately verified internal blocks.
