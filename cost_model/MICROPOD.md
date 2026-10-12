# MicroPod MCM product-family cost model

From `/home/sly/LLM/LLMCompassPlus`, using the `llmcompass_ae` Python environment:

```bash
python -B -m cost_model.micropod_cli --config cost_model/examples/micropod_5nm.json --output cost_model/examples/micropod_5nm_result.json
python -B -m unittest discover -s cost_model/tests -v
```

The sibling `cost_model/chiplet-actuary` checkout is required. The CLI accepts a
JSON config and writes a JSON result (or prints JSON if `--output` is omitted).
Configuration errors fail with a nonzero exit code; an area constraint violation
produces `feasible: false`, a reason and the area report, without fabricated costs.
Custom calibration/database paths are relative to the input JSON directory.

## Physical design and example

One identical die is manufactured for S/M/L systems with 1/2/4 dies, respectively.
Every die includes two physical UCIe ports, one PCIe x8 PHY and the same extra
hardware, including the S SKU whose UCIe ports are inactive. M uses two distinct
bidirectional links between its two dies; L uses a four-die ring. Explicit port
endpoints are reported. Counts of SMs, cores, cache and memory width are resource
sums, not claims of linear performance or unified caches/address spaces.

The example uses a Thor-derived 8-SM, 4-core, 16-MB-L2, 128-bit-memory die, 5nm
cost parameters and 100,000 systems per SKU. Demand numbers remain illustrative;
no market forecast is implied. Node selection does not geometrically scale the
area. DRAM chips, PCB, power, cooling and software are outside this cost scope.

`extra_hardware_area_mm2` defaults to zero and specifies TOTAL additional module
area per physical die, before proportional system overhead. It is neither per
port nor per SKU. It is included in the physical area, defect yield and the
upstream area-based NRE. Interconnect controller/adapter/buffer area is excluded
by the requested modeling assumption, not silently charged to this field.

## UCIe selection and area

`interconnect.target_bandwidth_gbps_per_port` is raw SINGLE-DIRECTION bandwidth in
decimal Gbit/s for ONE port. The example is 1024 Gbit/s = 128 GB/s. There is no
bandwidth efficiency factor. The default database is
`icnt_model/configs/UCIE.json`; its path and hash are reported for provenance.
This is a supplied physical-size calibration, not proof of a implemented PHY's
area in any particular process.

Only the `standard` package configurations are considered for organic-substrate
MCM. Selection minimizes PHY area among candidates with bandwidth at least the
target, then bandwidth surplus, then module count, then rate. Targets beyond the
database's maximum are rejected, never rounded down. The existing PHY area
function is reused. Each module's width is placed along the die edge; modules
are assumed adjacent rather than stacked in depth.

For 128 GB/s the selected port has two x16 modules at 32 GT/s, area 1.76022 mm2
and shoreline 1.143 mm. The two ports occupy 3.52044 mm2 before system overhead.

## Area and two-edge constraint

The shared area model now defaults to ONE PCIe x8 group in custom designs. Its
area and edge length use the average of the annotated x8 groups (two on Thor,
three on Orin); these manual rectangles differ in measured width. Reference
reconstruction retains ALL original groups, so calibration closure is unchanged.
There is no independently identified PCIe controller area to remove.

Existing components, PHY and extra hardware receive the calibrated proportional
residual overhead once. Package integration requires:

```
memory PHY length + two UCIe PHY lengths + USB length
  + one PCIe x8 length + HDMI/DP length <= 2 * sqrt(final die area)
```

USB/PCIe/display use horizontal lengths from the calibrated images. The display
measurement is the labelled HDMI/DP block, not an independently measured HDMI
only PHY. CSI and Ethernet remain in area, but are outside this user-specified
two-edge budget. The check is aggregate, not detailed floorplanning or routing.
An invalid die is rejected without inflating its area. S retains the physical
ports and uses the same check as M/L. `io_phy_shoreline` determines validity;
`memory_phy_shoreline` remains a diagnostic of memory alone.

Area API additions: `extra_hardware_area_mm2=0`, `pcie_x8_count=1`, and
`d2d_phy_length_per_port_mm`. The latter is required for package integration with
nonzero ports. `pcie_x8_count=None` restores all measured reference strips.
The corresponding area CLI flags are `--extra-hardware-area-mm2`,
`--pcie-x8-count`, and `--d2d-phy-length-per-port-mm`.

## Cost accounting

Full good dies only; there is no harvesting for this MicroPod design. Use the
upstream negative-binomial yield and gross-dies-per-wafer formulas. Each tier
gets its own appropriately sized OS package; S is a single-die OS package and
M/L use upstream MCM substrate cost multipliers. Package success probability
is `bonding_yield_os ** soc_count`; all required KGD/bumping/package loss terms
come from upstream `OS.cost_RE()`. There is no additional per-link fee, and no
second area multiplier for PHY. The substrate model is a first-order proxy for
ring routing costs, not a validated 128-GB/s physical package design.

For demand q and n dies per system, expected package attempts are q/y_package,
and required good dies are n*q/y_package. Sum these across SKUs before sizing
the common wafer campaign. `whole_wafers: true` rounds up this pooled wafer
count, charging the extra wafer fabrication cost once to the family; surplus
good dies get no sales credit or bumping charge. Results remain expectation
values, not stochastic delivery guarantees. With `whole_wafers: false`, tier
manufacturing costs exactly match upstream OS formulas. With rounding enabled,
per-tier costs retain those formulas and `family.wafer_rounding_cost` separately
reconciles to the total manufacturing bill.

The full physical module design and die design NRE are charged once across the
family. Every demanded SKU adds one distinct package-design NRE; a zero-demand
SKU does not incur a package-design NRE. No arbitrary shared-NRE allocation to
SKUs is reported. Extra hardware and PHY development are represented by upstream
area-based NRE, without additional separately calibrated licensing fees.

These changes also affect future harvesting runs through the shared area model
(one PCIe x8 by default). The harvesting example explicitly overrides this to four PCIe x8 groups, enables
PCIe/memory PHY harvesting, and checks four square-die edges. Its result is regenerated
for that configuration. MicroPod always retains one x8 group and two edges per die.
