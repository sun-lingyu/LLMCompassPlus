"""Standalone, cache-aware Ring AllGather and ReduceScatter.

Only rank 0 is simulated: hardware, shard sizes and initial cache states are
assumed symmetric across ranks. The layer driver controls execution order
and passes the complete cache state between communication and computation.

All bandwidths are bytes/second and all latencies are seconds. ``link_bandwidth``
is the one-way bandwidth of ONE physical link. The agreed aggregate model uses
X for TP=2 and 2X for TP=4, with P-1 dependent rounds in either case.

Example (AG input is the LOCAL shard; RS input is the FULL partial tensor)::

    # Defaults: T_startup=15e-6 seconds, bandwidth_efficiency=0.6.
    ag = Comm("ag", data_type_dict["fp16"], 4, 64e9)
    gathered = ag(Tensor([256, 4096], data_type_dict["fp16"]))
    latency = ag.compile_and_simulate(device, L2Cache_previous=previous_cache)
    # Pass ag.l2_status to the next operator's L2Cache_previous argument.

The cache model uses the same padded 16x16 entries as Matmul. M-shard boundaries
must be 16-row aligned so the in-place output views can be inherited without
copying or splitting physical cache entries. N tails use padded cache entries,
but network traffic counts logical payload bytes only; packed dtype rows must
contain a whole number of bytes. FP4 AG includes FP8
block scales; FP4 RS requires dequantization before this operator.
"""

from math import ceil, isfinite
from numbers import Integral

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, L2AccessType, L2Cache, Tensor, data_type_dict


def _positive_finite(name, value):
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


class L2CacheComm(L2Cache):
    """In-place full-tensor storage, with a separately exposed output view.

ACTIVATION reads the local communication tensor (including partials received
in earlier rounds); OUTPUT writes that SAME physical storage. The complete
previous physical cache, including unrelated dirty entries and LRU order, is
retained. Only the previous output's logical aliases become the new input.
"""

    def __init__(
        self,
        l2_size: int,
        M: int,
        N: int,
        input_rows: int,
        data_type: DataType,
        L2Cache_previous: L2Cache = None,
    ):
        policy = (
            L2Cache_previous.write_policy
            if L2Cache_previous is not None else "write_back"
        )
        super().__init__(l2_size, policy)
        self.M, self.N = M, N
        self.m_tiles = ceil(M / self.TILE_LENGTH)
        self.n_tiles = ceil(N / self.TILE_LENGTH)
        self.input_dtype = self.output_dtype = data_type
        self.input_tile_size = self.output_tile_size = (
            data_type.word_size * self.TILE_LENGTH**2
        )
        self.scale_columns = (
            ceil(N / data_type.scale_block_size) if data_type.name == "fp4" else 0
        )
        self.scale_tiles = ceil(self.scale_columns / self.TILE_LENGTH)
        self.scale_tile_size = data_type_dict["fp8"].word_size * self.TILE_LENGTH**2
        self.regions = {
            L2AccessType.ACTIVATION: (self.m_tiles, self.n_tiles, self.input_tile_size),
            L2AccessType.OUTPUT: (self.m_tiles, self.n_tiles, self.output_tile_size),
            L2AccessType.ACTIVATION_SCALE: (
                self.m_tiles, self.scale_tiles, self.scale_tile_size
            ),
            L2AccessType.OUTPUT_SCALE: (
                self.m_tiles, self.scale_tiles, self.scale_tile_size
            ),
        }
        largest_entry = max(
            self.input_tile_size, self.scale_tile_size if self.scale_tiles else 0
        )
        if l2_size < largest_entry:
            raise ValueError("L2 must hold at least one cache entry")
        if L2Cache_previous is not None:
            if L2Cache_previous.l2_size != l2_size:
                raise ValueError("Cannot inherit cache state with a different L2 capacity")
            previous_size = L2Cache_previous.output_tile_size
            if previous_size is not None and previous_size != self.input_tile_size:
                raise ValueError("Previous output dtype/entry size differs from comm input")

        def input_view(tile):
            role = {
                L2AccessType.OUTPUT: L2AccessType.ACTIVATION,
                L2AccessType.OUTPUT_SCALE: L2AccessType.ACTIVATION_SCALE,
            }.get(tile.access_type)
            if role is None:
                return None
            _, columns, _ = self.regions[role]
            row, column = tile.coord_tuple
            if 0 <= row < input_rows and 0 <= column < columns * self.TILE_LENGTH:
                return self.Tile(role, tile.coord_tuple)
            return None

        self.inherit(L2Cache_previous, mapper=input_view)
        # AG assumes its local input already occupies shard 0 of the output
        # allocation. RS overwrites local partials in place. These are explicit
        # same-address layout assumptions, not a free out-of-place copy.
        self.alias_role(L2AccessType.OUTPUT, L2AccessType.ACTIVATION)
        self.alias_role(L2AccessType.OUTPUT_SCALE, L2AccessType.ACTIVATION_SCALE)

    def access(self, access_type, coord_tuple, scope_tuple):
        height, width, tile_size = self.regions[access_type]
        return self._access(
            coord_tuple, scope_tuple, access_type, height, width, tile_size
        )

    def expose_output(self, rows: int):
        """Export only the final output's aliases, without changing residency.

        Our RS ring numbering ends with shard 0 at rank 0, so its output view
        needs no row offset. Non-output partials remain physically cached and
        dirty, but cannot be mistaken for the next operator's input.
        """
        for output_role, input_role in (
            (L2AccessType.OUTPUT, L2AccessType.ACTIVATION),
            (L2AccessType.OUTPUT_SCALE, L2AccessType.ACTIVATION_SCALE),
        ):
            for tile in list(self._aliases):
                if tile.access_type == output_role:
                    del self._aliases[tile]
            _, columns, tile_size = self.regions[input_role]
            self.regions[output_role] = (ceil(rows / self.TILE_LENGTH), columns, tile_size)
            for row in range(0, rows, self.TILE_LENGTH):
                for column in range(0, columns * self.TILE_LENGTH, self.TILE_LENGTH):
                    coord = (row, column)
                    self._aliases[self.Tile(output_role, coord)] = self._key(
                        self.Tile(input_role, coord)
                    )
        self.output_shape = (rows, self.N)


class Comm(Operator):
    """Independent AG/RS operator for an M-sharded two-dimensional tensor.

    ``primitive``: "ag" or "rs". ``tp_degree``: 2 or 4.
    ``T_startup``: effective fixed collective overhead, paid once (default
        15e-6 seconds). This is the single fitted intercept, not necessarily
        pure kernel launch latency.
    ``bandwidth_efficiency``: optional link efficiency, applied once (default 0.6).

    A round costs max(shard_bytes / aggregate_link_bandwidth,
    L2_bytes / L2_bandwidth, DRAM_bytes / effective_DRAM_bandwidth). Rounds are
    sequential; the max approximates effective pipelining *within* a round.
    Reduction arithmetic is assumed hidden; its reads/writes are still charged.
    No queues, packet timing, GEMM waves or compute/communication overlap.
    """

    def __init__(
        self,
        primitive: str,
        data_type: DataType,
        tp_degree: int,
        link_bandwidth: float,
        *,
        T_startup: float = 15e-6,
        bandwidth_efficiency: float = 0.6,
    ):
        super().__init__()
        if not isinstance(primitive, str) or primitive.lower() not in ("ag", "rs"):
            raise ValueError("primitive must be 'ag' or 'rs'")
        if not isinstance(tp_degree, Integral) or tp_degree not in (2, 4):
            raise ValueError("Only TP=2 and TP=4 are supported")
        _positive_finite("link_bandwidth (bytes/s)", link_bandwidth)
        _positive_finite("bandwidth_efficiency", bandwidth_efficiency)
        if bandwidth_efficiency > 1:
            raise ValueError("bandwidth_efficiency must not exceed 1")
        if not isfinite(T_startup) or T_startup < 0:
            raise ValueError("T_startup must be finite and nonnegative (seconds)")
        _positive_finite("data_type.word_size", data_type.word_size)
        if data_type.name == "fp4":
            if primitive.lower() == "rs":
                raise ValueError("FP4 RS requires dequantization before reduction")
            if (
                not isinstance(data_type.scale_block_size, Integral)
                or data_type.scale_block_size <= 0
            ):
                raise ValueError("FP4 AG requires a positive integer scale block size")
        self.primitive = primitive.lower()
        self.operator_name = "allgather" if self.primitive == "ag" else "reducescatter"
        self.data_type = self.input_dtype = self.output_dtype = data_type
        self.tp_degree = int(tp_degree)
        self.link_bandwidth = link_bandwidth
        self.bandwidth_efficiency = bandwidth_efficiency
        self.T_startup = T_startup
        self.round_count = self.tp_degree - 1

    @property
    def network_bandwidth(self):
        # Aggregate throughput only: no assignment of data to physical links.
        return self.link_bandwidth * self.bandwidth_efficiency * (
            2 if self.tp_degree == 4 else 1
        )

    def __call__(self, input: Tensor) -> Tensor:
        if input.dtype != self.data_type:
            raise ValueError("Input dtype must match the communication dtype")
        if len(input.shape) != 2 or any(
            not isinstance(dim, Integral) or dim <= 0 for dim in input.shape
        ):
            raise ValueError("Comm expects a positive two-dimensional [M, N] tensor")
        input_rows, self.N = map(int, input.shape)
        if not float(self.N * self.data_type.word_size).is_integer():
            raise ValueError("Packed dtype rows must contain a whole number of bytes")
        if self.primitive == "ag":
            self.shard_rows = input_rows
            self.M = input_rows * self.tp_degree
            output_rows = self.M
        else:
            if input_rows % self.tp_degree:
                raise ValueError("RS M must be divisible by tp_degree")
            self.M = input_rows
            self.shard_rows = self.M // self.tp_degree
            output_rows = self.shard_rows
        if self.shard_rows % L2Cache.TILE_LENGTH:
            raise ValueError(
                "M/tp_degree (AG local M) must be a multiple of 16 for "
                "zero-copy cache inheritance; unaligned shard views are not modeled"
            )
        self.input_shape = (input_rows, self.N)
        self.output_shape = (output_rows, self.N)
        self.shard_data_bytes = self.shard_rows * self.N * self.data_type.word_size
        self.shard_scale_bytes = (
            self.shard_rows * ceil(self.N / self.data_type.scale_block_size)
            * data_type_dict["fp8"].word_size
            if self.data_type.name == "fp4" else 0
        )
        self.shard_bytes = self.shard_data_bytes + self.shard_scale_bytes
        self.tensor_bytes = self.tp_degree * self.shard_bytes
        self.tx_bytes = self.rx_bytes = self.round_count * self.shard_bytes
        self.io_size = (2 if self.primitive == "ag" else 3) * self.tx_bytes
        self.mem_access_size = -1
        self.round_stats = []
        return Tensor(list(self.output_shape), self.data_type)

    def _iter_shard_entries(self):
        """A traffic-counting traversal, not a timed small-chunk pipeline.

        Send and receive accesses are interleaved per cache entry. FP4 data and
        scale planes are visited per tile row; a scale entry is charged once,
        not once per data tile sharing it.
        """
        step = L2Cache.TILE_LENGTH
        for row in range(0, self.shard_rows, step):
            for column in range(0, self.l2_status.n_tiles * step, step):
                yield L2AccessType.ACTIVATION, L2AccessType.OUTPUT, row, column
            for column in range(0, self.l2_status.scale_tiles * step, step):
                yield L2AccessType.ACTIVATION_SCALE, L2AccessType.OUTPUT_SCALE, row, column

    def compile_and_simulate(
        self,
        pcb_module: Device,
        L2Cache_previous: L2Cache = None,
        *,
        drain_at_end: bool = False,
    ):
        """Return seconds, including T_startup; retain cache state by default.

        An explicitly requested drain adds final persistence time using the
        base Operator convention. It is not performed at normal op boundaries.
        ``round_stats`` records per-round traffic and service times; counters
        describe this invocation only, even when the physical cache is warm.
        """
        if not hasattr(self, "input_shape"):
            raise RuntimeError("Call the operator with an input Tensor before simulation")
        frequency = pcb_module.compute_module.clock_freq
        l2_bandwidth = pcb_module.compute_module.l2_bandwidth_per_cycle * frequency
        dram_bandwidth = (
            pcb_module.io_module.bandwidth * pcb_module.io_module.bandwidth_efficiency
        )
        for name, value in (
            ("clock_freq", frequency),
            ("L2 bandwidth", l2_bandwidth),
            ("effective DRAM bandwidth", dram_bandwidth),
        ):
            _positive_finite(name, value)
        self.l2_status = L2CacheComm(
            pcb_module.compute_module.l2_size,
            self.M,
            self.N,
            self.input_shape[0],
            self.data_type,
            L2Cache_previous=L2Cache_previous,
        )
        cache = self.l2_status
        self.round_stats = []
        self.latency = self.T_startup
        step = L2Cache.TILE_LENGTH
        for k in range(self.round_count):
            # Logical ring: pull from rank P-1, expose data to rank 1. For RS
            # this numbering makes rank 0 own shard 0 after the final round.
            if self.primitive == "ag":
                send_shard = (-k) % self.tp_degree
                recv_shard = (-k - 1) % self.tp_degree
            else:
                send_shard = (-k - 1) % self.tp_degree
                recv_shard = (-k - 2) % self.tp_degree
            before = cache.metrics()
            l2_before = cache.l2_access_bytes
            for read_role, write_role, row, column in self._iter_shard_entries():
                send_coord = (send_shard * self.shard_rows + row, column)
                recv_coord = (recv_shard * self.shard_rows + row, column)
                # Source-side service for the neighbor's peer load. Incoming
                # peer loads are served by a symmetric remote cache, not a
                # second local read or a local receive-buffer write.
                cache.access(read_role, send_coord, (step, step))
                if self.primitive == "rs":
                    cache.access(read_role, recv_coord, (step, step))
                cache.access(write_role, recv_coord, (step, step))
            after = cache.metrics()
            l2_bytes = cache.l2_access_bytes - l2_before
            dram_bytes = after["dram_bytes"] - before["dram_bytes"]
            network_latency = self.shard_bytes / self.network_bandwidth
            l2_latency = l2_bytes / l2_bandwidth
            dram_latency = dram_bytes / dram_bandwidth
            latency = max(network_latency, l2_latency, dram_latency)
            self.round_stats.append(dict(
                round_index=k,
                send_shard=send_shard,
                recv_shard=recv_shard,
                depends_on_round=k - 1 if k else None,
                tx_bytes=self.shard_bytes,
                rx_bytes=self.shard_bytes,
                l2_bytes=l2_bytes,
                dram_read_bytes=after["read_bytes"] - before["read_bytes"],
                dram_writeback_bytes=(
                    after["evicted_writeback_bytes"] - before["evicted_writeback_bytes"]
                ),
                dram_write_through_bytes=(
                    after["write_through_bytes"] - before["write_through_bytes"]
                ),
                dram_bytes=dram_bytes,
                network_latency=network_latency,
                l2_latency=l2_latency,
                dram_latency=dram_latency,
                start_time=self.latency,
                latency=latency,
                end_time=self.latency + latency,
            ))
            self.latency += latency
        cache.expose_output(self.output_shape[0])
        self.l2_access_size = cache.l2_access_bytes
        result = self.finish_cache_accounting(pcb_module, drain_at_end)
        self.cache_metrics["l2_access_bytes"] = self.l2_access_size
        return result
