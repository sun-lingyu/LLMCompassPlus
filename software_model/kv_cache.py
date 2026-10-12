"""Explicit decode KV append: retain dirty cache-tail writes across attention."""
from math import ceil
from software_model.operators import Operator
from software_model.utils import L2Cache, L2AccessType as T

class KVCacheAppend(Operator):
    def __init__(self, rows, history, q_width, kv_width, dtype):
        super().__init__()
        self.rows, self.history, self.q_width, self.kv_width = rows, history, q_width, kv_width
        self.dtype = dtype

    def compile_and_simulate(self, pcb_module, L2Cache_previous=None, *, drain_at_end=False):
        c = L2Cache(pcb_module.compute_module.l2_size)
        c.output_tile_size = self.dtype.word_size * 256
        c.inherit(L2Cache_previous, lambda tile: tile if tile.access_type == T.OUTPUT else None)
        c.kv_appended = True
        rows, hist, qw, kw = self.rows, self.history, self.q_width, self.kv_width
        first_row = hist // 16 * 16
        dest_rows = ceil((hist % 16 + rows) / 16) * 16
        for role, offset in [(T.K, qw), (T.V, qw + kw)]:
            # Packed QKV source is a read, even though its preceding role was OUTPUT.
            c._access((0, offset), (ceil(rows/16)*16, kw), T.OUTPUT,
                      ceil(rows/16), (qw+2*kw)//16, c.output_tile_size, write=False)
            if hist % 16:
                c._access((first_row, 0), (16, kw), role, ceil((hist+rows)/16), kw//16, c.output_tile_size, write=False)
            c._access((first_row, 0), (dest_rows, kw), role,
                      ceil((hist+rows)/16), kw//16, c.output_tile_size, write=True)
        self.l2_status = c
        self.mem_access_size = c.total_mem_access_size
        cycles = max(self.get_io_cycle_count(self.mem_access_size, pcb_module),
                     c.l2_access_bytes / pcb_module.compute_module.l2_bandwidth_per_cycle)
        self.latency = cycles / pcb_module.compute_module.clock_freq
        return self.finish_cache_accounting(pcb_module, drain_at_end)
