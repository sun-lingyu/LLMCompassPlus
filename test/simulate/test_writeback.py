import unittest
from types import SimpleNamespace
from software_model.utils import L2Cache, L2AccessType as T, data_type_dict as D, Tensor
from software_model.quant import L2CacheQuant, Quant
from software_model.rope import L2CacheQKNormRoPE
from software_model.layernorm import L2CacheLayerNorm
from software_model.kv_cache import KVCacheAppend
from software_model.flashattn import L2CacheFlashAttn

def device(size=1<<20):
 return SimpleNamespace(compute_module=SimpleNamespace(l2_size=size, clock_freq=1e9,
    l2_bandwidth_per_cycle=1024, compute_latency_factor={'quant':1e-12}),
    io_module=SimpleNamespace(bandwidth=100e9,bandwidth_efficiency=1))
def access(c,role=T.OUTPUT,col=0,write=None):
 return c._access((0,col),(16,16),role,1,8,512,write=write)
class CacheTests(unittest.TestCase):
 def test_dirty_handoff_eviction_and_drain(self):
  a=L2Cache(1024);a.output_tile_size=512
  self.assertEqual(access(a),0);self.assertEqual(access(a,T.WEIGHT),512)
  b=L2CacheQuant(1024,16,16,D['fp16'],D['fp16'],a)
  self.assertEqual(b.occupied_size,1024)
  self.assertEqual(b.access(T.ACTIVATION,(0,0),(16,16)),0)
  self.assertEqual(b.access(T.OUTPUT,(0,0),(16,16)),0) # evicts old clean weight
  self.assertEqual(b.pending_writeback_bytes,1024)
  self.assertEqual(access(b,T.WEIGHT,col=16),1024) # dirty old input eviction + read
  self.assertEqual(b.writeback_bytes,512)
  self.assertEqual(b.drain(),512);self.assertEqual(b.drain(),0)
  self.assertEqual(a.pending_writeback_bytes,512) # candidate must not mutate predecessor
 def test_output_identity_and_capacity(self):
  a=L2Cache(4096);a.output_tile_size=512;access(a)
  b=L2CacheQuant(4096,16,16,D['fp16'],D['fp16'],a)
  b.access(T.OUTPUT,(0,0),(16,16))
  self.assertEqual(b.occupied_size,1024)
  self.assertNotEqual(a._key(a.Tile(T.OUTPUT,(0,0))),b._key(b.Tile(T.OUTPUT,(0,0))))
 def test_rope_inplace(self):
  a=L2Cache(4096);a.output_tile_size=512
  for col in (0,16,32):access(a,col=col)
  b=L2CacheQKNormRoPE(4096,16,32,D['fp16'],D['fp16'],a)
  b.access(T.ACTIVATION,(0,0),(16,32));b.access(T.OUTPUT,(0,0),(16,32))
  self.assertEqual(b.occupied_size,1536);self.assertEqual(b.pending_writeback_bytes,1536)
  self.assertIn(b.Tile(T.OUTPUT,(0,32)),b.resident_tiles)
 def test_residual_identity(self):
  a=L2CacheLayerNorm(8192,16,16,D['fp16'],D['fp16'],D['fp16'])
  a.access(T.RESIDUAL_OUTPUT,(0,0),(16,16));a.access(T.OUTPUT,(0,0),(16,16))
  q=L2CacheQuant(8192,16,16,D['fp16'],D['fp16'],a)
  q.access(T.OUTPUT,(0,0),(16,16))
  b=L2CacheLayerNorm(8192,16,16,D['fp16'],D['fp16'],D['fp16'],q)
  self.assertEqual(b.access(T.RESIDUAL_INPUT,(0,0),(16,16)),0)
  self.assertEqual(b._role_ids[T.RESIDUAL_INPUT],a._role_ids[T.RESIDUAL_OUTPUT])
 def test_scale_drain_bytes(self):
  c=L2CacheQuant(4096,16,16,D['fp16'],D['fp4'])
  c.access(T.OUTPUT,(0,0),(16,16));c.access(T.OUTPUT_SCALE,(0,0),(16,16))
  self.assertEqual(c.pending_writeback_bytes,128+256);self.assertEqual(c.drain(),384)
 def test_drain_latency_contract(self):
  op=Quant(D['fp16'],D['fp4']);op(Tensor([16,16],D['fp16']))
  latency=op.compile_and_simulate(device(),drain_at_end=True)
  self.assertEqual(op.drain_bytes,384);self.assertEqual(op.drain_cycles,4)
  # Memory-bound: 6 read cycles + 4 drain cycles; no additive L2 penalty.
  self.assertAlmostEqual(latency,10/1e9)
  self.assertEqual(op.l2_status.pending_writeback_bytes,0)
  op.compile_and_simulate(device(),drain_at_end=False)
  self.assertEqual(op.drain_cycles,0);self.assertEqual(op.l2_status.pending_writeback_bytes,384)
 def test_compute_hides_drain_but_keeps_traffic(self):
  d=device();d.compute_module.compute_latency_factor['quant']=1e-9
  op=Quant(D['fp16'],D['fp4']);op(Tensor([16,16],D['fp16']))
  latency=op.compile_and_simulate(d,drain_at_end=True)
  self.assertAlmostEqual(latency,256/1e9)
  self.assertEqual(op.drain_cycles,4)
  self.assertEqual(op.mem_access_size,512+384)
  self.assertEqual(op.cache_metrics['total_dram_bytes'],896)
  self.assertEqual(op.l2_status.pending_writeback_bytes,0)
 def test_l2_hides_drain_without_double_count(self):
  d=device();d.compute_module.l2_bandwidth_per_cycle=1
  op=Quant(D['fp16'],D['fp4']);op(Tensor([16,16],D['fp16']))
  latency=op.compile_and_simulate(d,drain_at_end=True)
  self.assertAlmostEqual(latency,op.l2_status.l2_access_bytes/1e9)
  self.assertGreater(op.l2_status.l2_access_bytes,10)
  self.assertEqual(op.drain_cycles,4)
 def test_inherited_dirty_drain_and_predecessor_unchanged(self):
  d=device();d.compute_module.compute_latency_factor['quant']=1e-9
  first=Quant(D['fp16'],D['fp16']);first(Tensor([16,16],D['fp16']))
  first.compile_and_simulate(d,drain_at_end=False)
  previous=first.l2_status;pending=previous.pending_writeback_bytes
  second=Quant(D['fp16'],D['fp4']);second(Tensor([16,16],D['fp16']))
  second.compile_and_simulate(d,previous,drain_at_end=True)
  self.assertEqual(previous.pending_writeback_bytes,pending)
  self.assertEqual(second.drain_bytes,pending+384)
  self.assertEqual(second.cache_metrics['read_bytes'],0)
  self.assertEqual(second.l2_status.pending_writeback_bytes,0)
 def test_kv_append_handoff(self):
  d=device();a=L2Cache(d.compute_module.l2_size);a.output_tile_size=512
  a._access((0,0),(16,48),T.OUTPUT,1,3,512)
  op=KVCacheAppend(16,32,16,16,D['fp16'])
  op.compile_and_simulate(d,a,drain_at_end=False)
  b=L2CacheFlashAttn(d.compute_module.l2_size,1,1,16,48,16,D['fp16'],D['fp16'],False,1,L2Cache_previous=op.l2_status)
  self.assertEqual(b.access(T.Q,(0,0),(16,16)),0)
  self.assertEqual(b.access(T.K,(32,0),(16,16)),0)
  self.assertEqual(b.access(T.V,(32,0),(16,16)),0)
  self.assertEqual(b.access(T.K,(0,0),(16,16)),512)
 def test_write_through(self):
  a=L2Cache(512,write_policy='write_through')
  self.assertEqual(access(a),512);self.assertEqual(access(a),512)
  self.assertEqual(a.drain(),0)
if __name__=='__main__':unittest.main()
