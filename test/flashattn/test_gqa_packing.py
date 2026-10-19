import unittest
from unittest.mock import patch
from hardware_model.device import device_dict
from software_model.flashattn import FlashAttn
from software_model.utils import Tensor,data_type_dict

class ThorGQAPackingTest(unittest.TestCase):
    def candidates(self,heads):
        dt=data_type_dict
        op=FlashAttn(dt['fp16'],dt['fp32'],dt['fp16'],True,True,1,'Thor')
        op(Tensor([128,heads,128],dt['fp16']),Tensor([128,8,128],dt['fp16']),Tensor([128,8,128],dt['fp16']))
        seen=set()
        def simulate(mapping,pcb,cache):
            seen.add(mapping.gqa_packing_size)
            return 100
        with patch.object(op,'simulate',side_effect=simulate):
            op.compile_and_simulate(device_dict['Thor'],drain_at_end=True)
        return seen
    def test_qwen14b_group_five_falls_back(self):
        self.assertEqual(self.candidates(40),{1})
    def test_supported_group_four_keeps_packing(self):
        self.assertEqual(self.candidates(32),{1,4})

if __name__=='__main__':unittest.main()
