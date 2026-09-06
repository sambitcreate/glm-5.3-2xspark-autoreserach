"""Policy-shape tests only; these do not qualify GPU performance."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('prefill_dispatch', Path(__file__).resolve().parents[1] / 'campaign/prefill_dispatch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DispatchPolicyTests(unittest.TestCase):
    def test_gate_up_threshold(self):
        self.assertFalse(module.use_minblocks2(512,4096,2048))
        self.assertFalse(module.use_minblocks2(1023,4096,2048))
        self.assertTrue(module.use_minblocks2(1024,4096,2048))

    def test_down_threshold(self):
        self.assertFalse(module.use_minblocks2(1024,1024,4096))
        self.assertFalse(module.use_minblocks2(2047,1024,4096))
        self.assertTrue(module.use_minblocks2(2048,1024,4096))

    def test_unknown_shapes_keep_baseline(self):
        for k,n in ((256,256),(4096,4096),(2048,4096),(1024,2048)):
            self.assertFalse(module.use_minblocks2(7168,k,n))


if __name__ == '__main__':
    unittest.main()
