import unittest
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / 'legged_gym/utils/signed_scan.py'
spec = importlib.util.spec_from_file_location('signed_scan', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scan_bin_ids = module.scan_bin_ids
signed_scan_observations = module.signed_scan_observations


class SignedDenseScanTest(unittest.TestCase):
    def test_positive_hurdle_and_negative_gap_survive_pooling(self):
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        y = torch.tensor([-0.1, 0.0, 0.1])
        ids = scan_bin_ids(x, y, 2, 1)
        low = torch.zeros(1, 12)
        high = torch.zeros(1, 12)
        high[0, 4] = 0.1   # A narrow positive hurdle.
        low[0, 7] = -0.6   # A narrow negative gap.
        base = torch.tensor([[-0.18]])
        scan = signed_scan_observations(low, high, base, ids, 2, 1, 5.0)
        self.assertEqual(tuple(scan.shape), (1, 4))
        self.assertAlmostEqual(scan[0, 0].item(), -1.4, places=5)
        self.assertAlmostEqual(scan[0, 3].item(), 2.1, places=5)

    def test_dense_layout_populates_every_compact_bin(self):
        x = torch.arange(1, 34) * 0.1
        y = torch.arange(-3, 4) * 0.1
        ids = scan_bin_ids(x, y, 16, 5)
        self.assertEqual(len(ids), 231)
        self.assertEqual(len(torch.unique(ids)), 80)


if __name__ == '__main__':
    unittest.main()
