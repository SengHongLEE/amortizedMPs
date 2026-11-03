"""The diagnostic gate must release on support or timeout and reset locally."""
import importlib.util
from pathlib import Path
import unittest
import torch
from test_equilibrium_dynamics import module, Config
from test_adaptive_equilibrium import timing_config

PATH = Path(__file__).resolve().parents[1] / 'legged_gym/scripts/contact_phase_gate.py'
spec = importlib.util.spec_from_file_location('contact_phase_gate', PATH)
gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_module)


class GateTests(unittest.TestCase):
    def make(self, wait):
        obj = module().AdaptiveEquilibriumDynamics(
            timing_cfg=timing_config(), equilibrium_cfg=Config.control.equilibrium,
            num_envs=2, device='cpu', nominal_clearance=.15, max_clearance=.2)
        gate = gate_module.ContactPhaseGate(obj, wait, .025)
        return obj, gate

    def test_disabled_gate_preserves_controller(self):
        old, old_gate = self.make(0)
        old._kuramoto_reflex_substep = old_gate.original_step
        new, gate = self.make(0)
        a = torch.zeros(2, 19)
        a[:, 0] = 2
        for _ in range(100):
            x = torch.stack(old.intrinsic_dynamics(a, torch.zeros(2, 3)), -1)
            y = torch.stack(new.intrinsic_dynamics(a, torch.zeros(2, 3)), -1)
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertFalse(gate.waiting.any())

    def test_wait_releases_on_support_and_deadline(self):
        obj, gate = self.make(.08)
        obj._contact_available = True
        beta = obj.selected_duty_factors(obj._gait_ids)
        obj.X[:, 2] = torch.pi * (2 - 1.5 * beta) - .001
        gate.step(.001)
        self.assertTrue(gate.waiting.all())
        obj._contact[0] = True
        for _ in range(30):
            gate.step(.001)
        self.assertFalse(gate.waiting[0].any())
        self.assertTrue(gate.waiting[1].all())
        for _ in range(55):
            gate.step(.001)
        self.assertFalse(gate.waiting[1].any())
        self.assertTrue(torch.isfinite(obj.X).all())

    def test_reset_is_partial(self):
        obj, gate = self.make(.08)
        gate.waiting[:] = True
        gate.elapsed[:] = .03
        gate.support[:] = .01
        obj.reset(torch.tensor([0]))
        self.assertFalse(gate.waiting[0].any())
        self.assertTrue(gate.waiting[1].all())
        self.assertEqual(gate.elapsed[0].sum().item(), 0.)
        torch.testing.assert_close(gate.elapsed[1], torch.full((4,), .03))


if __name__ == '__main__':
    unittest.main()
