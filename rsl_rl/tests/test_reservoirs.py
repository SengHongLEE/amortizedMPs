import math
import os
import sys
import unittest

import torch


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

import rsl_rl.modules as public_modules
from rsl_rl.modules.reservoirs import AnalogReservoir, LIFReservoir


class PublicReservoirAPITest(unittest.TestCase):
    def test_exports_supported_reservoir_types(self):
        expected_exports = {
            "AnalogReservoir": AnalogReservoir,
            "LIFReservoir": LIFReservoir,
        }

        for name, expected_type in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(getattr(public_modules, name), expected_type)


class AnalogReservoirTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.reservoir = AnalogReservoir(
            input_dim=5,
            reservoir_dim=8,
            connectivity=1.0,
            activation="relu",
        )

    def test_relu_update_is_nonnegative_and_exposes_state_as_features(self):
        observations = torch.randn(3, 5, requires_grad=True)
        initial_state = torch.zeros(3, 8, requires_grad=True)
        next_state = self.reservoir.update_state(
            observations,
            initial_state,
        )

        self.assertEqual(self.reservoir.state_dim, 8)
        self.assertEqual(self.reservoir.feature_dim, 8)
        self.assertEqual(next_state.shape, (3, 8))
        self.assertTrue((next_state >= 0.0).all())
        self.assertIs(
            self.reservoir.features_from_state(next_state),
            next_state,
        )
        self.assertFalse(next_state.requires_grad)

    def test_default_sparse_initialization_succeeds_for_multiple_seeds(self):
        for seed in range(32):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                reservoir = AnalogReservoir(
                    input_dim=3,
                    reservoir_dim=2,
                )
                radius = torch.linalg.eigvals(
                    reservoir.w_res.to(torch.float64)
                ).abs().max()
                self.assertAlmostEqual(radius.item(), 0.9, places=5)

    def test_max_float32_spectral_target_succeeds_across_seeds(self):
        target_radius = torch.finfo(torch.float32).max

        for seed in (0, 4, 16, 17):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                try:
                    reservoir = AnalogReservoir(
                        input_dim=3,
                        reservoir_dim=2,
                        connectivity=0.1,
                        spectral_radius=target_radius,
                    )
                except ValueError as error:
                    self.fail(f"Valid spectral target was rejected: {error}")

                actual_radius = torch.linalg.eigvals(
                    reservoir.w_res.to(torch.float64)
                ).abs().max()
                self.assertTrue(torch.isclose(
                    actual_radius,
                    torch.tensor(target_radius, dtype=torch.float64),
                    rtol=1e-4,
                    atol=0.0,
                ))

    def test_fallback_handles_min_connectivity_with_small_target(self):
        smallest_positive_float32 = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ).item()
        try:
            reservoir = AnalogReservoir(
                input_dim=3,
                reservoir_dim=2,
                connectivity=smallest_positive_float32,
                spectral_radius=1e-20,
            )
        except ValueError as error:
            self.fail(f"Valid fallback configuration was rejected: {error}")

        actual_radius = torch.linalg.eigvals(
            reservoir.w_res.to(torch.float64)
        ).abs().max()
        self.assertTrue(torch.isclose(
            actual_radius,
            torch.tensor(1e-20, dtype=torch.float64),
            rtol=1e-4,
            atol=0.0,
        ))

    def test_rejects_single_unit_reservoir(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            AnalogReservoir(input_dim=3, reservoir_dim=1)

    def test_fixed_random_weights_and_bias_are_buffers(self):
        parameter_names = dict(self.reservoir.named_parameters())
        buffer_names = dict(self.reservoir.named_buffers())

        for name in ("w_in", "w_res", "reservoir_bias"):
            self.assertNotIn(name, parameter_names)
            self.assertIn(name, buffer_names)

    def test_recurrent_matrix_has_zero_diagonal_and_requested_radius(self):
        reservoir = AnalogReservoir(
            input_dim=3,
            reservoir_dim=12,
            connectivity=1.0,
            spectral_radius=0.9,
        )

        self.assertTrue(torch.equal(
            reservoir.w_res.diagonal(),
            torch.zeros(12),
        ))
        radius = torch.linalg.eigvals(
            reservoir.w_res.to(torch.float64)
        ).abs().max()
        self.assertTrue(torch.isclose(
            radius,
            torch.tensor(0.9, dtype=torch.float64),
            rtol=1e-4,
            atol=0.0,
        ))

    def test_rejects_spectral_target_that_underflows_scaled_matrix(self):
        smallest_positive_float32 = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ).item()
        matrix = torch.tensor([
            [0.0, 2.0],
            [0.5, 0.0],
        ])

        with self.assertRaisesRegex(
            ValueError,
            "too small|unrepresentable",
        ):
            AnalogReservoir._scale_spectral_radius(
                matrix,
                target_radius=smallest_positive_float32,
            )

    def test_rejects_spectral_target_that_cannot_be_achieved_accurately(self):
        smallest_positive_float32 = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ).item()
        matrix = torch.tensor([
            [0.0, 1.0],
            [2.0, 0.0],
        ])

        with self.assertRaisesRegex(ValueError, "accurately"):
            AnalogReservoir._scale_spectral_radius(
                matrix,
                target_radius=smallest_positive_float32,
            )

    def test_rejects_invalid_observation_and_state_shapes(self):
        invalid_calls = (
            lambda: self.reservoir.update_state(
                torch.randn(3, 4), torch.zeros(3, 8)
            ),
            lambda: self.reservoir.update_state(
                torch.randn(3, 5), torch.zeros(3, 7)
            ),
            lambda: self.reservoir.update_state(
                torch.randn(2, 3, 5), torch.zeros(6, 8)
            ),
            lambda: self.reservoir.features_from_state(torch.zeros(3, 7)),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()

    def test_rejects_invalid_base_configuration(self):
        invalid_arguments = (
            {"connectivity": 0.0},
            {"connectivity": 1.1},
            {"spectral_radius": 0.0},
            {"spectral_radius": -1.0},
            {"leak_rate": 0.0},
            {"leak_rate": 1.1},
            {"activation": "unknown"},
            {"train_reservoir": True},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                AnalogReservoir(
                    input_dim=5,
                    reservoir_dim=8,
                    **arguments,
                )

    def test_rejects_nonfinite_and_negative_float_configuration(self):
        invalid_arguments = (
            {"spectral_radius": math.nan},
            {"spectral_radius": math.inf},
            {"spectral_radius": 1e100},
            {"connectivity": math.nan},
            {"connectivity": math.inf},
            {"input_scale": math.nan},
            {"input_scale": math.inf},
            {"input_scale": 1e100},
            {"input_scale": -0.1},
            {"bias_scale": math.nan},
            {"bias_scale": math.inf},
            {"bias_scale": 1e100},
            {"bias_scale": -0.1},
            {"leak_rate": math.nan},
            {"leak_rate": math.inf},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                AnalogReservoir(
                    input_dim=5,
                    reservoir_dim=8,
                    **arguments,
                )

    def test_rejects_positive_parameters_that_underflow_float32(self):
        invalid_arguments = (
            {"spectral_radius": 1e-100},
            {"connectivity": 1e-100},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                try:
                    AnalogReservoir(
                        input_dim=5,
                        reservoir_dim=8,
                        **arguments,
                    )
                except ValueError:
                    continue
                except Exception as error:
                    self.fail(
                        f"Expected ValueError, got "
                        f"{type(error).__name__}: {error}"
                    )
                self.fail("ValueError not raised")


class LIFReservoirTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.reservoir = LIFReservoir(
            input_dim=5,
            reservoir_dim=8,
            connectivity=1.0,
            lif_threshold=0.1,
        )

    def test_packs_membrane_and_binary_spikes_and_uses_spikes_as_features(self):
        observations = torch.randn(3, 5, requires_grad=True)
        initial_state = torch.zeros(3, 16, requires_grad=True)
        next_state = self.reservoir.update_state(
            observations,
            initial_state,
        )
        _, spikes = next_state.chunk(2, dim=-1)

        self.assertEqual(self.reservoir.state_dim, 16)
        self.assertEqual(self.reservoir.feature_dim, 8)
        self.assertEqual(next_state.shape, (3, 16))
        self.assertTrue(torch.logical_or(spikes == 0.0, spikes == 1.0).all())
        self.assertTrue(torch.equal(
            self.reservoir.features_from_state(next_state),
            spikes,
        ))
        self.assertFalse(next_state.requires_grad)

    def test_previous_spikes_drive_recurrent_current(self):
        reservoir = LIFReservoir(
            input_dim=5,
            reservoir_dim=8,
            connectivity=1.0,
            lif_beta=0.0,
            lif_threshold=0.5,
            num_reservoir_steps=1,
        )
        with torch.no_grad():
            reservoir.w_in.zero_()
            reservoir.reservoir_bias.zero_()
            reservoir.w_res.copy_(torch.eye(8))

        next_state = reservoir.update_state(
            torch.zeros(2, 5),
            torch.cat(
                (torch.zeros(2, 8), torch.ones(2, 8)),
                dim=-1,
            ),
        )
        _, spikes = next_state.chunk(2, dim=-1)

        self.assertTrue(torch.equal(spikes, torch.ones_like(spikes)))

    def test_rejects_invalid_observation_and_packed_state_shapes(self):
        invalid_calls = (
            lambda: self.reservoir.update_state(
                torch.randn(3, 4), torch.zeros(3, 16)
            ),
            lambda: self.reservoir.update_state(
                torch.randn(3, 5), torch.zeros(3, 8)
            ),
            lambda: self.reservoir.update_state(
                torch.randn(2, 3, 5), torch.zeros(6, 16)
            ),
            lambda: self.reservoir.features_from_state(torch.zeros(3, 8)),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()

    def test_rejects_invalid_lif_configuration(self):
        invalid_arguments = (
            {"lif_beta": -0.1},
            {"lif_beta": 1.0},
            {"lif_threshold": 0.0},
            {"reset_mode": "invalid"},
            {"num_reservoir_steps": 0},
            {"num_reservoir_steps": 1.5},
            {"train_reservoir": True},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                LIFReservoir(
                    input_dim=5,
                    reservoir_dim=8,
                    **arguments,
                )

    def test_rejects_nonfinite_lif_configuration(self):
        invalid_arguments = (
            {"lif_beta": math.nan},
            {"lif_beta": math.inf},
            {"lif_threshold": math.nan},
            {"lif_threshold": math.inf},
            {"surrogate_alpha": math.nan},
            {"surrogate_alpha": math.inf},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                LIFReservoir(
                    input_dim=5,
                    reservoir_dim=8,
                    **arguments,
                )


if __name__ == "__main__":
    unittest.main()
