import importlib.util
import unittest

import torch
import torch.nn as nn

import rsl_rl.modules as public_modules
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.modules.actor_critic_twin import ActorCriticTwin
from rsl_rl.modules.actors import MLPActor, get_activation
from rsl_rl.modules.snn import LIFNeuron, SNNActor


class PublicActorAPITest(unittest.TestCase):
    def test_exports_supported_actor_types(self):
        expected_exports = {
            "ActorCritic": ActorCritic,
            "ActorCriticTwin": ActorCriticTwin,
            "ActorCriticRecurrent": ActorCriticRecurrent,
            "MLPActor": MLPActor,
            "SNNActor": SNNActor,
        }

        for name, expected_type in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(getattr(public_modules, name), expected_type)

    def test_does_not_export_removed_actor_types(self):
        removed_names = (
            "ActorCriticReservoir",
            "ActorCriticSNNReservoir",
            "ActorCriticSNN",
            "ReservoirActor",
            "SNNReservoirActor",
            "MLPReadout",
            "SNNReadout",
            "ReservoirReadoutActor",
        )

        for name in removed_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(public_modules, name))

    def test_removed_actor_modules_cannot_be_located(self):
        removed_module_names = (
            "rsl_rl.modules.actor_critic_reservoir",
            "rsl_rl.modules.actor_critic_snn_reservoir",
            "rsl_rl.modules.actor_critic_reservoir_combinations",
            "rsl_rl.modules.actor_critic_SNN",
            "rsl_rl.modules.reservoir_dynamics",
            "rsl_rl.modules.reservoir_readouts",
            "rsl_rl.modules.reservoir_readout_actor",
        )

        for module_name in removed_module_names:
            with self.subTest(module=module_name):
                try:
                    module_spec = importlib.util.find_spec(module_name)
                except Exception as error:
                    self.fail(
                        f"Looking up {module_name} raised "
                        f"{type(error).__name__}: {error}"
                    )
                self.assertIsNone(module_spec)


class MLPActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_accepts_arbitrary_leading_dimensions_and_backpropagates(self):
        actor = MLPActor(5, [8, 4], 2, activation="tanh")
        observations = torch.randn(2, 3, 5, requires_grad=True)

        actions = actor(observations)
        actions.sum().backward()

        self.assertEqual(actions.shape, (2, 3, 2))
        self.assertIsNotNone(observations.grad)
        self.assertIsNotNone(actor.network[0].weight.grad)

    def test_activation_names_are_supported(self):
        expected_types = {
            "elu": nn.ELU,
            "selu": nn.SELU,
            "relu": nn.ReLU,
            "crelu": nn.ReLU,
            "lrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        for name, expected_type in expected_types.items():
            with self.subTest(name=name):
                self.assertIsInstance(get_activation(name), expected_type)

        with self.assertRaises(ValueError):
            get_activation("unknown")

    def test_rejects_invalid_dimensions(self):
        invalid_arguments = [
            (0, [4], 2),
            (3, [], 2),
            (3, [4, 0], 2),
            (3, [4], 0),
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    MLPActor(*arguments)

    def test_rejects_wrong_observation_dimension(self):
        actor = MLPActor(5, [4], 2)

        with self.assertRaisesRegex(ValueError, "5"):
            actor(torch.randn(3, 4))

    def test_uses_required_initialization(self):
        actor = MLPActor(5, [8, 4], 2)
        linear_layers = [
            layer for layer in actor.network
            if isinstance(layer, nn.Linear)
        ]

        for layer in linear_layers:
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))

        hidden_gram = linear_layers[0].weight.T @ linear_layers[0].weight
        self.assertTrue(
            torch.allclose(
                hidden_gram,
                2.0 * torch.eye(5),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        output_gram = (
            linear_layers[-1].weight
            @ linear_layers[-1].weight.T
        )
        self.assertTrue(
            torch.allclose(
                output_gram,
                0.0001 * torch.eye(2),
                atol=1e-7,
                rtol=1e-5,
            )
        )


class SNNActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_accepts_arbitrary_leading_dimensions_and_backpropagates(self):
        actor = SNNActor(5, [8, 4], 2, num_snn_steps=3)
        observations = torch.randn(2, 3, 5, requires_grad=True)

        actions = actor(observations)
        actions.sum().backward()

        self.assertEqual(actions.shape, (2, 3, 2))
        self.assertIsNotNone(observations.grad)
        self.assertGreater(observations.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(actor.linear_layers[0].weight.grad)
        self.assertGreater(
            actor.linear_layers[0].weight.grad.abs().sum().item(),
            0.0,
        )

    def test_records_detached_bounded_spike_rate_per_hidden_layer(self):
        actor = SNNActor(
            5,
            [8, 4],
            2,
            num_snn_steps=3,
            input_scale=3.0,
        )

        actor(torch.randn(6, 5))

        self.assertEqual(len(actor.last_spike_rates), 2)
        for spike_rate in actor.last_spike_rates:
            self.assertEqual(spike_rate.ndim, 0)
            self.assertFalse(spike_rate.requires_grad)
            self.assertGreaterEqual(spike_rate.item(), 0.0)
            self.assertLessEqual(spike_rate.item(), 1.0)

    def test_empty_batch_records_finite_zero_spike_rates(self):
        actor = SNNActor(5, [8, 4], 2).double()
        observations = torch.empty(0, 5, dtype=torch.float64)

        actions = actor(observations)

        self.assertEqual(actions.shape, (0, 2))
        self.assertEqual(len(actor.last_spike_rates), 2)
        for spike_rate in actor.last_spike_rates:
            self.assertEqual(spike_rate.ndim, 0)
            self.assertEqual(spike_rate.dtype, observations.dtype)
            self.assertEqual(spike_rate.device, observations.device)
            self.assertFalse(spike_rate.requires_grad)
            self.assertTrue(torch.isfinite(spike_rate))
            self.assertEqual(spike_rate.item(), 0.0)

    def test_membrane_state_is_reset_between_forward_calls(self):
        actor = SNNActor(5, [8, 4], 2, num_snn_steps=3)
        observations = torch.randn(6, 5)

        first = actor(observations)
        second = actor(observations)

        self.assertTrue(torch.equal(first, second))

    def test_rejects_wrong_observation_dimension(self):
        actor = SNNActor(5, [4], 2)

        with self.assertRaisesRegex(ValueError, "5"):
            actor(torch.randn(3, 4))

    def test_rejects_invalid_parameters(self):
        invalid_actor_kwargs = [
            {"input_dim": 0, "hidden_dims": [4], "output_dim": 2},
            {"input_dim": 3, "hidden_dims": [], "output_dim": 2},
            {"input_dim": 3, "hidden_dims": [4], "output_dim": 0},
            {
                "input_dim": 3,
                "hidden_dims": [4],
                "output_dim": 2,
                "num_snn_steps": 0,
            },
        ]
        for kwargs in invalid_actor_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SNNActor(**kwargs)

        for kwargs in [
            {"beta": -0.1},
            {"beta": 1.0},
            {"threshold": 0.0},
            {"surrogate_alpha": 0.0},
            {"reset_mode": "invalid"},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LIFNeuron(**kwargs)

    def test_uses_required_initialization(self):
        actor = SNNActor(5, [8], 2)

        hidden_gram = (
            actor.linear_layers[0].weight.T
            @ actor.linear_layers[0].weight
        )
        self.assertTrue(
            torch.allclose(
                hidden_gram,
                torch.eye(5),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        output_gram = actor.output_layer.weight @ actor.output_layer.weight.T
        self.assertTrue(
            torch.allclose(
                output_gram,
                0.0001 * torch.eye(2),
                atol=1e-7,
                rtol=1e-5,
            )
        )
        for layer in [*actor.linear_layers, actor.output_layer]:
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))


if __name__ == "__main__":
    unittest.main()
