import os
import sys
import unittest

import torch
import torch.nn as nn


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

import rsl_rl.modules as public_modules
from rsl_rl.modules.actors import MLPActor
from rsl_rl.modules.reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
    _ReservoirReadoutActor,
)
from rsl_rl.modules.reservoirs import AnalogReservoir, LIFReservoir
from rsl_rl.modules.snn import SNNActor


ACTOR_CASES = (
    (AnalogReservoirMLPReadoutActor, AnalogReservoir, MLPActor, 7),
    (AnalogReservoirSNNReadoutActor, AnalogReservoir, SNNActor, 7),
    (LIFReservoirMLPReadoutActor, LIFReservoir, MLPActor, 14),
    (LIFReservoirSNNReadoutActor, LIFReservoir, SNNActor, 14),
)


class PublicReservoirActorAPITest(unittest.TestCase):
    def test_exports_supported_reservoir_actor_types(self):
        for actor_type, _, _, _ in ACTOR_CASES:
            with self.subTest(name=actor_type.__name__):
                self.assertIs(
                    getattr(public_modules, actor_type.__name__),
                    actor_type,
                )


class ReservoirActorCombinationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_all_named_combinations_use_the_requested_components(self):
        for actor_type, reservoir_type, readout_type, state_dim in ACTOR_CASES:
            with self.subTest(actor=actor_type.__name__):
                actor = actor_type(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=7,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=(6, 4),
                )

                self.assertIs(type(actor), actor_type)
                self.assertIsInstance(actor.reservoir, reservoir_type)
                self.assertIsInstance(actor.readout, readout_type)
                self.assertTrue(actor.is_reservoir)
                self.assertEqual(actor.input_dim, 5)
                self.assertEqual(actor.output_dim, 2)
                self.assertEqual(actor.reservoir_dim, 7)
                self.assertEqual(actor.reservoir_state_dim, state_dim)
                self.assertFalse(actor.include_input_in_readout)
                self.assertEqual(actor.readout.input_dim, 7)

    def test_forward_supports_arbitrary_leading_dimensions_and_replay(self):
        observations = torch.randn(2, 3, 5)

        for actor_type, _, _, state_dim in ACTOR_CASES:
            with self.subTest(actor=actor_type.__name__):
                actor = actor_type(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=7,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=(6,),
                    reservoir_lif_threshold=0.1,
                    snn_lif_threshold=0.1,
                )
                initial_states = torch.zeros(2, 3, state_dim)

                actions, updated_states = actor(
                    observations,
                    initial_states,
                )
                replayed_actions = actor.readout_process(
                    observations,
                    updated_states,
                )

                self.assertEqual(actions.shape, (2, 3, 2))
                self.assertEqual(updated_states.shape, (2, 3, state_dim))
                self.assertTrue(torch.equal(actions, replayed_actions))

    def test_none_state_and_direct_input_features_are_supported(self):
        for actor_type, _, _, state_dim in ACTOR_CASES:
            with self.subTest(actor=actor_type.__name__):
                actor = actor_type(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=7,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=(6,),
                    include_input_in_readout=True,
                )

                actions, updated_states = actor(torch.randn(4, 5))

                self.assertEqual(actions.shape, (4, 2))
                self.assertEqual(updated_states.shape, (4, state_dim))
                self.assertTrue(actor.include_input_in_readout)
                self.assertEqual(actor.readout.input_dim, 12)

    def test_snn_combinations_define_zero_rates_for_empty_batches(self):
        snn_cases = (
            (AnalogReservoirSNNReadoutActor, 7),
            (LIFReservoirSNNReadoutActor, 14),
        )
        observations = torch.empty(0, 5, dtype=torch.float64)

        for actor_type, state_dim in snn_cases:
            with self.subTest(actor=actor_type.__name__):
                actor = actor_type(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=7,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=(6, 4),
                ).double()

                actions, updated_states = actor(observations)

                self.assertEqual(actions.shape, (0, 2))
                self.assertEqual(updated_states.shape, (0, state_dim))
                self.assertEqual(len(actor.readout.last_spike_rates), 2)
                for spike_rate in actor.readout.last_spike_rates:
                    self.assertEqual(spike_rate.ndim, 0)
                    self.assertEqual(spike_rate.dtype, observations.dtype)
                    self.assertEqual(spike_rate.device, observations.device)
                    self.assertFalse(spike_rate.requires_grad)
                    self.assertTrue(torch.isfinite(spike_rate))
                    self.assertEqual(spike_rate.item(), 0.0)

    def test_reservoir_is_fixed_and_detached_but_readout_has_gradients(self):
        for actor_type, _, _, _ in ACTOR_CASES:
            with self.subTest(actor=actor_type.__name__):
                actor = actor_type(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=7,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=(6,),
                )

                actions, updated_states = actor(
                    torch.randn(4, 5, requires_grad=True)
                )
                actions.sum().backward()

                reservoir_parameters = dict(
                    actor.reservoir.named_parameters()
                )
                reservoir_buffers = dict(actor.reservoir.named_buffers())
                self.assertEqual(reservoir_parameters, {})
                for name in ("w_in", "w_res", "reservoir_bias"):
                    self.assertIn(name, reservoir_buffers)
                    self.assertFalse(reservoir_buffers[name].requires_grad)
                self.assertFalse(updated_states.requires_grad)
                readout_gradients = [
                    parameter.grad
                    for parameter in actor.readout.parameters()
                ]
                self.assertTrue(all(
                    gradient is not None
                    for gradient in readout_gradients
                ))
                self.assertGreater(
                    sum(
                        gradient.abs().sum().item()
                        for gradient in readout_gradients
                    ),
                    0.0,
                )

    def test_parameters_are_routed_only_to_the_relevant_components(self):
        analog_mlp = AnalogReservoirMLPReadoutActor(
            input_dim=5,
            output_dim=2,
            reservoir_dim=7,
            reservoir_connectivity=1.0,
            leak_rate=0.25,
            reservoir_activation="relu",
            readout_hidden_dims=(6,),
            readout_activation="tanh",
            reservoir_lif_beta=-1.0,
            num_snn_steps=0,
        )
        lif_snn = LIFReservoirSNNReadoutActor(
            input_dim=5,
            output_dim=2,
            reservoir_dim=7,
            reservoir_connectivity=1.0,
            reservoir_lif_beta=0.4,
            reservoir_lif_threshold=0.3,
            reservoir_surrogate_alpha=2.0,
            reservoir_reset_mode="zero",
            readout_hidden_dims=(6,),
            num_snn_steps=2,
            snn_lif_beta=0.3,
            snn_lif_threshold=0.2,
            snn_surrogate_alpha=3.0,
            snn_reset_mode="zero",
            snn_input_scale=1.5,
            leak_rate=-1.0,
            reservoir_activation="unknown",
        )

        self.assertEqual(analog_mlp.reservoir.leak_rate, 0.25)
        self.assertIsInstance(analog_mlp.reservoir.activation, nn.ReLU)
        self.assertIsInstance(analog_mlp.readout.network[1], nn.Tanh)
        self.assertAlmostEqual(lif_snn.reservoir.lif_neuron.beta, 0.4)
        self.assertAlmostEqual(
            lif_snn.reservoir.lif_neuron.threshold,
            0.3,
        )
        self.assertEqual(lif_snn.reservoir.lif_neuron.reset_mode, "zero")
        self.assertEqual(lif_snn.readout.num_snn_steps, 2)
        self.assertEqual(lif_snn.readout.lif_layers[0].beta, 0.3)
        self.assertEqual(lif_snn.readout.lif_layers[0].threshold, 0.2)
        self.assertEqual(lif_snn.readout.lif_layers[0].reset_mode, "zero")
        self.assertEqual(lif_snn.readout.input_scale, 1.5)


class ReservoirActorValidationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.reservoir = AnalogReservoir(
            input_dim=5,
            reservoir_dim=7,
            connectivity=1.0,
        )

    def make_actor(self, include_input=False):
        readout_input_dim = 12 if include_input else 7
        return _ReservoirReadoutActor(
            reservoir=self.reservoir,
            readout=MLPActor(readout_input_dim, (6,), 2),
            output_dim=2,
            include_input_in_readout=include_input,
        )

    def test_base_rejects_readout_input_and_output_mismatches(self):
        with self.assertRaisesRegex(ValueError, "readout input_dim"):
            _ReservoirReadoutActor(
                reservoir=self.reservoir,
                readout=MLPActor(6, (4,), 2),
                output_dim=2,
            )

        with self.assertRaisesRegex(ValueError, "readout output_dim"):
            _ReservoirReadoutActor(
                reservoir=self.reservoir,
                readout=MLPActor(7, (4,), 3),
                output_dim=2,
            )

    def test_forward_rejects_invalid_input_and_complete_state_shape(self):
        actor = self.make_actor()

        invalid_calls = (
            lambda: actor(torch.tensor(1.0)),
            lambda: actor(torch.randn(2, 4)),
            lambda: actor(torch.randn(2, 3, 5), torch.zeros(6, 7)),
            lambda: actor(torch.randn(2, 3, 5), torch.zeros(2, 4, 7)),
            lambda: actor(torch.randn(2, 3, 5), torch.zeros(2, 3, 6)),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_readout_process_validates_shapes_without_updating_reservoir(self):
        actor = self.make_actor(include_input=True)
        observations = torch.randn(2, 3, 5)
        states = torch.randn(2, 3, 7)
        original_states = states.clone()

        actions = actor.readout_process(observations, states)

        self.assertEqual(actions.shape, (2, 3, 2))
        self.assertTrue(torch.equal(states, original_states))
        invalid_calls = (
            lambda: actor.readout_process(
                torch.randn(2, 4, 5),
                states,
            ),
            lambda: actor.readout_process(
                torch.randn(2, 3, 4),
                states,
            ),
            lambda: actor.readout_process(
                observations,
                torch.randn(2, 3, 6),
            ),
            lambda: actor.readout_process(
                observations,
                torch.randn(6, 7),
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
