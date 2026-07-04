import ast
import io
import inspect
import unittest

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

import rsl_rl.modules.actor_critic as actor_critic_module
from rsl_rl.algorithms import PPO, PPOtwin
from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.modules.actors import MLPActor
from rsl_rl.modules.reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
)
from rsl_rl.modules.snn import SNNActor


ACTOR_TYPES = {
    "mlp": MLPActor,
    "snn": SNNActor,
    "analog_reservoir_mlp": AnalogReservoirMLPReadoutActor,
    "analog_reservoir_snn": AnalogReservoirSNNReadoutActor,
    "lif_reservoir_mlp": LIFReservoirMLPReadoutActor,
    "lif_reservoir_snn": LIFReservoirSNNReadoutActor,
}
RESERVOIR_TYPES = {
    "analog_reservoir_mlp": 6,
    "analog_reservoir_snn": 6,
    "lif_reservoir_mlp": 12,
    "lif_reservoir_snn": 12,
}


def make_model(actor_type="mlp"):
    return ActorCritic(
        5,
        7,
        2,
        actor_type=actor_type,
        actor_hidden_dims=(8, 4),
        critic_hidden_dims=(8, 4),
        reservoir_dim=6,
        reservoir_connectivity=1.0,
        readout_hidden_dims=(6, 4),
        num_snn_steps=2,
        reservoir_lif_threshold=0.1,
        snn_lif_threshold=0.1,
    )


class ActorCriticFactoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_actor_critic_does_not_mutate_normal_validation_api(self):
        original_setter = Normal.set_default_validate_args
        original_setter_descriptor = inspect.getattr_static(
            Normal,
            "set_default_validate_args",
        )
        original_validate_args = Distribution._validate_args

        try:
            model = make_model("mlp")
            model.act(torch.randn(3, 5))

            self.assertTrue(callable(Normal.set_default_validate_args))
            self.assertIs(
                Normal.set_default_validate_args,
                original_setter,
            )
            self.assertIs(
                inspect.getattr_static(
                    Normal,
                    "set_default_validate_args",
                ),
                original_setter_descriptor,
            )
            self.assertEqual(
                Distribution._validate_args,
                original_validate_args,
            )
        finally:
            if "set_default_validate_args" in Normal.__dict__:
                del Normal.set_default_validate_args
            Distribution._validate_args = original_validate_args

    def test_registry_selects_all_six_actor_classes_and_preserves_critic(self):
        for actor_type, expected_class in ACTOR_TYPES.items():
            with self.subTest(actor_type=actor_type):
                model = make_model(actor_type)
                linear_layers = [
                    module
                    for module in model.critic
                    if isinstance(module, nn.Linear)
                ]

                self.assertIs(type(model.actor), expected_class)
                self.assertEqual(model.actor_type, actor_type)
                self.assertEqual(
                    [
                        (layer.in_features, layer.out_features)
                        for layer in linear_layers
                    ],
                    [(7, 8), (8, 4), (4, 1)],
                )

    def test_mlp_is_the_default_and_activation_is_reexported(self):
        model = ActorCritic(
            5,
            7,
            2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
        )

        self.assertIs(type(model.actor), MLPActor)
        self.assertEqual(model.actor_type, "mlp")
        self.assertIsInstance(get_activation("elu"), nn.ELU)

    def test_invalid_actor_type_is_explicit_and_registry_does_not_use_eval(self):
        with self.assertRaisesRegex(ValueError, "actor_type"):
            ActorCritic(5, 7, 2, actor_type="unknown")

        syntax_tree = ast.parse(inspect.getsource(actor_critic_module))
        eval_calls = [
            node
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "eval"
        ]
        self.assertEqual(eval_calls, [])

    def test_stateless_actor_distribution_and_critic_interfaces(self):
        observations = torch.randn(3, 5)
        critic_observations = torch.randn(3, 7)

        for actor_type in ("mlp", "snn"):
            with self.subTest(actor_type=actor_type):
                model = make_model(actor_type)
                actions = model.act(observations)

                self.assertEqual(actions.shape, (3, 2))
                self.assertEqual(model.action_mean.shape, (3, 2))
                self.assertEqual(model.action_std.shape, (3, 2))
                self.assertEqual(model.get_actions_log_prob(actions).shape, (3,))
                self.assertEqual(model.entropy.shape, (3,))
                self.assertEqual(
                    model.act_inference(observations).shape,
                    (3, 2),
                )
                self.assertEqual(
                    model.evaluate(critic_observations).shape,
                    (3, 1),
                )
                self.assertFalse(model.is_reservoir)
                with self.assertRaisesRegex(RuntimeError, "reservoir"):
                    model.act_for_ppo_update(observations, None)

    def test_snn_actor_mean_backpropagates_through_surrogate_gradient(self):
        model = make_model("snn")
        observations = torch.full((3, 5), 2.0, requires_grad=True)

        model.update_distribution(observations)
        model.action_mean.sum().backward()

        self.assertIsNotNone(observations.grad)
        self.assertGreater(observations.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.actor.linear_layers[0].weight.grad)

    def test_reservoir_paths_update_return_and_replay_saved_state(self):
        observations = torch.randn(3, 5)

        for actor_type, state_width in RESERVOIR_TYPES.items():
            with self.subTest(actor_type=actor_type):
                model = make_model(actor_type)
                initial_states = torch.zeros(3, state_width)

                actions, next_states = model.act(
                    observations,
                    initial_states,
                )
                rollout_mean = model.action_mean.clone()
                replay_actions = model.act_for_ppo_update(
                    observations,
                    next_states,
                )
                inference_mean, inference_states = model.act_inference(
                    observations,
                    initial_states,
                )

                self.assertTrue(model.is_reservoir)
                self.assertEqual(actions.shape, (3, 2))
                self.assertFalse(actions.requires_grad)
                self.assertEqual(next_states.shape, (3, state_width))
                self.assertEqual(replay_actions.shape, (3, 2))
                self.assertTrue(torch.equal(model.action_mean, rollout_mean))
                self.assertTrue(torch.equal(inference_mean, rollout_mean))
                self.assertTrue(torch.equal(inference_states, next_states))
                self.assertEqual(
                    model.evaluate(torch.randn(3, 7)).shape,
                    (3, 1),
                )

    def test_reservoir_paths_require_state(self):
        model = make_model("analog_reservoir_mlp")
        observations = torch.randn(3, 5)

        for call in (
            lambda: model.update_distribution(observations),
            lambda: model.act(observations),
            lambda: model.act_inference(observations),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "reservoir_states"):
                    call()

    def test_all_actor_types_round_trip_strict_state_dict(self):
        for actor_type in ACTOR_TYPES:
            with self.subTest(actor_type=actor_type):
                model = make_model(actor_type)
                rebuilt_model = make_model(actor_type)
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)

                rebuilt_model.load_state_dict(
                    torch.load(
                        checkpoint,
                        map_location="cpu",
                        weights_only=True,
                    ),
                    strict=True,
                )

                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(
                        value,
                        rebuilt_model.state_dict()[name],
                    ))

    def test_ppotwin_updates_all_actor_types(self):
        observations = torch.full((4, 5), 2.0)
        critic_observations = torch.randn(4, 7)
        env_ids = torch.arange(4)

        for actor_type in ACTOR_TYPES:
            with self.subTest(actor_type=actor_type):
                torch.manual_seed(31)
                model = make_model(actor_type)
                algorithm = PPOtwin(
                    model,
                    num_learning_epochs=1,
                    num_mini_batches=1,
                    learning_rate=1e-2,
                    schedule="fixed",
                    device="cpu",
                )
                algorithm.init_storage(4, 1, [5], [7], [2])

                readout_before = None
                if model.is_reservoir:
                    reservoir_states = torch.zeros(
                        4,
                        model.actor.reservoir_state_dim,
                    )
                    readout_before = {
                        name: parameter.detach().clone()
                        for name, parameter
                        in model.actor.readout.named_parameters()
                    }
                    actions, _ = algorithm.act(
                        env_ids,
                        torch.zeros(4, dtype=torch.long),
                        observations,
                        critic_observations,
                        reservoir_states,
                    )
                else:
                    actions = algorithm.act(
                        env_ids,
                        torch.zeros(4, dtype=torch.long),
                        observations,
                        critic_observations,
                    )

                algorithm.process_env_step(
                    1,
                    env_ids,
                    torch.tensor([1.0, 2.0, 3.0, 4.0]),
                    torch.zeros(4, dtype=torch.bool),
                    critic_observations,
                    {},
                )
                algorithm.compute_returns(critic_observations)
                losses = algorithm.update()

                self.assertEqual(actions.shape, (4, 2))
                self.assertTrue(all(torch.isfinite(
                    torch.tensor(loss),
                ) for loss in losses))
                if model.is_reservoir:
                    self.assertTrue(any(
                        not torch.equal(
                            readout_before[name],
                            parameter.detach(),
                        )
                        for name, parameter
                        in model.actor.readout.named_parameters()
                    ))
                    self.assertIsNone(model.actor.reservoir.w_in.grad)
                    self.assertIsNone(model.actor.reservoir.w_res.grad)

    def test_ordinary_ppo_rejects_reservoir_actor_at_act(self):
        model = make_model("analog_reservoir_mlp")
        algorithm = PPO(model)

        with self.assertRaisesRegex(RuntimeError, "PPOtwin"):
            algorithm.act(torch.randn(2, 5), torch.randn(2, 7))

    def test_ordinary_ppo_accepts_stateless_actors(self):
        for actor_type in ("mlp", "snn"):
            with self.subTest(actor_type=actor_type):
                model = make_model(actor_type)
                actions = PPO(model).act(
                    torch.full((2, 5), 2.0),
                    torch.randn(2, 7),
                )

                self.assertEqual(actions.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
