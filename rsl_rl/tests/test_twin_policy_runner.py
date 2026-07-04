import contextlib
import ast
import copy
import io
import inspect
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


RSL_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.runners.twin_policy_runner import TwinPolicyRunner
import rsl_rl.runners.twin_policy_runner as twin_policy_runner_module


ACTOR_STATE_DIMS = {
    "mlp": None,
    "snn": None,
    "analog_reservoir_mlp": 6,
    "analog_reservoir_snn": 6,
    "lif_reservoir_mlp": 12,
    "lif_reservoir_snn": 12,
}


class FakeEnvironment:
    def __init__(self):
        self.num_envs = 3
        self.num_obs = 5
        self.num_privileged_obs = 7
        self.episode_length_buf = torch.tensor([0, 1, 2])
        self.mu_shape = 1
        self.omega_shape = 1
        self.cfg = type("Cfg", (), {
            "control": type("Control", (), {
                "mu_cycle": torch.tensor([1, 2, 2]),
                "omega_cycle": 1,
            })(),
        })()

    def step(self, actions):
        observations = torch.zeros(3, self.num_obs)
        privileged_observations = torch.zeros(
            3,
            self.num_privileged_obs,
        )
        rewards = torch.zeros(3)
        dones = torch.zeros(3, dtype=torch.bool)
        return observations, privileged_observations, rewards, dones, {}

    def reset(self):
        return (
            torch.zeros(self.num_envs, self.num_obs),
            torch.zeros(self.num_envs, self.num_privileged_obs),
        )


class RecordingPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.last_observations = None

    def __call__(self, observations):
        self.last_observations = observations
        return torch.ones(
            observations.shape[0],
            self.action_dim,
            device=observations.device,
        )


class RecordingReservoirPolicy(RecordingPolicy):
    def __call__(self, observations, reservoir_states):
        self.last_observations = observations
        return (
            torch.ones(
                observations.shape[0],
                self.action_dim,
                device=observations.device,
            ),
            reservoir_states + 2.0,
        )


class RecordingWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class TwinPolicyRunnerTest(unittest.TestCase):
    def test_update_ids_support_scalar_period(self):
        episode_lengths = torch.tensor([0, 1, 2, 3, 4])

        update_ids = TwinPolicyRunner.get_update_env_ids(
            episode_lengths,
            2,
        )

        self.assertTrue(torch.equal(
            update_ids,
            torch.tensor([0, 2, 4]),
        ))

    def test_update_ids_support_per_env_periods(self):
        episode_lengths = torch.tensor([0, 1, 2, 3, 4])
        periods = torch.tensor([2, 1, 2, 3, 5])

        update_ids = TwinPolicyRunner.get_update_env_ids(
            episode_lengths,
            periods,
        )

        self.assertTrue(torch.equal(
            update_ids,
            torch.tensor([0, 1, 2, 3]),
        ))

    def test_runner_builds_factory_actor_types_and_declared_states(self):
        for actor_type, expected_state_dim in ACTOR_STATE_DIMS.items():
            with self.subTest(actor_type=actor_type):
                runner = TwinPolicyRunner(
                    FakeEnvironment(),
                    self.make_train_cfg(actor_type),
                    device="cpu",
                )

                self.assertEqual(
                    runner.mu_alg.actor_critic.actor_type,
                    actor_type,
                )
                self.assertEqual(
                    runner.omega_alg.actor_critic.actor_type,
                    actor_type,
                )
                if expected_state_dim is None:
                    self.assertIsNone(runner.mu_reservoir_states)
                    self.assertIsNone(runner.omega_reservoir_states)
                else:
                    self.assertEqual(
                        runner.mu_reservoir_states.shape,
                        (3, expected_state_dim),
                    )
                    self.assertEqual(
                        runner.omega_reservoir_states.shape,
                        (3, expected_state_dim),
                    )

    def test_runner_only_imports_unified_actor_critic_policies(self):
        tree = ast.parse(inspect.getsource(twin_policy_runner_module))
        module_import = next(
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "rsl_rl.modules"
        )

        self.assertEqual(
            {alias.name for alias in module_import.names},
            {"ActorCritic", "ActorCriticTwin"},
        )
        self.assertFalse(hasattr(
            TwinPolicyRunner,
            "get_reservoir_state_dim",
        ))

    def test_runner_rejects_unknown_policy_and_algorithm_names(self):
        cases = (
            ("policy_class_name", "UnknownPolicy", "ActorCritic"),
            ("algorithm_class_name", "UnknownAlgorithm", "PPOtwin"),
        )

        for config_key, invalid_name, supported_name in cases:
            with self.subTest(config_key=config_key):
                train_cfg = self.make_train_cfg("mlp")
                train_cfg["runner"][config_key] = invalid_name

                error = None
                try:
                    TwinPolicyRunner(
                        FakeEnvironment(),
                        train_cfg,
                        device="cpu",
                    )
                except Exception as caught_error:
                    error = caught_error

                self.assertIsInstance(error, ValueError)
                self.assertRegex(
                    str(error),
                    f"{invalid_name}.*{supported_name}",
                )

    def test_runner_does_not_evaluate_malicious_policy_expression(self):
        marker = "TWIN_POLICY_RUNNER_EVAL_EXECUTED"
        malicious_name = (
            f"__import__('os').environ.__setitem__('{marker}', '1') "
            "or ActorCritic"
        )
        train_cfg = self.make_train_cfg("mlp")
        train_cfg["runner"]["policy_class_name"] = malicious_name
        os.environ.pop(marker, None)

        try:
            with self.assertRaisesRegex(ValueError, "ActorCritic"):
                TwinPolicyRunner(
                    FakeEnvironment(),
                    train_cfg,
                    device="cpu",
                )
            self.assertNotIn(marker, os.environ)
        finally:
            os.environ.pop(marker, None)

    def test_inference_only_evaluates_due_environments(self):
        runner = TwinPolicyRunner.__new__(TwinPolicyRunner)
        runner.env = FakeEnvironment()
        runner.device = "cpu"
        runner.mu_cached = torch.zeros(3, 1)
        runner.omega_cached = torch.zeros(3, 1)
        runner.actions = torch.zeros(3, 2)
        mu_policy = RecordingPolicy(1)
        omega_policy = RecordingPolicy(1)
        observations = torch.zeros(3, 2)

        runner.inference_rollout(
            observations,
            mu_policy,
            omega_policy,
        )

        self.assertEqual(mu_policy.last_observations.shape[0], 2)
        self.assertEqual(omega_policy.last_observations.shape[0], 3)

    def test_inference_resets_both_reservoir_states_for_done_envs(self):
        runner = TwinPolicyRunner.__new__(TwinPolicyRunner)
        runner.env = FakeEnvironment()
        runner.env.step = lambda actions: (
            torch.zeros(3, runner.env.num_obs),
            torch.zeros(3, runner.env.num_privileged_obs),
            torch.zeros(3),
            torch.tensor([False, True, True]),
            {},
        )
        runner.device = "cpu"
        runner.mu_cached = torch.zeros(3, 1)
        runner.omega_cached = torch.zeros(3, 1)
        runner.actions = torch.zeros(3, 2)
        runner.mu_reservoir_states = torch.ones(3, 2)
        runner.omega_reservoir_states = torch.ones(3, 2)

        runner.inference_rollout(
            torch.zeros(3, runner.env.num_obs),
            RecordingReservoirPolicy(1),
            RecordingReservoirPolicy(1),
        )

        self.assertTrue(torch.equal(
            runner.mu_reservoir_states,
            torch.tensor([
                [3.0, 3.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]),
        ))
        self.assertTrue(torch.equal(
            runner.omega_reservoir_states,
            torch.tensor([
                [3.0, 3.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]),
        ))

    def test_checkpoint_restores_both_optimizers(self):
        runner = TwinPolicyRunner.__new__(TwinPolicyRunner)
        runner.device = "cpu"
        runner.current_learning_iteration = 7
        runner.mu_alg = self.make_algorithm()
        runner.omega_alg = self.make_algorithm()
        self.take_optimizer_step(runner.mu_alg, 1.0)
        self.take_optimizer_step(runner.omega_alg, 2.0)
        expected_mu_optimizer = copy.deepcopy(
            runner.mu_alg.optimizer.state_dict()
        )
        expected_omega_optimizer = copy.deepcopy(
            runner.omega_alg.optimizer.state_dict()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pt")
            runner.save(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            self.assertIn("mu_optimizer_state_dict", checkpoint)
            self.assertIn("omega_optimizer_state_dict", checkpoint)

            runner.mu_alg.optimizer = torch.optim.Adam(
                runner.mu_alg.actor_critic.parameters(),
                lr=0.123,
            )
            runner.omega_alg.optimizer = torch.optim.Adam(
                runner.omega_alg.actor_critic.parameters(),
                lr=0.456,
            )
            runner.current_learning_iteration = -1
            runner.load(checkpoint_path, load_optimizer=True)

            self.assert_nested_equal(
                expected_mu_optimizer,
                runner.mu_alg.optimizer.state_dict(),
            )
            self.assert_nested_equal(
                expected_omega_optimizer,
                runner.omega_alg.optimizer.state_dict(),
            )
            self.assertEqual(runner.current_learning_iteration, 7)

    def test_checkpoint_loading_is_strict(self):
        runner = TwinPolicyRunner.__new__(TwinPolicyRunner)
        runner.device = "cpu"
        runner.current_learning_iteration = 0
        runner.mu_alg = self.make_algorithm()
        runner.omega_alg = self.make_algorithm()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pt")
            runner.save(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["mu_model_state_dict"].pop("weight")
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaises(RuntimeError):
                runner.load(checkpoint_path, load_optimizer=False)

    def test_log_methods_handle_empty_episode_buffer(self):
        runner = TwinPolicyRunner.__new__(TwinPolicyRunner)
        runner.device = "cpu"
        runner.env = SimpleNamespace(num_envs=4)
        runner.num_steps_per_env = 2
        runner.writer = RecordingWriter()
        runner.mu_tot_timesteps = 0
        runner.mu_tot_time = 0
        runner.omega_tot_timesteps = 0
        runner.omega_tot_time = 0
        actor_critic = SimpleNamespace(std=torch.ones(1))
        runner.mu_alg = SimpleNamespace(
            actor_critic=actor_critic,
            learning_rate=1e-3,
        )
        runner.omega_alg = SimpleNamespace(
            actor_critic=actor_critic,
            learning_rate=1e-3,
        )
        common = {
            "rewbuffer": [],
            "lenbuffer": [],
            "tot_iter": 10,
            "num_learning_iterations": 10,
        }
        mu_locs = {
            **common,
            "mu_ep_infos": [],
            "mu_collection_time": 0.1,
            "mu_learn_time": 0.2,
            "mean_mu_value_loss": 1.0,
            "mean_mu_surrogate_loss": 2.0,
            "mu_it": 0,
        }
        omega_locs = {
            **common,
            "omega_ep_infos": [],
            "omega_collection_time": 0.1,
            "omega_learn_time": 0.2,
            "mean_omega_value_loss": 1.0,
            "mean_omega_surrogate_loss": 2.0,
            "omega_it": 0,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            runner.mu_log(mu_locs)
            runner.omega_log(omega_locs)

    @staticmethod
    def make_algorithm():
        actor_critic = nn.Linear(2, 1)
        return type("Algorithm", (), {
            "actor_critic": actor_critic,
            "optimizer": torch.optim.Adam(actor_critic.parameters()),
        })()

    @staticmethod
    def take_optimizer_step(algorithm, scale):
        algorithm.optimizer.zero_grad()
        loss = algorithm.actor_critic(
            torch.full((2, 2), scale),
        ).square().sum()
        loss.backward()
        algorithm.optimizer.step()

    def assert_nested_equal(self, expected, actual):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self.assert_nested_equal(expected_item, actual_item)
        elif isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        else:
            self.assertEqual(expected, actual)

    @staticmethod
    def make_train_cfg(actor_type):
        return {
            "runner": {
                "policy_class_name": "ActorCritic",
                "algorithm_class_name": "PPOtwin",
                "num_steps_per_env": 1,
                "save_interval": 1,
            },
            "algorithm": {
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "schedule": "fixed",
            },
            "policy": {
                "actor_type": actor_type,
                "actor_hidden_dims": (8, 4),
                "critic_hidden_dims": (8, 4),
                "reservoir_dim": 6,
                "reservoir_connectivity": 1.0,
                "readout_hidden_dims": (6, 4),
                "num_snn_steps": 2,
                "reservoir_lif_threshold": 0.1,
                "snn_lif_threshold": 0.1,
            },
        }


if __name__ == "__main__":
    unittest.main()
