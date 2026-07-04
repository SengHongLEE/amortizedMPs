# Actor Module Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated Actor-Critic reservoir hierarchy with six reusable Actor classes selected by one configurable `ActorCritic`.

**Architecture:** `MLPActor` and `SNNActor` are standalone actors and reservoir readouts. Fixed Analog/LIF reservoirs are composed with either readout by one private base and four explicit public reservoir Actor classes. `ActorCritic` owns the actor registry, original MLP critic, and Gaussian policy interface; `PPOtwin` retains the stateful reservoir rollout/update path.

**Tech Stack:** Python 3.8, PyTorch 2.4.1, `unittest`, local `rsl_rl`

---

## File Map

Create:

- `rsl_rl/rsl_rl/modules/actors.py`: activation lookup and `MLPActor`.
- `rsl_rl/rsl_rl/modules/snn.py`: surrogate spike, LIF neuron, and `SNNActor`.
- `rsl_rl/rsl_rl/modules/reservoirs.py`: fixed Analog/LIF reservoir dynamics.
- `rsl_rl/rsl_rl/modules/reservoir_actors.py`: private composition base and four public reservoir actors.
- `rsl_rl/tests/test_actors.py`: standalone MLP/SNN actor behavior.
- `rsl_rl/tests/test_reservoirs.py`: Analog/LIF state behavior.
- `rsl_rl/tests/test_reservoir_actors.py`: four composition classes and replay behavior.
- `rsl_rl/tests/test_actor_critic_factory.py`: six-way factory, critic, PPO, checkpoint, and runner integration.

Modify:

- `rsl_rl/rsl_rl/modules/actor_critic.py`: single Actor-Critic factory and state-aware policy delegation.
- `rsl_rl/rsl_rl/modules/__init__.py`: new public exports only.
- `rsl_rl/rsl_rl/modules/actor_critic_twin.py`: import `ActorCritic` directly, preserving behavior.
- `rsl_rl/rsl_rl/runners/twin_policy_runner.py`: remove combination imports and allocate state from the selected actor.
- `rsl_rl/rsl_rl/algorithms/ppo.py`: reject reservoir policies on the ordinary PPO action path.

Delete:

- `rsl_rl/rsl_rl/modules/actor_critic_reservoir.py`
- `rsl_rl/rsl_rl/modules/actor_critic_snn_reservoir.py`
- `rsl_rl/rsl_rl/modules/actor_critic_reservoir_combinations.py`
- `rsl_rl/rsl_rl/modules/actor_critic_SNN.py`
- `rsl_rl/rsl_rl/modules/reservoir_dynamics.py`
- `rsl_rl/rsl_rl/modules/reservoir_readouts.py`
- `rsl_rl/rsl_rl/modules/reservoir_readout_actor.py`
- `rsl_rl/tests/test_actor_critic_reservoir.py`
- `rsl_rl/tests/test_actor_critic_snn_reservoir.py`
- `rsl_rl/tests/test_actor_critic_reservoir_combinations.py`
- `rsl_rl/tests/test_actor_critic_snn.py`
- `rsl_rl/tests/test_reservoir_dynamics.py`
- `rsl_rl/tests/test_reservoir_readouts.py`
- `rsl_rl/tests/test_reservoir_readout_actor.py`

Keep unchanged except for import fallout:

- `rsl_rl/rsl_rl/modules/actor_critic_recurrent.py`
- `rsl_rl/rsl_rl/modules/actor_critic_hdrl.py`
- Twin, recurrent, and hierarchical policy behavior and tests.

### Task 1: Standalone MLP and SNN Actors

**Files:**

- Create: `rsl_rl/rsl_rl/modules/actors.py`
- Create: `rsl_rl/rsl_rl/modules/snn.py`
- Create: `rsl_rl/tests/test_actors.py`

- [ ] **Step 1: Write failing tests for the two public stateless actors**

Create tests that prove both classes expose the same component contract and
that `SNNActor` remains stateless across policy decisions:

```python
import unittest

import torch

from rsl_rl.modules.actors import MLPActor
from rsl_rl.modules.snn import SNNActor


class StatelessActorsTest(unittest.TestCase):
    def test_mlp_actor_preserves_leading_dimensions_and_backpropagates(self):
        actor = MLPActor(5, [8, 4], 2, activation="elu")
        output = actor(torch.randn(2, 3, 5))
        output.sum().backward()

        self.assertEqual(actor.input_dim, 5)
        self.assertEqual(actor.output_dim, 2)
        self.assertEqual(output.shape, (2, 3, 2))
        self.assertIsNotNone(actor.network[0].weight.grad)

    def test_snn_actor_preserves_leading_dimensions_and_records_rates(self):
        actor = SNNActor(
            5,
            [8, 4],
            2,
            num_snn_steps=3,
            lif_threshold=0.1,
        )
        output = actor(torch.randn(2, 3, 5))
        output.sum().backward()

        self.assertEqual(output.shape, (2, 3, 2))
        self.assertEqual(len(actor.last_spike_rates), 2)
        self.assertIsNotNone(actor.linear_layers[0].weight.grad)
        for rate in actor.last_spike_rates:
            self.assertFalse(rate.requires_grad)
            self.assertGreaterEqual(rate.item(), 0.0)
            self.assertLessEqual(rate.item(), 1.0)

    def test_both_actors_reject_wrong_input_width(self):
        actors = (
            MLPActor(5, [4], 2),
            SNNActor(5, [4], 2),
        )
        for actor in actors:
            with self.subTest(actor=type(actor).__name__):
                with self.assertRaisesRegex(ValueError, "Expected"):
                    actor(torch.randn(3, 6))
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_actors -v
```

Expected: `ModuleNotFoundError` for `rsl_rl.modules.actors`.

- [ ] **Step 3: Implement `MLPActor` and activation validation**

Build `actors.py` around an explicit activation table and an MLP with the
component metadata required by reservoir composition:

```python
from typing import Sequence

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "crelu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as error:
        raise ValueError(f"Invalid activation: {name}.") from error


class MLPActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers = []
        for index, (source, target) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(source, target))
            if index < len(dimensions) - 2:
                layers.append(get_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected observations[..., {self.input_dim}], "
                f"received {tuple(observations.shape)}."
            )
        return self.network(observations)
```

Retain the existing orthogonal actor/readout initialization: hidden linear
layers use gain `sqrt(2)`, the output layer uses gain `0.01`, and biases are
zero.

- [ ] **Step 4: Move and rename the SNN primitives**

Move `SurrogateSpike` and `LIFNeuron` from `actor_critic_SNN.py` into `snn.py`.
Rename `SNNMLPActor` to `SNNActor`. Its constructor arguments, in order, are
`input_dim`, `hidden_dims`, `output_dim`, `num_snn_steps=4`,
`lif_beta=0.9`, `lif_threshold=1.0`, `surrogate_alpha=5.0`,
`reset_mode="subtract"`, and `input_scale=1.0`.

Keep `input_dim`, `output_dim`, `hidden_dims`, `last_spike_rates`, input-width
validation, leading-dimension flatten/restore, zero membrane initialization per
call, and continuous averaged output exactly as in the current implementation.

- [ ] **Step 5: Run the tests and verify GREEN**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_actors -v
```

Expected: all `StatelessActorsTest` tests pass.

- [ ] **Step 6: Commit the standalone actors**

```bash
git add rsl_rl/rsl_rl/modules/actors.py \
  rsl_rl/rsl_rl/modules/snn.py \
  rsl_rl/tests/test_actors.py
git commit -m "refactor: add reusable mlp and snn actors"
```

### Task 2: Consolidated Reservoir Dynamics

**Files:**

- Create: `rsl_rl/rsl_rl/modules/reservoirs.py`
- Create: `rsl_rl/tests/test_reservoirs.py`

- [ ] **Step 1: Write failing tests using the new module and parameter names**

```python
import unittest

import torch

from rsl_rl.modules.reservoirs import AnalogReservoir, LIFReservoir


class ReservoirsTest(unittest.TestCase):
    def test_analog_state_is_the_readout_feature(self):
        reservoir = AnalogReservoir(
            input_dim=5,
            reservoir_dim=8,
            connectivity=1.0,
            activation="relu",
        )
        state = reservoir.update_state(
            torch.randn(3, 5),
            torch.zeros(3, 8),
        )
        self.assertEqual(state.shape, (3, 8))
        self.assertTrue((state >= 0.0).all())
        self.assertTrue(torch.equal(
            reservoir.features_from_state(state),
            state,
        ))

    def test_lif_state_packs_membrane_and_binary_spikes(self):
        reservoir = LIFReservoir(
            input_dim=5,
            reservoir_dim=8,
            connectivity=1.0,
            lif_threshold=0.1,
        )
        state = reservoir.update_state(
            torch.randn(3, 5),
            torch.zeros(3, 16),
        )
        _, spikes = state.chunk(2, dim=-1)
        self.assertEqual(reservoir.state_dim, 16)
        self.assertEqual(reservoir.feature_dim, 8)
        self.assertTrue(torch.logical_or(
            spikes == 0.0,
            spikes == 1.0,
        ).all())
        self.assertTrue(torch.equal(
            reservoir.features_from_state(state),
            spikes,
        ))

    def test_random_weights_are_buffers(self):
        reservoir = AnalogReservoir(5, 8, connectivity=1.0)
        self.assertNotIn("w_in", dict(reservoir.named_parameters()))
        self.assertIn("w_in", dict(reservoir.named_buffers()))
        self.assertIn("w_res", dict(reservoir.named_buffers()))
        self.assertIn("reservoir_bias", dict(reservoir.named_buffers()))

    def test_trainable_reservoir_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train_reservoir=True"):
            LIFReservoir(5, 8, train_reservoir=True)
```

Also carry over the current tests for invalid observation/state shapes,
connectivity, spectral radius, leak rate, activation, LIF beta/threshold, and
reset mode.

- [ ] **Step 2: Run the tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_reservoirs -v
```

Expected: `ModuleNotFoundError` for `rsl_rl.modules.reservoirs`.

- [ ] **Step 3: Move the fixed reservoir base and Analog reservoir**

Move `_FixedReservoirBase` and `AnalogReservoir` from
`reservoir_dynamics.py` to `reservoirs.py`. Replace the old activation import
with:

```python
from .actors import get_activation
```

Preserve sparse initialization, zero diagonal, fan-in scaling, exact spectral
radius scaling, buffer registration, validation, detached state updates, and
the `input_dim`, `reservoir_dim`, `state_dim`, and `feature_dim` attributes.

- [ ] **Step 4: Move `LIFReservoir` onto the shared SNN primitive**

Use:

```python
from .snn import LIFNeuron
```

Retain the packed `[membrane, spike]` state and recurrent drive from the
previous spike:

```python
membrane, spikes = states.chunk(2, dim=-1)
for _ in range(self.num_reservoir_steps):
    recurrent_drive = F.linear(spikes, self.w_res)
    spikes, membrane = self.lif_neuron(
        input_drive + recurrent_drive,
        membrane,
    )
return torch.cat((membrane.detach(), spikes.detach()), dim=-1)
```

- [ ] **Step 5: Run the tests and verify GREEN**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_reservoirs -v
```

Expected: all reservoir tests pass.

- [ ] **Step 6: Commit the reservoir components**

```bash
git add rsl_rl/rsl_rl/modules/reservoirs.py \
  rsl_rl/tests/test_reservoirs.py
git commit -m "refactor: consolidate reservoir dynamics"
```

### Task 3: Four Explicit Reservoir Readout Actors

**Files:**

- Create: `rsl_rl/rsl_rl/modules/reservoir_actors.py`
- Create: `rsl_rl/tests/test_reservoir_actors.py`

- [ ] **Step 1: Write the failing four-class composition matrix**

```python
import unittest

import torch

from rsl_rl.modules.actors import MLPActor
from rsl_rl.modules.reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
)
from rsl_rl.modules.reservoirs import AnalogReservoir, LIFReservoir
from rsl_rl.modules.snn import SNNActor


CASES = (
    (AnalogReservoirMLPReadoutActor, AnalogReservoir, MLPActor, 8),
    (AnalogReservoirSNNReadoutActor, AnalogReservoir, SNNActor, 8),
    (LIFReservoirMLPReadoutActor, LIFReservoir, MLPActor, 16),
    (LIFReservoirSNNReadoutActor, LIFReservoir, SNNActor, 16),
)


class ReservoirActorsTest(unittest.TestCase):
    def test_all_public_combinations_replay_saved_state(self):
        observations = torch.randn(2, 3, 5)
        for actor_class, reservoir_class, readout_class, state_dim in CASES:
            with self.subTest(actor=actor_class.__name__):
                actor = actor_class(
                    input_dim=5,
                    output_dim=2,
                    reservoir_dim=8,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=[6, 4],
                    reservoir_lif_threshold=0.1,
                    snn_lif_threshold=0.1,
                    num_snn_steps=3,
                )
                actions, states = actor(
                    observations,
                    torch.zeros(2, 3, state_dim),
                )
                replay = actor.readout_process(
                    observations.reshape(-1, 5),
                    states.reshape(-1, state_dim),
                ).reshape(2, 3, 2)

                self.assertIsInstance(actor.reservoir, reservoir_class)
                self.assertIsInstance(actor.readout, readout_class)
                self.assertEqual(actions.shape, (2, 3, 2))
                self.assertEqual(states.shape, (2, 3, state_dim))
                self.assertTrue(torch.equal(actions, replay))
```

Add tests for `include_input_in_readout=True`, zero-state creation when state
is `None`, invalid state width, and detached reservoir buffers.

- [ ] **Step 2: Run the tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_reservoir_actors -v
```

Expected: import failure because `reservoir_actors.py` does not exist.

- [ ] **Step 3: Implement the private composition base**

Define `_ReservoirReadoutActor(nn.Module)` with:

```python
is_reservoir = True

def __init__(self, reservoir, readout, output_dim, include_input_in_readout):
    super().__init__()
    expected_width = reservoir.feature_dim
    if include_input_in_readout:
        expected_width += reservoir.input_dim
    if readout.input_dim != expected_width:
        raise ValueError(
            f"Expected readout input_dim {expected_width}, "
            f"received {readout.input_dim}."
        )
    self.reservoir = reservoir
    self.readout = readout
    self.input_dim = reservoir.input_dim
    self.output_dim = output_dim
    self.reservoir_state_dim = reservoir.state_dim
    self.include_input_in_readout = include_input_in_readout
```

Move the current flatten/update/readout/reshape logic from
`ReservoirReadoutActor`, but keep the new base private. `readout_process` must
extract features without updating reservoir state.

- [ ] **Step 4: Implement four public construction classes**

Each public class has the same configuration signature. Use one internal
builder to select `AnalogReservoir` or `LIFReservoir` and one to select
`MLPActor` or `SNNActor`. The effective readout width is:

```python
readout_input_dim = reservoir.feature_dim + (
    input_dim if include_input_in_readout else 0
)
```

The four public classes must be real named classes:

```python
class AnalogReservoirMLPReadoutActor(_ReservoirReadoutActor):
    reservoir_class = AnalogReservoir
    readout_class = MLPActor


class AnalogReservoirSNNReadoutActor(_ReservoirReadoutActor):
    reservoir_class = AnalogReservoir
    readout_class = SNNActor


class LIFReservoirMLPReadoutActor(_ReservoirReadoutActor):
    reservoir_class = LIFReservoir
    readout_class = MLPActor


class LIFReservoirSNNReadoutActor(_ReservoirReadoutActor):
    reservoir_class = LIFReservoir
    readout_class = SNNActor
```

Route `reservoir_lif_*` only to `LIFReservoir`, and route `snn_*` only to an
`SNNActor` readout. Do not accept `train_reservoir=True`.

- [ ] **Step 5: Run the tests and verify GREEN**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_reservoir_actors -v
```

Expected: all composition matrix and validation tests pass.

- [ ] **Step 6: Commit the four reservoir actors**

```bash
git add rsl_rl/rsl_rl/modules/reservoir_actors.py \
  rsl_rl/tests/test_reservoir_actors.py
git commit -m "refactor: add explicit reservoir readout actors"
```

### Task 4: One ActorCritic Factory for Six Actors

**Files:**

- Modify: `rsl_rl/rsl_rl/modules/actor_critic.py`
- Create: `rsl_rl/tests/test_actor_critic_factory.py`

- [ ] **Step 1: Write failing six-way selection and critic tests**

```python
import unittest

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
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


class ActorCriticFactoryTest(unittest.TestCase):
    def test_registry_selects_all_six_actor_classes(self):
        for actor_type, expected_class in ACTOR_TYPES.items():
            with self.subTest(actor_type=actor_type):
                model = ActorCritic(
                    5,
                    7,
                    2,
                    actor_type=actor_type,
                    actor_hidden_dims=[8, 4],
                    critic_hidden_dims=[8, 4],
                    reservoir_dim=8,
                    reservoir_connectivity=1.0,
                    readout_hidden_dims=[6, 4],
                )
                self.assertIsInstance(model.actor, expected_class)
                linear_layers = [
                    module for module in model.critic
                    if isinstance(module, nn.Linear)
                ]
                self.assertEqual(
                    [(layer.in_features, layer.out_features)
                     for layer in linear_layers],
                    [(7, 8), (8, 4), (4, 1)],
                )

    def test_invalid_actor_type_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "actor_type"):
            ActorCritic(5, 5, 2, actor_type="unknown")
```

Add tests that `mlp` is the default, MLP/SNN `act()` returns a sampled tensor,
reservoir `act()` returns `(sample, state)`, `act_for_ppo_update()` exactly
replays the saved-state mean, and `evaluate()` always returns `[batch, 1]`.

- [ ] **Step 2: Run the factory tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_actor_critic_factory -v
```

Expected: failure because `ActorCritic` does not accept `actor_type` and the
new Actor modules are not integrated.

- [ ] **Step 3: Replace Actor construction with an explicit registry**

Import the six Actor classes directly and define:

```python
ACTOR_REGISTRY = {
    "mlp": MLPActor,
    "snn": SNNActor,
    "analog_reservoir_mlp": AnalogReservoirMLPReadoutActor,
    "analog_reservoir_snn": AnalogReservoirSNNReadoutActor,
    "lif_reservoir_mlp": LIFReservoirMLPReadoutActor,
    "lif_reservoir_snn": LIFReservoirSNNReadoutActor,
}
```

Add explicit constructor parameters for the configuration names in the design.
Use `actor_hidden_dims` for standalone actors and `readout_hidden_dims` for
reservoir readouts. After the selected Actor has been constructed, set:

```python
self.actor_type = actor_type
self.actor = actor
self.is_reservoir = getattr(self.actor, "is_reservoir", False)
```

Reject unknown `actor_type` before building the critic or distribution. Keep
`get_activation` import-compatible by importing it from `.actors` in this
module.

- [ ] **Step 4: Preserve the original critic and Gaussian distribution**

Keep `self.critic` as the original `nn.Sequential` topology:

```python
critic_dims = [num_critic_obs, *critic_hidden_dims, 1]
critic_layers = []
for index, (source, target) in enumerate(
    zip(critic_dims[:-1], critic_dims[1:])
):
    critic_layers.append(nn.Linear(source, target))
    if index < len(critic_dims) - 2:
        critic_layers.append(get_activation(activation))
self.critic = nn.Sequential(*critic_layers)
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
```

Retain `action_mean`, `action_std`, `entropy`, log-probability, sampling, and
`evaluate()` behavior.

- [ ] **Step 5: Add state-aware delegation without wrapper subclasses**

Use the selected actor to implement both paths:

```python
def update_distribution(self, observations, reservoir_states=None):
    if self.is_reservoir:
        if reservoir_states is None:
            raise ValueError("reservoir_states are required.")
        mean, reservoir_states = self.actor(
            observations,
            reservoir_states,
        )
    else:
        mean = self.actor(observations)
    self.distribution = Normal(mean, mean * 0.0 + self.std)
    return reservoir_states

def act(self, observations, reservoir_states=None, **kwargs):
    next_states = self.update_distribution(
        observations,
        reservoir_states,
    )
    actions = self.distribution.sample()
    if self.is_reservoir:
        return actions.detach(), next_states
    return actions

def act_for_ppo_update(self, observations, reservoir_states):
    if not self.is_reservoir:
        raise RuntimeError("act_for_ppo_update requires a reservoir actor.")
    mean = self.actor.readout_process(observations, reservoir_states)
    self.distribution = Normal(mean, mean * 0.0 + self.std)
    return self.distribution.sample()
```

Mirror this distinction in `act_inference()`.

- [ ] **Step 6: Run factory tests and verify GREEN**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest tests.test_actor_critic_factory -v
```

Expected: all six factory, critic, distribution, and state-path tests pass.

- [ ] **Step 7: Commit the unified policy**

```bash
git add rsl_rl/rsl_rl/modules/actor_critic.py \
  rsl_rl/tests/test_actor_critic_factory.py
git commit -m "refactor: select six actors from one actor critic"
```

### Task 5: PPOtwin, Runner, and Checkpoint Integration

**Files:**

- Modify: `rsl_rl/rsl_rl/algorithms/ppo.py`
- Modify: `rsl_rl/rsl_rl/runners/twin_policy_runner.py`
- Modify: `rsl_rl/tests/test_actor_critic_factory.py`
- Modify: `rsl_rl/tests/test_twin_policy_runner.py`

- [ ] **Step 1: Add failing PPO/PPOtwin integration tests**

For each entry in `ACTOR_TYPES`, run one `PPOtwin` transition and update:

```python
algorithm = PPOtwin(
    model,
    num_learning_epochs=1,
    num_mini_batches=1,
    device="cpu",
)
algorithm.init_storage(4, 1, [5], [None], [2])
env_ids = torch.arange(4)
step_ids = torch.zeros(4, dtype=torch.long)
observations = torch.randn(4, 5)

if model.is_reservoir:
    states = torch.zeros(
        4,
        model.actor.reservoir_state_dim,
    )
    actions, states = algorithm.act(
        env_ids, step_ids, observations, observations, states
    )
else:
    actions = algorithm.act(
        env_ids, step_ids, observations, observations
    )

algorithm.process_env_step(
    1,
    env_ids,
    torch.tensor([1.0, 2.0, 3.0, 4.0]),
    torch.zeros(4, dtype=torch.bool),
    observations,
    {},
)
algorithm.compute_returns(observations)
losses = algorithm.update()
self.assertEqual(actions.shape, (4, 2))
self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
```

For reservoir actors, assert a readout parameter changes while `w_in` and
`w_res` remain buffers with no gradient.

- [ ] **Step 2: Add failing ordinary-PPO rejection and checkpoint tests**

```python
def test_ordinary_ppo_rejects_reservoir_action_path(self):
    model = self.make_model("analog_reservoir_mlp")
    algorithm = PPO(model, device="cpu")
    with self.assertRaisesRegex(RuntimeError, "PPOtwin"):
        algorithm.act(torch.randn(2, 5), torch.randn(2, 5))
```

For each actor type, save `model.state_dict()`, reconstruct the same
`ActorCritic`, and load with `strict=True`.

- [ ] **Step 3: Add a failing unified runner allocation test**

Configure only the common policy class and actor type:

```python
train_cfg = {
    "runner": {
        "policy_class_name": "ActorCritic",
        "algorithm_class_name": "PPOtwin",
        "num_steps_per_env": 1,
        "save_interval": 1,
    },
    "policy": {
        "actor_type": "lif_reservoir_mlp",
        "actor_hidden_dims": [8, 4],
        "critic_hidden_dims": [8, 4],
        "reservoir_dim": 8,
        "reservoir_connectivity": 1.0,
        "readout_hidden_dims": [6, 4],
    },
    "algorithm": {
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
    },
}
runner = TwinPolicyRunner(FakeEnvironment(), train_cfg, device="cpu")
self.assertEqual(runner.mu_reservoir_states.shape, (4, 16))
self.assertEqual(runner.omega_reservoir_states.shape, (4, 16))
```

Repeat with `analog_reservoir_mlp` and assert width `8`.

- [ ] **Step 4: Run integration tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest \
  tests.test_actor_critic_factory \
  tests.test_twin_policy_runner -v
```

Expected: runner import/allocation or ordinary-PPO rejection tests fail.

- [ ] **Step 5: Add the ordinary PPO guard**

At the start of `PPO.act()` add:

```python
if getattr(self.actor_critic, "is_reservoir", False):
    raise RuntimeError(
        "Reservoir actors require PPOtwin so rollout states can be stored."
    )
```

Do not put this check in `PPO.__init__`, because `PPOtwin.__init__` delegates
to it.

- [ ] **Step 6: Simplify TwinPolicyRunner policy lookup and allocation**

Replace imports of all deleted Actor-Critic wrappers with only the policy
classes still used by the runner:

```python
from rsl_rl.modules import ActorCritic, ActorCriticTwin
```

Allocate state directly from the selected actor:

```python
if self.mu_alg.actor_critic.is_reservoir:
    state_dim = self.mu_alg.actor_critic.actor.reservoir_state_dim
    self.mu_reservoir_states = torch.zeros(
        self.env.num_envs,
        state_dim,
        device=self.device,
    )
```

Apply the same logic to omega and remove the
`get_reservoir_state_dim(actor_critic, default_dim)` fallback and its
`policy_cfg["reservoir_dim"]` argument. Preserve selective environment updates
and done-state zeroing.

- [ ] **Step 7: Run integration tests and verify GREEN**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest \
  tests.test_actor_critic_factory \
  tests.test_twin_policy_runner -v
```

Expected: all factory, PPO/PPOtwin, checkpoint, and runner tests pass.

- [ ] **Step 8: Commit integration changes**

```bash
git add rsl_rl/rsl_rl/algorithms/ppo.py \
  rsl_rl/rsl_rl/runners/twin_policy_runner.py \
  rsl_rl/tests/test_actor_critic_factory.py \
  rsl_rl/tests/test_twin_policy_runner.py
git commit -m "refactor: integrate actor factory with ppo twin"
```

### Task 6: Public API Cutover and Legacy Deletion

**Files:**

- Modify: `rsl_rl/rsl_rl/modules/__init__.py`
- Modify: `rsl_rl/rsl_rl/modules/actor_critic_twin.py`
- Delete all legacy modules and tests listed in the File Map.
- Test: `rsl_rl/tests/test_actors.py`
- Test: `rsl_rl/tests/test_reservoirs.py`
- Test: `rsl_rl/tests/test_reservoir_actors.py`

- [ ] **Step 1: Add failing public API assertions**

Add:

```python
import rsl_rl.modules as modules

self.assertIs(modules.MLPActor, MLPActor)
self.assertIs(modules.SNNActor, SNNActor)
self.assertIs(modules.AnalogReservoir, AnalogReservoir)
self.assertIs(modules.LIFReservoir, LIFReservoir)
```

In the reservoir actor tests, assert all four public classes are exported.
Also assert removed compatibility names are absent:

```python
for removed_name in (
    "ActorCriticReservoir",
    "ActorCriticSNNReservoir",
    "ActorCriticSNN",
    "MLPReadout",
    "SNNReadout",
    "ReservoirReadoutActor",
):
    self.assertFalse(hasattr(modules, removed_name))
```

- [ ] **Step 2: Run public API tests and verify RED**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest \
  tests.test_actors \
  tests.test_reservoirs \
  tests.test_reservoir_actors -v
```

Expected: new exports are absent and legacy exports remain present.

- [ ] **Step 3: Replace module exports**

Keep existing unrelated exports and add:

```python
from .actor_critic import ActorCritic, get_activation
from .actors import MLPActor
from .snn import LIFNeuron, SNNActor, SurrogateSpike
from .reservoirs import AnalogReservoir, LIFReservoir
from .reservoir_actors import (
    AnalogReservoirMLPReadoutActor,
    AnalogReservoirSNNReadoutActor,
    LIFReservoirMLPReadoutActor,
    LIFReservoirSNNReadoutActor,
)
```

Remove every import of a deleted compatibility class.

- [ ] **Step 4: Remove the remaining package-level import cycle**

Change `actor_critic_twin.py` from:

```python
from rsl_rl.modules import ActorCritic
```

to:

```python
from .actor_critic import ActorCritic
```

Preserve `ActorCriticTwin` methods and behavior.

- [ ] **Step 5: Delete legacy files and superseded tests**

Delete exactly the legacy module and test files listed in the File Map. Search
for stale imports:

```bash
cd /home/ubuntu/amortizedMPs
rg -n \
  "actor_critic_reservoir|actor_critic_snn_reservoir|actor_critic_SNN|reservoir_dynamics|reservoir_readouts|reservoir_readout_actor|ActorCriticSNN|ActorCriticReservoir|MLPReadout|SNNReadout" \
  rsl_rl/rsl_rl rsl_rl/tests
```

Expected: no matches.

- [ ] **Step 6: Run the public API and preserved-policy tests**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest \
  tests.test_actors \
  tests.test_reservoirs \
  tests.test_reservoir_actors \
  tests.test_actor_critic_twin -v
```

Expected: every discovered test passes.

- [ ] **Step 7: Commit the API cutover**

```bash
git add -A rsl_rl/rsl_rl/modules rsl_rl/tests
git commit -m "refactor: remove legacy reservoir policy hierarchy"
```

### Task 7: Full Verification and Requirement Audit

**Files:**

- Review: `docs/superpowers/specs/2026-07-04-actor-module-consolidation-design.md`
- Review all files changed by Tasks 1–6.

- [ ] **Step 1: Run the complete unit test suite**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass with no import errors or failures.

- [ ] **Step 2: Compile the package and tests**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python -m compileall -q rsl_rl tests
```

Expected: exit code `0` and no output.

- [ ] **Step 3: Prove the public selection matrix**

```bash
cd /home/ubuntu/amortizedMPs/rsl_rl
python - <<'PY'
from rsl_rl.modules import ActorCritic

actor_types = (
    "mlp",
    "snn",
    "analog_reservoir_mlp",
    "analog_reservoir_snn",
    "lif_reservoir_mlp",
    "lif_reservoir_snn",
)
for actor_type in actor_types:
    model = ActorCritic(
        5,
        5,
        2,
        actor_type=actor_type,
        actor_hidden_dims=[8, 4],
        critic_hidden_dims=[8, 4],
        reservoir_dim=8,
        reservoir_connectivity=1.0,
        readout_hidden_dims=[6, 4],
    )
    print(actor_type, type(model.actor).__name__, type(model.critic).__name__)
PY
```

Expected: six lines mapping to the six approved public Actor class names, with
`Sequential` as every critic type.

- [ ] **Step 4: Audit deleted names and module boundaries**

```bash
cd /home/ubuntu/amortizedMPs
rg -n \
  "ActorCriticReservoir|ActorCriticSNNReservoir|ActorCriticSNN|MLPReadout|SNNReadout|ReservoirReadoutActor" \
  rsl_rl/rsl_rl rsl_rl/tests
git diff --check
git status --short
```

Expected: the search has no matches, `git diff --check` has no output, and
status contains only intentional changes.

- [ ] **Step 5: Compare implementation against every design requirement**

Confirm explicitly:

- six public Actor classes exist;
- `MLPActor` and `SNNActor` are used directly as readouts;
- one `ActorCritic` selects actors via `actor_type`;
- all critics use the original MLP topology;
- fixed reservoirs are buffers and return detached state;
- only `PPOtwin` handles reservoir state;
- saved post-update state is used for PPO readout reevaluation;
- Twin, recurrent, and HDRL paths still import and pass tests;
- no legacy class, file, config alias, or checkpoint compatibility remains.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required source or test corrections:

```bash
git add -A rsl_rl
git commit -m "test: complete actor module consolidation verification"
```

If no corrections were required, do not create an empty commit.
