# Reservoir and Readout Combinations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent Analog/LIF reservoir components, independent MLP/SNN readouts, and four explicit Actor-Critic combinations without changing existing policy behavior.

**Architecture:** Fixed reservoir dynamics expose a common state/feature interface. Readouts consume reservoir features and produce continuous action means. A generic composite actor connects one reservoir and one readout, while four thin `ActorCriticReservoir` subclasses select the component types.

**Tech Stack:** Python 3.8, PyTorch 2.4.1, `unittest`, local `rsl_rl`

---

### Task 1: Independent Reservoir Dynamics

**Files:**
- Create: `rsl_rl/rsl_rl/modules/reservoir_dynamics.py`
- Create: `rsl_rl/tests/test_reservoir_dynamics.py`

- [ ] **Step 1: Write failing component tests**

Test `AnalogReservoir` with `tanh` and `relu` activations:

```python
reservoir = AnalogReservoir(
    input_dim=5,
    reservoir_dim=8,
    connectivity=1.0,
    activation="relu",
)
self.assertEqual(reservoir.state_dim, 8)
self.assertEqual(reservoir.feature_dim, 8)
next_states = reservoir.update_state(
    torch.randn(3, 5),
    torch.zeros(3, 8),
)
self.assertEqual(next_states.shape, (3, 8))
self.assertTrue((next_states >= 0.0).all())
```

Test `LIFReservoir`:

```python
reservoir = LIFReservoir(
    input_dim=5,
    reservoir_dim=8,
    connectivity=1.0,
)
self.assertEqual(reservoir.state_dim, 16)
self.assertEqual(reservoir.feature_dim, 8)
next_states = reservoir.update_state(
    torch.randn(3, 5),
    torch.zeros(3, 16),
)
_, spikes = next_states.chunk(2, dim=-1)
self.assertTrue(torch.logical_or(
    spikes == 0.0,
    spikes == 1.0,
).all())
self.assertTrue(torch.equal(
    reservoir.features_from_state(next_states),
    spikes,
))
```

Both tests assert `w_in`, `w_res`, and `reservoir_bias` are buffers without
gradients. Add validation tests for state widths, invalid activation, invalid
connectivity, invalid LIF parameters, and `train_reservoir=True`.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_reservoir_dynamics.py' -v
```

Expected: import failure because `reservoir_dynamics` does not exist.

- [ ] **Step 3: Implement the common fixed-weight base**

Create `_FixedReservoirBase(nn.Module)` with:

```python
self.input_dim = input_dim
self.reservoir_dim = reservoir_dim
self.feature_dim = reservoir_dim
self.num_reservoir_steps = num_reservoir_steps
self.register_buffer("w_in", w_in)
self.register_buffer("w_res", scaled_w_res)
self.register_buffer("reservoir_bias", reservoir_bias)
```

Reuse the current sparse Gaussian initialization, zero diagonal, expected
fan-in scaling, exact eigenvalue spectral-radius scaling, uniform input/bias
initialization, and `F.linear` input projection.

- [ ] **Step 4: Implement `AnalogReservoir`**

Set `state_dim = reservoir_dim`, create the configured activation, and
implement:

```python
for _ in range(self.num_reservoir_steps):
    candidate = self.activation(
        input_drive + F.linear(states, self.w_res)
    )
    states = (
        (1.0 - self.leak_rate) * states
        + self.leak_rate * candidate
    )
return states.detach()
```

`features_from_state` returns the state.

- [ ] **Step 5: Implement `LIFReservoir`**

Set `state_dim = 2 * reservoir_dim`, use the existing `LIFNeuron`, split
`[membrane, spike]`, apply recurrent current from the previous spikes, and
return the detached packed state. `features_from_state` returns the spike
half.

- [ ] **Step 6: Run tests and verify GREEN**

Run the command from Step 2. Expected: all reservoir tests pass.

### Task 2: Independent Readouts

**Files:**
- Create: `rsl_rl/rsl_rl/modules/reservoir_readouts.py`
- Create: `rsl_rl/tests/test_reservoir_readouts.py`

- [ ] **Step 1: Write failing readout tests**

Test `MLPReadout` output shape, nonzero gradients, and initialization:

```python
readout = MLPReadout(
    input_dim=8,
    output_dim=2,
    hidden_dims=[6, 4],
    activation="elu",
)
output = readout(torch.randn(3, 8))
self.assertEqual(output.shape, (3, 2))
output.sum().backward()
self.assertIsNotNone(readout.network[0].weight.grad)
```

Test `SNNReadout` is an `SNNMLPActor`, emits continuous outputs, provides
`last_spike_rates`, and propagates surrogate gradients.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_reservoir_readouts.py' -v
```

Expected: import failure because `reservoir_readouts` does not exist.

- [ ] **Step 3: Implement `MLPReadout`**

Build an `nn.Sequential` from `[input_dim, *hidden_dims, output_dim]`.
Initialize hidden layers orthogonally with gain `sqrt(2)`, the final layer
with gain `0.01`, and all biases to zero.

- [ ] **Step 4: Implement `SNNReadout`**

Subclass `SNNMLPActor` and forward the existing SNN configuration. Do not add
cross-decision state.

- [ ] **Step 5: Run tests and verify GREEN**

Run the command from Step 2. Expected: all readout tests pass.

### Task 3: Generic Composite Actor

**Files:**
- Create: `rsl_rl/rsl_rl/modules/reservoir_readout_actor.py`
- Create: `rsl_rl/tests/test_reservoir_readout_actor.py`

- [ ] **Step 1: Write failing composition tests**

Construct all four pairs from real components and verify:

```python
actor = ReservoirReadoutActor(
    reservoir=reservoir,
    readout=readout,
    output_dim=2,
    include_input_in_readout=False,
)
actions, next_states = actor(observations, states)
self.assertEqual(actions.shape, (2, 3, 2))
self.assertEqual(
    next_states.shape,
    (2, 3, reservoir.state_dim),
)
```

Verify `readout_process(observations, next_states)` exactly reproduces the
rollout action mean, invalid observation/state shapes raise `ValueError`, and
`include_input_in_readout=True` uses `input_dim + feature_dim`.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_reservoir_readout_actor.py' -v
```

Expected: import failure because the composite actor does not exist.

- [ ] **Step 3: Implement the composite**

Store the supplied modules and expose:

```python
self.input_dim = reservoir.input_dim
self.output_dim = output_dim
self.reservoir_dim = reservoir.reservoir_dim
self.reservoir_state_dim = reservoir.state_dim
```

`forward` validates shapes, creates zero state when omitted, flattens leading
dimensions, calls `reservoir.update_state`, extracts features, optionally
concatenates observations, calls the readout, and restores leading shapes.

`readout_process` performs only feature extraction and readout evaluation so
the inherited Reservoir PPO path can re-evaluate saved post-update states.

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all composition tests pass.

### Task 4: Four Actor-Critic Policies

**Files:**
- Create: `rsl_rl/rsl_rl/modules/actor_critic_reservoir_combinations.py`
- Modify: `rsl_rl/rsl_rl/modules/__init__.py`
- Modify: `rsl_rl/rsl_rl/runners/twin_policy_runner.py`
- Create: `rsl_rl/tests/test_actor_critic_reservoir_combinations.py`

- [ ] **Step 1: Write failing wrapper and export tests**

Import the four public classes. For each class, instantiate a small policy and
verify:

```python
self.assertIsInstance(model, ActorCriticReservoir)
self.assertTrue(model.is_reservoir)
self.assertEqual(actions.shape, (3, 2))
self.assertEqual(
    next_states.shape[-1],
    model.actor.reservoir_state_dim,
)
```

Verify each wrapper contains its expected reservoir and readout types, is
exported from `rsl_rl.modules`, and is visible in
`rsl_rl.runners.twin_policy_runner`.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_reservoir_combinations.py' -v
```

Expected: import failure because the wrapper module does not exist.

- [ ] **Step 3: Implement `ActorCriticComposedReservoir`**

Derive from `ActorCriticReservoir`, preserve the inherited critic,
distribution, `act`, `act_for_ppo_update`, and inference methods, then replace
`self.actor` with a `ReservoirReadoutActor`.

Use class attributes on four thin subclasses to select:

```python
reservoir_class = AnalogReservoir or LIFReservoir
readout_class = MLPReadout or SNNReadout
```

Accept the common configuration superset defined in the design. Build the
readout with input width:

```python
reservoir.feature_dim + (
    num_actor_obs if include_input_in_readout else 0
)
```

- [ ] **Step 4: Export all four policies**

Add the four imports to `rsl_rl.modules` and to `TwinPolicyRunner`'s existing
policy lookup namespace. Do not change current task configurations.

- [ ] **Step 5: Run tests and verify GREEN**

Run the command from Step 2. Expected: all wrapper tests pass.

### Task 5: PPO, Runner, Checkpoint, and Compatibility Matrix

**Files:**
- Modify: `rsl_rl/tests/test_actor_critic_reservoir_combinations.py`
- Existing tests: `rsl_rl/tests/test_actor_critic_reservoir.py`
- Existing tests: `rsl_rl/tests/test_actor_critic_snn_reservoir.py`
- Existing tests: `rsl_rl/tests/test_twin_policy_runner.py`

- [ ] **Step 1: Add parameterized combination integration tests**

For each of the four wrappers:

1. create `PPOtwin`;
2. allocate two-environment storage;
3. collect and complete one transition;
4. compute returns;
5. run one PPO update;
6. assert finite losses;
7. assert a readout parameter changed;
8. assert reservoir buffers have no gradients.

Also save and strictly reload each state dict.

- [ ] **Step 2: Add runner allocation tests**

Instantiate `TwinPolicyRunner` with fake environments for one Analog and one
LIF wrapper. Assert state widths are respectively `reservoir_dim` and
`2 * reservoir_dim`.

- [ ] **Step 3: Run targeted tests**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_reservoir_combinations.py' -v
```

Expected: all matrix tests pass.

- [ ] **Step 4: Run full verification**

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m compileall -q rsl_rl tests
git diff --check
```

Expected: all tests pass, compilation succeeds, and no whitespace errors are
reported.

### Task 6: Completion Audit

**Files:**
- Review all new component and wrapper modules
- Review: `docs/superpowers/specs/2026-07-04-reservoir-readout-combinations-design.md`

- [ ] **Step 1: Prove every explicit requirement**

Verify from source and tests that:

- Analog and LIF reservoirs are independent classes;
- activation is configurable for Analog;
- MLP and SNN readouts are independent classes;
- all four explicit combination wrappers exist;
- all wrappers inherit `ActorCriticReservoir`;
- existing policies and current task configuration remain unchanged;
- PPO re-evaluation uses saved post-update reservoir features;
- old and new checkpoint paths load strictly;
- the full test suite passes.

- [ ] **Step 2: Preserve user-owned worktree changes**

Run:

```bash
git status --short
git -C rsl_rl status --short
```

Do not commit the nested repository because it contains existing user-owned
modified and untracked files. Report task files separately from pre-existing
changes.
