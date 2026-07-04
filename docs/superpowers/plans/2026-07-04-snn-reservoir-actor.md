# SNN Reservoir Actor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed recurrent LIF reservoir with the existing trainable, feed-forward `SNNMLPActor` readout while preserving the current Reservoir/Twin PPO execution structure.

**Architecture:** `SNNReservoirActor` derives from `ReservoirActor`, packs membrane and spike tensors into one external state, and replaces the inherited MLP readout with `SNNMLPActor`. `ActorCriticSNNReservoir` derives from `ActorCriticReservoir`, so rollout and PPO update APIs remain unchanged. The Twin runner reads `reservoir_state_dim` from the actor and continues passing a single state tensor.

**Tech Stack:** Python 3.8, PyTorch 2.4.1, `unittest`, local `rsl_rl`

---

### Task 1: Declare Reservoir State Width

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/actor_critic_reservoir.py`
- Modify: `rsl_rl/rsl_rl/runners/twin_policy_runner.py`
- Test: `rsl_rl/tests/test_actor_critic_reservoir.py`
- Test: `rsl_rl/tests/test_twin_policy_runner.py`

- [ ] **Step 1: Write failing tests**

Add an assertion that a normal reservoir exposes its state width:

```python
self.assertEqual(actor.reservoir_state_dim, 8)
```

Add a runner helper test:

```python
def test_get_reservoir_state_dim_prefers_actor_declaration(self):
    actor_critic = SimpleNamespace(
        actor=SimpleNamespace(reservoir_state_dim=16),
    )
    self.assertEqual(
        TwinPolicyRunner.get_reservoir_state_dim(actor_critic, 8),
        16,
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_reservoir.py' -v
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_twin_policy_runner.py' -v
```

Expected: failures because `reservoir_state_dim` and
`get_reservoir_state_dim` do not exist.

- [ ] **Step 3: Implement the state-width contract**

In `ReservoirActor.__init__`:

```python
self.reservoir_state_dim = reservoir_dim
```

Use `self.reservoir_state_dim` for the expected external state shape and
reshape in `ReservoirActor.forward`.

Add to `TwinPolicyRunner`:

```python
@staticmethod
def get_reservoir_state_dim(actor_critic, default_dim):
    return getattr(
        actor_critic.actor,
        "reservoir_state_dim",
        default_dim,
    )
```

Allocate `mu_reservoir_states` and `omega_reservoir_states` with this helper.

- [ ] **Step 4: Run tests and verify GREEN**

Run the two commands from Step 2. Expected: all tests pass.

### Task 2: Expose Feed-Forward SNN Spike Statistics

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/actor_critic_SNN.py`
- Create: `rsl_rl/tests/test_actor_critic_snn.py`

- [ ] **Step 1: Write failing tests**

Create tests that instantiate `SNNMLPActor`, run a forward pass, and verify:

```python
self.assertEqual(actions.shape, (3, 2))
self.assertEqual(len(actor.last_spike_rates), 2)
for spike_rate in actor.last_spike_rates:
    self.assertEqual(spike_rate.ndim, 0)
    self.assertFalse(spike_rate.requires_grad)
    self.assertGreaterEqual(spike_rate.item(), 0.0)
    self.assertLessEqual(spike_rate.item(), 1.0)
```

Also verify a second forward call replaces rather than appends statistics.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_snn.py' -v
```

Expected: failure because `last_spike_rates` does not exist.

- [ ] **Step 3: Record detached spike rates**

Initialize:

```python
self.last_spike_rates = []
```

Inside `forward`, accumulate the detached mean spike rate for each hidden
layer over internal SNN timesteps, then assign one scalar per layer:

```python
spike_sums = [
    observations.new_zeros(())
    for _ in self.hidden_dims
]
...
spike_sums[layer_index] += spike.detach().mean()
...
self.last_spike_rates = [
    spike_sum / float(self.num_snn_steps)
    for spike_sum in spike_sums
]
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 3: Implement the LIF Reservoir with SNN Readout

**Files:**
- Create: `rsl_rl/rsl_rl/modules/actor_critic_snn_reservoir.py`
- Create: `rsl_rl/tests/test_actor_critic_snn_reservoir.py`

- [ ] **Step 1: Write failing behavior tests**

Test imports and instantiate:

```python
actor = SNNReservoirActor(
    input_dim=5,
    output_dim=2,
    reservoir_dim=8,
    readout_hidden_dims=[6, 4],
    connectivity=1.0,
    num_reservoir_steps=2,
    num_snn_steps=3,
)
```

Verify:

```python
self.assertIsInstance(actor, ReservoirActor)
self.assertIsInstance(actor.readout, SNNMLPActor)
self.assertEqual(actor.reservoir_state_dim, 16)
```

Run a forward pass with observations `[2, 3, 5]` and state `[2, 3, 16]`.
Verify action shape `[2, 3, 2]`, state shape `[2, 3, 16]`, and binary spike
half:

```python
_, spikes = next_states.chunk(2, dim=-1)
self.assertTrue(torch.logical_or(
    spikes == 0.0,
    spikes == 1.0,
).all())
```

Verify an invalid state width raises `ValueError`. Verify
`readout_process(observations, packed_states)` produces the same output as
calling `actor.readout` with the extracted spike features.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_snn_reservoir.py' -v
```

Expected: import failure because the new module does not exist.

- [ ] **Step 3: Implement `SNNReservoirActor`**

Create a class derived from `ReservoirActor`. Call the parent constructor to
retain fixed random input/recurrent buffers and spectral scaling. Set:

```python
self.reservoir_state_dim = 2 * reservoir_dim
self.lif_neuron = LIFNeuron(
    beta=lif_beta,
    threshold=lif_threshold,
    surrogate_alpha=surrogate_alpha,
    reset_mode=reset_mode,
)
self.readout = SNNMLPActor(
    input_dim=readout_input_dim,
    hidden_dims=readout_hidden_dims,
    output_dim=output_dim,
    num_snn_steps=num_snn_steps,
    beta=readout_lif_beta,
    threshold=readout_lif_threshold,
    surrogate_alpha=surrogate_alpha,
    reset_mode=reset_mode,
    input_scale=snn_input_scale,
)
```

Override reservoir memory:

```python
membrane, spikes = reservoir_states.chunk(2, dim=-1)
for _ in range(self.num_reservoir_steps):
    recurrent_drive = F.linear(spikes, self.w_res)
    spikes, membrane = self.lif_neuron(
        input_drive + recurrent_drive,
        membrane,
    )
return torch.cat(
    [membrane.detach(), spikes.detach()],
    dim=-1,
)
```

Override readout processing to extract the spike half and optionally
concatenate observations before passing features to `SNNMLPActor`.

- [ ] **Step 4: Implement `ActorCriticSNNReservoir`**

Derive it from `ActorCriticReservoir`. Preserve the inherited distribution,
act, PPO-update, and inference methods. After parent initialization, replace
`self.actor` with `SNNReservoirActor` using the same reservoir parameters plus
the SNN parameters.

- [ ] **Step 5: Run tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 4: Export and Select the New Policy

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/__init__.py`
- Modify: `rsl_rl/rsl_rl/runners/twin_policy_runner.py`
- Test: `rsl_rl/tests/test_actor_critic_snn_reservoir.py`

- [ ] **Step 1: Write failing export test**

Test:

```python
from rsl_rl.modules import ActorCriticSNNReservoir

self.assertEqual(
    ActorCriticSNNReservoir.__name__,
    "ActorCriticSNNReservoir",
)
```

Also verify the runner module exposes the class used by its existing `eval`
lookup.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 3 test command. Expected: import failure from `rsl_rl.modules`.

- [ ] **Step 3: Add module and runner imports**

Add:

```python
from .actor_critic_snn_reservoir import ActorCriticSNNReservoir
```

to `rsl_rl.modules`, and import `ActorCriticSNNReservoir` in
`twin_policy_runner.py` alongside the existing policy classes.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 3 test command. Expected: all tests pass.

### Task 5: Verify PPO Gradient and End-to-End Update

**Files:**
- Modify: `rsl_rl/tests/test_actor_critic_snn_reservoir.py`

- [ ] **Step 1: Add gradient and PPO integration tests**

Instantiate `ActorCriticSNNReservoir` with small dimensions. Verify
`act_for_ppo_update` followed by backward gives gradients to the SNN readout
linear layers while `w_in` and `w_res` remain buffers without gradients.

Create `PPOtwin`, initialize two-environment storage, collect one transition,
finish it, compute returns, and update. Assert:

```python
self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
self.assertFalse(torch.equal(
    readout_weight_before,
    model.actor.readout.linear_layers[0].weight,
))
```

- [ ] **Step 2: Run tests and verify RED or meaningful initial failure**

Run the Task 3 test command. Confirm failures identify missing integration
behavior rather than test setup errors.

- [ ] **Step 3: Make the smallest integration corrections**

Correct only state extraction, actor construction, or runner allocation
needed for the tests. Do not modify rewards, GAE, critic behavior, or Twin
event scheduling.

- [ ] **Step 4: Run targeted and full verification**

Run:

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_actor_critic_snn_reservoir.py' -v
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
/home/ubuntu/anaconda3/envs/cpgil/bin/python -m compileall -q rsl_rl tests
git diff --check
```

Expected: all tests pass, compilation succeeds, and no whitespace errors are
reported.

### Task 6: Completion Audit

**Files:**
- Review all files listed above

- [ ] **Step 1: Verify explicit requirements**

Confirm from source and tests:

- the new script exists independently;
- it inherits the current Reservoir implementation;
- it reuses `SNNMLPActor`;
- `ActorCriticSNN` remains usable independently;
- existing rollout, PPO, critic, and checkpoint interfaces remain intact;
- code style matches neighboring modules;
- spike statistics are diagnostic only.

- [ ] **Step 2: Inspect repository status**

Run:

```bash
git status --short
git -C rsl_rl status --short
```

Report only files changed for this task and distinguish them from pre-existing
worktree changes. Do not commit the nested repository because its current
untracked and modified files contain user-owned work outside this task.
