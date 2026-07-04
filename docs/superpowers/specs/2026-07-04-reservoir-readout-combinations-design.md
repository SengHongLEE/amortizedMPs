# Reservoir and Readout Combinations Design

## Goal

Separate reservoir dynamics from readout networks and expose four explicit
Actor-Critic policy classes:

- `ActorCriticAnalogReservoirMLPReadout`;
- `ActorCriticAnalogReservoirSNNReadout`;
- `ActorCriticLIFReservoirMLPReadout`;
- `ActorCriticLIFReservoirSNNReadout`.

All four policies preserve the existing `ActorCriticReservoir` rollout, PPO,
critic, runner, storage, inference, and checkpoint-container interfaces.

## Compatibility

The existing classes remain available and retain their current behavior:

- `ReservoirActor`;
- `ActorCriticReservoir`;
- `SNNReservoirActor`;
- `ActorCriticSNNReservoir`.

The new combination framework is added alongside these classes. It does not
rewrite their module hierarchy or state-dict keys, so existing checkpoints and
task configurations continue to use the existing policy classes.

The new policies use a nested component hierarchy and therefore have their own
state-dict keys. They are not presented as strict state-dict replacements for
the existing policies.

## Component Structure

```text
Reservoir dynamics
├── AnalogReservoir
└── LIFReservoir

Readout networks
├── MLPReadout
└── SNNReadout

Composite actor
└── ReservoirReadoutActor(reservoir, readout)

Actor-Critic wrappers
├── ActorCriticAnalogReservoirMLPReadout
├── ActorCriticAnalogReservoirSNNReadout
├── ActorCriticLIFReservoirMLPReadout
└── ActorCriticLIFReservoirSNNReadout
```

Reservoir and readout components are independent `nn.Module` classes. The
composite actor owns one of each and handles observation validation, state
validation, optional direct observation features, flattening, and restoration
of leading tensor dimensions.

## Reservoir Interface

Both reservoirs expose:

```python
reservoir_dim
state_dim
feature_dim
update_state(observations, states)
features_from_state(states)
```

The reservoir receives flattened observations with shape
`[batch, input_dim]`. `update_state` returns a detached state because reservoir
weights remain fixed and PPO trains only the selected readout and the existing
Actor-Critic parameters.

### AnalogReservoir

`AnalogReservoir` implements the current leaky reservoir equation:

```text
candidate = activation(W_in * observation + W_res * state + bias)
state = (1 - leak_rate) * state + leak_rate * candidate
```

Its properties are:

```text
state_dim = reservoir_dim
feature_dim = reservoir_dim
features_from_state(state) = state
```

`reservoir_activation` remains configurable through the existing activation
names, including `tanh`, `relu`, `elu`, `selu`, `lrelu`, and `sigmoid`.

The input and sparse recurrent matrices use the same initialization and
spectral-radius scaling rules as the existing `ReservoirActor`.

### LIFReservoir

`LIFReservoir` stores:

```text
[membrane, spike]
```

and updates:

```text
recurrent_current = W_res * previous_spike
membrane = beta * membrane + W_in * observation + bias + recurrent_current
spike = H(membrane - threshold)
membrane = reset(membrane, spike)
```

Its properties are:

```text
state_dim = 2 * reservoir_dim
feature_dim = reservoir_dim
features_from_state(state) = spike
```

The spike tensor is binary. Reservoir membrane and spike state persist across
policy decisions and are reset by the existing runner at episode boundaries.

## Readout Interface

Both readouts implement:

```python
forward(features) -> continuous_action_mean
```

The composite actor determines the readout input width:

```text
reservoir.feature_dim
+ input_dim when include_input_in_readout=True
```

### MLPReadout

`MLPReadout` is a standard `nn.Sequential`-style MLP with configurable hidden
dimensions and activation. It preserves the current readout initialization:

- orthogonal hidden weights with gain `sqrt(2)`;
- orthogonal output weights with gain `0.01`;
- zero biases.

### SNNReadout

`SNNReadout` reuses the existing `SNNMLPActor` implementation. It is
feed-forward across policy decisions:

- membrane starts from zero for each policy call;
- membrane persists across `num_snn_steps` internal timesteps;
- hidden layers emit spikes with surrogate gradients;
- the final layer produces a continuous action mean;
- `last_spike_rates` remains available for sparsity diagnostics.

It does not introduce recurrent PPO state or sequence minibatches.

## Composite Actor

`ReservoirReadoutActor` provides the same public behavior expected by
`ActorCriticReservoir`:

```python
forward(observations, reservoir_states)
readout_process(observations, reservoir_states)
reservoir_state_dim
```

During rollout, `forward` updates the reservoir and evaluates the selected
readout. During PPO update, `readout_process` extracts features from the saved
post-update state and reevaluates only the trainable readout. This preserves
the current random-minibatch PPO logic because reservoir dynamics are fixed.

## Actor-Critic Wrappers

An internal `ActorCriticComposedReservoir` derives from
`ActorCriticReservoir`. It retains all inherited policy-distribution methods
and replaces only `self.actor` with `ReservoirReadoutActor`.

The four public wrappers select reservoir and readout component classes
without conditional branches in their forward paths.

All wrappers accept a common configuration superset so the same task
configuration can switch policy classes:

- standard actor, critic, activation, and action-noise parameters;
- analog reservoir leak rate and activation;
- LIF beta, threshold, and reset mode;
- reservoir size, connectivity, spectral radius, input scale, bias scale, and
  internal step count;
- readout hidden dimensions and activation;
- SNN timestep, beta, threshold, surrogate alpha, reset mode, and input scale;
- `include_input_in_readout`;
- `train_reservoir`, which must remain `False`.

## Policy Selection

Examples:

```python
policy_class_name = "ActorCriticLIFReservoirMLPReadout"
```

```python
policy_class_name = "ActorCriticAnalogReservoirSNNReadout"
reservoir_activation = "relu"
```

The classes are exported through `rsl_rl.modules` and imported into
`TwinPolicyRunner` so its existing policy-class lookup continues to work.

No existing task configuration is automatically changed.

## Error Handling

Construction rejects:

- non-positive input, output, or reservoir dimensions;
- connectivity outside `(0, 1]`;
- non-positive spectral radius;
- invalid leak rate, LIF beta, threshold, reset mode, or activation;
- empty readout hidden dimensions where required by `SNNReadout`;
- `train_reservoir=True`.

Forward calls reject observation or state shapes that do not match the
selected reservoir.

## Validation

Tests cover:

- the independent component interfaces;
- configurable Analog activation;
- Analog state and feature dimensions;
- LIF packed-state dimensions and binary spikes;
- MLP and SNN readout output dimensions and gradients;
- all four component pairings;
- correct readout component types;
- fixed reservoir buffers and trainable readout parameters;
- arbitrary observation leading dimensions;
- PPO rollout, stored-state reevaluation, return calculation, and update for
  all four wrappers;
- runner state allocation for Analog and LIF policies;
- strict checkpoint round trips for all four wrappers;
- unchanged behavior of existing Reservoir and SNN Reservoir policies;
- the complete existing test suite.
