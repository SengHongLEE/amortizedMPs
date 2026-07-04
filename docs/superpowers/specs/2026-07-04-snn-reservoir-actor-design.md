# SNN Reservoir Actor Design

## Goal

Add an actor that combines a fixed recurrent LIF reservoir with the existing
trainable, feed-forward `SNNMLPActor` readout. Preserve the current
`ActorCriticReservoir`, Twin PPO, rollout storage, runner, critic, and
checkpoint structures. Keep `ActorCriticSNN` available as the independent SNN
MLP actor for comparison.

## Scope

The implementation will:

- add a new `SNNReservoirActor` derived from `ReservoirActor`;
- add a new `ActorCriticSNNReservoir` derived from
  `ActorCriticReservoir`;
- reuse `SNNMLPActor` as the trainable readout;
- export `ActorCriticSNNReservoir` through `rsl_rl.modules`;
- make the Twin runner allocate the actor's declared reservoir-state width;
- add unit and integration tests for state handling, spikes, gradients,
  module export, and a complete PPO update.

The implementation will not:

- replace or remove `ActorCriticReservoir` or `ActorCriticSNN`;
- make the SNN readout recurrent across policy decisions;
- train the random reservoir weights;
- change the MLP critic, rewards, GAE, Twin scheduling, or checkpoint format;
- add a dependency on a neuromorphic hardware SDK.

## Architecture

Two policy variants share the same SNN MLP component:

```text
Independent SNN actor

observation
    -> SNNMLPActor
    -> continuous action mean
```

```text
SNN reservoir actor

observation
    -> fixed random recurrent LIF reservoir
    -> reservoir spikes
    -> SNNMLPActor
    -> continuous action mean
```

`ActorCriticSNN` remains the independent policy wrapper around
`SNNMLPActor`. `ActorCriticSNNReservoir` uses the same `SNNMLPActor` class as
its readout, so the independent and reservoir-backed experiments use the same
trainable SNN implementation.

## LIF Reservoir State

The LIF reservoir maintains two tensors per environment:

- membrane potential with shape `[reservoir_dim]`;
- the previous reservoir spike with shape `[reservoir_dim]`.

They are packed into one external state tensor with shape
`[2 * reservoir_dim]`:

```text
[membrane, spike]
```

This preserves the existing runner and storage convention of passing one
reservoir-state tensor. The state is carried across policy decisions and reset
only when the corresponding environment finishes an episode.

For each internal reservoir step:

```text
recurrent_current = W_res * previous_spike
membrane = beta * membrane + input_drive + recurrent_current
spike = H(membrane - threshold)
membrane = reset(membrane, spike)
```

The input and recurrent matrices remain fixed registered buffers. The
recurrent matrix retains the existing sparse initialization and spectral-radius
scaling. The reservoir returns a detached packed state because PPO does not
backpropagate through the random reservoir dynamics.

## SNN Readout

The readout is an instance of the existing `SNNMLPActor`.

Its input contains:

- the current reservoir spike;
- optionally the original observation when
  `include_input_in_readout=True`.

The readout membrane potentials start from zero for every policy decision and
persist only through its `num_snn_steps` internal timesteps. This is a
feed-forward SNN with temporal rate coding inside one inference call, not an
RSNN across decisions.

All SNN readout linear weights are trainable with the existing surrogate
gradient. The final layer remains continuous so it can parameterize the mean
of the existing Gaussian action distribution.

## PPO and State Data Flow

During rollout:

1. The runner selects environments whose Twin policy is due.
2. `SNNReservoirActor` receives observations and packed reservoir states.
3. The fixed LIF reservoir produces the new membrane and spike state.
4. `SNNMLPActor` converts the current spike features into an action mean.
5. The policy samples an action from the existing Gaussian distribution.
6. Rollout storage saves the packed post-update reservoir state used by the
   readout.

During PPO update:

1. Storage returns observations and the saved packed states in random
   minibatches.
2. The policy extracts the saved reservoir spikes.
3. The stateless SNN readout recomputes the current action distribution.
4. PPO updates the SNN readout, continuous output layer, critic, and action
   standard deviation.

Because the reservoir is fixed and the SNN readout has no cross-decision
state, this reconstruction does not require sequence minibatches or BPTT.

## Integration

`ActorCriticSNNReservoir` will preserve the public reservoir-policy methods:

- `act(observations, reservoir_states)`;
- `act_for_ppo_update(observations, reservoir_states)`;
- `update_distribution(observations, reservoir_states)`;
- `act_inference(observations, reservoir_states)`.

The Twin runner will obtain the actual state width from
`actor.reservoir_state_dim`, falling back to the configured `reservoir_dim`
for compatibility. Existing `ActorCriticReservoir` continues to declare a
state width equal to `reservoir_dim`; the SNN reservoir declares
`2 * reservoir_dim`.

The new policy becomes selectable through:

```python
policy_class_name = "ActorCriticSNNReservoir"
```

No existing task configuration will be switched automatically.

## Configuration

The new wrapper accepts the existing reservoir and SNN parameters:

- `reservoir_dim`;
- `reservoir_connectivity`;
- `spectral_radius`;
- `input_scale`;
- `reservoir_bias_scale`;
- `num_reservoir_steps`;
- `lif_beta`;
- `lif_threshold`;
- `reset_mode`;
- `readout_hidden_dims`;
- `num_snn_steps`;
- `surrogate_alpha`;
- `snn_input_scale`;
- `include_input_in_readout`.

`train_reservoir=True` remains unsupported. Only the SNN readout and the
existing actor-critic trainable parameters are optimized.

## Spike Statistics

`SNNMLPActor` will expose the most recent mean spike rate for each hidden
layer. The statistic is detached diagnostic data and does not alter the
forward output or gradients.

The statistic supports comparisons of spike sparsity between the independent
SNN actor and the reservoir-backed SNN readout. It is not presented as a
hardware power measurement; actual energy depends on deployment hardware,
encoding, routing, memory traffic, and compiler mapping.

## Validation

Tests will verify:

- LIF reservoir membrane and spike state shapes;
- reservoir spikes are binary;
- arbitrary observation leading dimensions are preserved;
- invalid packed-state shapes are rejected;
- the SNN readout is an instance of `SNNMLPActor`;
- independent and reservoir-backed policies both produce continuous action
  means;
- PPO gradients reach the SNN readout but not fixed reservoir buffers;
- module export and policy class lookup work;
- the Twin runner allocates `2 * reservoir_dim` state width;
- one complete `PPOtwin` rollout, return calculation, and update produces
  finite losses and changes SNN readout weights;
- all existing tests continue to pass.
