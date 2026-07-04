# Actor Module Consolidation Design

## Goal

Replace the duplicated legacy and composed actor implementations with one
component-based actor stack. A single `ActorCritic` policy selects one of six
public actors through `actor_type`, while every policy continues to use the
original MLP critic and Gaussian action distribution.

The six public actors are:

- `MLPActor`;
- `SNNActor`;
- `AnalogReservoirMLPReadoutActor`;
- `AnalogReservoirSNNReadoutActor`;
- `LIFReservoirMLPReadoutActor`;
- `LIFReservoirSNNReadoutActor`.

`MLPActor` and `SNNActor` are both complete standalone actors and reusable
reservoir readouts.

## Scope

This redesign covers the feed-forward, SNN, and reservoir actor stack. Existing
Twin, recurrent, and hierarchical policy behavior remains in scope only where
imports or shared `ActorCritic` behavior must continue to work:

- `ActorCriticTwin`;
- `ActorCriticRecurrent`;
- `actor_critic_hdrl`.

Reservoir actors are required to work with `TwinPolicyRunner` and `PPOtwin`.
They are not required to work with the ordinary `OnPolicyRunner` and `PPO`.
Standalone MLP and SNN actors continue to work with ordinary PPO.

## Compatibility Boundary

Backward compatibility is intentionally not retained for:

- `ActorCriticReservoir`;
- `ActorCriticSNNReservoir`;
- `ActorCriticSNN`;
- `ReservoirActor`;
- `SNNReservoirActor`;
- `MLPReadout`;
- `SNNReadout`;
- legacy reservoir checkpoint state-dict keys;
- legacy policy class names and configurations.

Existing checkpoints using those classes must not be presented as compatible
with the new module hierarchy.

## Module Structure

```text
rsl_rl/rsl_rl/modules/
├── actor_critic.py
├── actors.py
├── snn.py
├── reservoirs.py
├── reservoir_actors.py
├── actor_critic_twin.py
├── actor_critic_recurrent.py
├── actor_critic_hdrl.py
└── __init__.py
```

Responsibilities:

- `actor_critic.py` owns the single configurable `ActorCritic`, the original
  MLP critic construction, Gaussian action distribution, and actor registry.
- `actors.py` owns `MLPActor`.
- `snn.py` owns `SurrogateSpike`, `LIFNeuron`, and `SNNActor`.
- `reservoirs.py` owns `AnalogReservoir` and `LIFReservoir`.
- `reservoir_actors.py` owns one internal composite implementation and four
  explicit public reservoir actor classes.
- `__init__.py` exports only the intended public policy and component classes.

The following duplicated files are removed:

- `actor_critic_reservoir.py`;
- `actor_critic_snn_reservoir.py`;
- `actor_critic_reservoir_combinations.py`;
- `actor_critic_SNN.py`;
- `reservoir_dynamics.py`;
- `reservoir_readouts.py`;
- `reservoir_readout_actor.py`.

## Common Actor Interface

Every public actor exposes:

```python
input_dim
output_dim
```

Stateless actors implement:

```python
forward(observations) -> action_mean
```

Reservoir actors additionally expose:

```python
is_reservoir = True
reservoir_state_dim
forward(observations, reservoir_states)
    -> (action_mean, updated_reservoir_states)
readout_process(observations, reservoir_states)
    -> action_mean
```

The internal `_ReservoirReadoutActor` composes one reservoir and one
`MLPActor` or `SNNActor`. The four public reservoir actors select concrete
component types without duplicating rollout or readout logic.

## Standalone Actors

### MLPActor

`MLPActor` builds a configurable MLP from `input_dim`, `hidden_dims`,
`output_dim`, and `activation`. It accepts arbitrary leading tensor dimensions
and preserves them in its output.

The same class is instantiated:

- from `num_actor_obs` to `num_actions` as a standalone actor;
- from reservoir readout feature width to `num_actions` as an MLP readout.

### SNNActor

`SNNActor` retains the current stateless-across-decisions SNN behavior:

- membrane state starts at zero for every call;
- the same input is presented for `num_snn_steps`;
- hidden LIF layers emit spikes with surrogate gradients;
- a continuous linear output is averaged across internal timesteps;
- spike-rate diagnostics remain available.

The same class is instantiated as either a standalone actor or a reservoir
readout. It does not add recurrent PPO state.

## Reservoir Components

Both reservoirs remain fixed random feature generators. Their input and
recurrent matrices are registered buffers, not trainable parameters.
`train_reservoir=True` remains unsupported because PPO minibatches reuse stored
post-update reservoir states.

### AnalogReservoir

The state update is:

```text
candidate = activation(W_in * observation + W_res * state + bias)
state = (1 - leak_rate) * state + leak_rate * candidate
```

Its state and readout feature widths both equal `reservoir_dim`.

### LIFReservoir

The persistent state packs:

```text
[membrane, spike]
```

Recurrent input is computed from the previous spike. Its state width is
`2 * reservoir_dim`, while its readout features contain only the spike half and
have width `reservoir_dim`.

Both reservoirs return detached updated state.

## Reservoir Actor Data Flow

During rollout:

```text
observation + previous reservoir state
                  |
                  v
       Analog/LIF reservoir update
                  |
                  v
        detached updated state
                  |
                  v
          MLPActor/SNNActor readout
                  |
                  v
        action mean + updated state
```

When `include_input_in_readout=True`, the original observation is concatenated
with reservoir features before the readout.

During `PPOtwin` update, `readout_process` uses the stored post-update state and
does not run reservoir dynamics again. This makes random minibatch
reevaluation consistent with rollout and trains only the readout, action
standard deviation, and critic.

`TwinPolicyRunner` allocates state using
`actor_critic.actor.reservoir_state_dim`, updates only selected environments,
and zeros state for completed environments.

## Unified ActorCritic

`ActorCritic` accepts `actor_type` and constructs its actor through an explicit
registry. Supported values are:

```text
mlp
snn
analog_reservoir_mlp
analog_reservoir_snn
lif_reservoir_mlp
lif_reservoir_snn
```

The default is `mlp`.

Example:

```python
policy_class_name = "ActorCritic"

actor_type = "lif_reservoir_snn"
reservoir_dim = 128
readout_hidden_dims = [64, 32]
num_reservoir_steps = 3
num_snn_steps = 4
```

The policy delegates action-mean computation to the selected actor while
retaining:

- the original MLP critic architecture;
- the trainable diagonal Gaussian standard deviation;
- action sampling, inference, log probability, and entropy behavior;
- `evaluate()` for critic observations.

For reservoir actors, it also provides the state-aware `act`,
`act_inference`, and `act_for_ppo_update` paths expected by `PPOtwin`.
For stateless actors, the ordinary PPO interfaces remain unchanged.

The ordinary PPO path must reject a reservoir actor with a clear error instead
of failing later because reservoir state was omitted.

## Configuration

Configuration names separate reservoir LIF parameters from SNN actor/readout
parameters.

Common policy parameters include:

- `actor_type`;
- `actor_hidden_dims`;
- `critic_hidden_dims`;
- `activation`;
- `init_noise_std`.

Reservoir parameters include:

- `reservoir_dim`;
- `reservoir_connectivity`;
- `spectral_radius`;
- `reservoir_input_scale`;
- `reservoir_bias_scale`;
- `num_reservoir_steps`;
- `leak_rate`;
- `reservoir_activation`;
- `reservoir_lif_beta`;
- `reservoir_lif_threshold`;
- `reservoir_surrogate_alpha`;
- `reservoir_reset_mode`.

Readout parameters include:

- `readout_hidden_dims`;
- `readout_activation`;
- `include_input_in_readout`.

SNN actor/readout parameters include:

- `num_snn_steps`;
- `snn_lif_beta`;
- `snn_lif_threshold`;
- `snn_surrogate_alpha`;
- `snn_reset_mode`;
- `snn_input_scale`.

`actor_hidden_dims` configures standalone MLP/SNN actors.
`readout_hidden_dims` configures MLP/SNN actors used as reservoir readouts.

Construction validates the selected actor type and relevant parameters.
Invalid dimensions, activations, reservoir settings, reset modes, and state
shapes raise explicit errors.

## Runner Selection

All six choices use:

```python
policy_class_name = "ActorCritic"
```

`actor_type` selects the actor. `TwinPolicyRunner` no longer imports every
combination-specific Actor-Critic wrapper. Existing unrelated policy classes
remain available for their existing call paths.

## Validation

Tests cover:

- direct construction and output shapes for all six public actors;
- gradients through all trainable actors and readouts;
- `MLPActor` and `SNNActor` used both standalone and as readouts;
- arbitrary leading tensor dimensions for stateless actors;
- fixed Analog and LIF reservoir buffers;
- Analog state updates and configurable activation;
- LIF packed-state dimensions and binary spikes;
- episode-boundary reservoir reset;
- all six `actor_type` registry selections through one `ActorCritic`;
- the original MLP critic structure for every actor selection;
- `PPOtwin` rollout and update for all six actors;
- stored-state reservoir readout reevaluation;
- strict checkpoint round trips for the new hierarchy;
- ordinary PPO behavior for standalone MLP and SNN actors;
- an explicit ordinary-PPO rejection for reservoir actors;
- unchanged Twin, recurrent, and hierarchical tests.

No test requires compatibility with removed classes, configurations, files, or
checkpoint keys.
