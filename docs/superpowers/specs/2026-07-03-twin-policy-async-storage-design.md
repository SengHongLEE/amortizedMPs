# Twin Policy Async Storage Design

## Goal

Reduce `MP_twin_a1` training time while preserving independent per-environment
decision frequencies for the mu and omega policies.

## Constraints

- Each environment may use a different mu period and omega period.
- Mu and omega continue to own separate policies, optimizers, and rollout buffers.
- Existing configuration keys, policy state-dict keys, and runner entry points remain compatible.
- Default PPO hyperparameters and the current continuous asynchronous training
  semantics remain unchanged.

## Design

`TwinPolicyRunner` computes update IDs through one helper that supports scalar or
per-environment period tensors. It passes endpoint critic observations to
`PPOtwin`, uses a preallocated combined-action buffer, and checks a Python
`storage.ready` property instead of synchronizing on `storage_done.all()` every
simulation step.

`RolloutStorage_twin` remains an asynchronous flat event buffer. Starts are
written into preallocated contiguous slices and mapped to their environment by
`env_ptrs`; no per-environment tensor list is maintained. Endpoints complete
only tracked starts and save the endpoint critic observation needed to
bootstrap the last event from each environment.

For GAE, completed events are stable-sorted by environment. Their within-env
event ranks are computed with `bincount` and tensor operations, scattered into
a temporary padded event matrix, and processed with a loop over event depth.
The loop length is the largest number of events produced by one environment,
not `num_envs * events`.

`PPOtwin` calculates duration through the slot selected by `env_ptrs`, so
accumulated reward is divided by the correct positive duration. Its return
calculation evaluates endpoint critic observations only once per update.

`ActorCriticTwin` delegates its network construction to `ActorCritic`. This
removes broken dead mu/omega-specific members while preserving actor, critic,
and std parameter names used by existing checkpoints.

## Compatibility

Existing twin model checkpoints remain loadable because policy parameter names
do not change. New checkpoints save both optimizer states and separate mu/omega
iteration counters; loading keeps fallbacks for older checkpoints containing
only `iter`.

## Verification

CPU tests cover irregular per-env event counts, buffer overflow, lifecycle
reset, positive duration lookup, asynchronous GAE, actor state-dict
compatibility, and a minimal PPO update. A CUDA microbenchmark compares the old
per-env loop baseline with the new storage path when CUDA is available.
