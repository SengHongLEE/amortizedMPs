# Twin Policy Async Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize twin-policy asynchronous rollout storage and correct transition duration handling without forcing equal per-environment update frequencies.

**Architecture:** Keep separate flat event buffers for mu and omega. Replace per-env Python tensor lists with contiguous writes and stable-sort/padded segmented GAE, and centralize scalar-or-vector update period handling in the runner.

**Tech Stack:** Python 3.8, PyTorch 2.4, unittest/pytest-compatible tests.

---

### Task 1: Actor-Critic Compatibility

**Files:**
- Modify: `rsl_rl/rsl_rl/modules/actor_critic_twin.py`
- Create: `rsl_rl/tests/test_actor_critic_twin.py`

- [ ] Write a failing test asserting action/value shapes and the checkpoint keys `actor.*`, `critic.*`, and `std`.
- [ ] Run `/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest rsl_rl/tests/test_actor_critic_twin.py -v` and verify the dead twin API assertion fails.
- [ ] Replace duplicated construction and invalid mu/omega members with an `ActorCritic` constructor delegation.
- [ ] Re-run the targeted test and verify it passes.

### Task 2: Asynchronous Flat Storage

**Files:**
- Modify: `rsl_rl/rsl_rl/storage/rollout_storage_twin.py`
- Create: `rsl_rl/tests/test_rollout_storage_twin.py`

- [ ] Write failing tests for irregular env event counts, partial-capacity writes, endpoint mapping, `ready`, `clear`, and segmented GAE.
- [ ] Run `/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest rsl_rl/tests/test_rollout_storage_twin.py -v` and verify failures reference missing new storage behavior.
- [ ] Implement contiguous start writes, tracked endpoint completion, endpoint critic observations, and Python readiness counters.
- [ ] Implement stable-sort/padded segmented GAE and capacity-aware mini-batches.
- [ ] Re-run the storage tests and verify they pass.

### Task 3: PPO Duration and Bootstrap

**Files:**
- Modify: `rsl_rl/rsl_rl/algorithms/ppo_twin.py`
- Create: `rsl_rl/tests/test_ppo_twin.py`

- [ ] Write failing tests proving duration is read through `env_ptrs`, untracked events are ignored, and endpoint observations bootstrap returns.
- [ ] Run `/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest rsl_rl/tests/test_ppo_twin.py -v` and verify the duration test fails against the old flat index lookup.
- [ ] Implement filtered endpoint processing and one-pass endpoint bootstrap evaluation.
- [ ] Re-run the PPO tests and verify they pass.

### Task 4: Runner Integration

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/twin_policy_runner.py`
- Create: `rsl_rl/tests/test_twin_policy_runner.py`

- [ ] Write failing unit tests for scalar and per-env cycle tensors.
- [ ] Run `/home/ubuntu/anaconda3/envs/cpgil/bin/python -m unittest rsl_rl/tests/test_twin_policy_runner.py -v` and verify the helper is missing.
- [ ] Add centralized update-ID calculation, preallocated action assembly, endpoint critic propagation, readiness checks, exact stopping comparison, and checkpoint compatibility.
- [ ] Re-run the runner tests and verify they pass.

### Task 5: Verification

**Files:**
- Test: `rsl_rl/tests/test_actor_critic_twin.py`
- Test: `rsl_rl/tests/test_rollout_storage_twin.py`
- Test: `rsl_rl/tests/test_ppo_twin.py`
- Test: `rsl_rl/tests/test_twin_policy_runner.py`

- [ ] Run all four unittest files together.
- [ ] Compile the four modified production files with `py_compile`.
- [ ] Run a CPU minimal actor/storage/PPO update smoke test and check shapes and finite losses.
- [ ] Review `git diff --check` and the final diff for unrelated changes.
