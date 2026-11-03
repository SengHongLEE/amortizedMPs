# Dynamics-only jump reflex validation

## Objective
Keep actor/critic observation shapes and existing four-gait action interface unchanged. Derive jump preparation only from internal dynamics state, including end-effector equilibrium state. At least one gait in the primitive library must support smooth run-up, takeoff, flight, landing, and gait continuation; the policy must learn to select a capable gait before obstacles. Prove paired hurdle and gap traversal improvement in an isolated pipeline before any `MP_full_a1` integration.

## Gates
1. Inspect existing trajectories and establish baseline plus false-trigger controls. Complete for seeds 123/321 at difficulty 0.75.
2. Implement an isolated detector and transient controller with test-first behavior checks. Complete for the present prototype.
3. Validate transitions, physical lift, landing, gait continuation, and false-trigger rate for at least one obstacle-capable gait. In progress; stronger support stroke helps trot gaps and canter obstacles. One late flat false arm was seen in the seed-777 canter control.
4. Train and validate policy gait selection from existing terrain observations, while the jump trigger consumes only internal dynamics state. In progress; an offline-distilled, opt-in readout-edge policy recognizes the holdout hurdle edges, but has not been jointly optimized for reliable traversal.
5. Run paired multi-seed hurdle, single-gap, and continuous-gap evaluation; tune only on a separate calibration set. In progress; gap-only landing-x modulation improved first-band gap totals in seeds 2026/2027, but aggregate gains over no-reflex are small and second-band traversal remains unproven.
6. Integrate into default `MP_full_a1` only if the full physical and traversal gates pass. Pending.

## Current gate decision
The opt-in task and evaluator work. A learned readout-edge selector picks canter before obstacle height edges from the unchanged actor observation. An opt-in gap landing target improves some first-band gaps, while second-band continuous gaps remain near 3/16. Training-only reward shaping can penalize off-target trigger candidates, but three controlled 30-iteration trials did not produce net reflex benefit; the corrected reward checkpoint scored 25/48 first-band with reflex versus 29/48 with reflex disabled. A harder hurdle probe found only 2/32 successes with reflex versus 0/32 without. Flat controls show zero actual false arms in the tested seeds. The reward defaults to disabled in the isolated task. Physical reliability and multi-seed generalization remain unproven. Do not integrate into default `MP_full_a1` yet.

## Non-negotiable constraints
- No actor or critic observation-space edits.
- No direct height-map or oracle obstacle-position input to jump preparation.
- Keep current ordinary gait phase running through jump and resume it smoothly.
- Record negative results; airborne events alone do not prove traversal improvement.

## Issues encountered
- `cpgil` interpreter alone could not compile Isaac Gym extension because `ninja` was absent from PATH; resolved with `PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH`.
- An older five-primitive checkpoint can output gait ID 4 against the current four-gait controller; baseline evaluator uses `--force-primitive 2` to make comparison executable.
