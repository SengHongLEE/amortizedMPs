# MP_full_a1 signed dense scan

Objective: preserve narrow positive hurdle signals and extend/densify terrain lookahead without unbounded policy observation growth; keep old checkpoint behavior available for paired evaluation.

- [complete] Inspect observation, terrain sampling, actor interfaces, and memory budget.
- [complete] Implement opt-in dense signed scan with bounded output dimensions and matching configuration validation.
- [complete] Connect compact terrain encoding to selector and modulator, including gradient path and inference consistency.
- [complete] Add focused tests for positive/negative obstacle preservation, tensor dimensions, and encoder gradients.
- [complete] Run final relevant tests and simulator checks; report compatibility and limitations.

Decisions: use opt-in `MP_full_signed_dense_a1`. Keep the 60-point legacy scan default for old checkpoints. Dense points: 33 x 7 over 0.1–3.3 m x ±0.3 m; compact output: two channels x 16 x 5. Use separate trainable 32D encoders for both actors.
