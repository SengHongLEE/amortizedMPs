"""Time one-robot actor calls and the explicit rule with GPU synchronization."""

import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'legged_gym/legged_gym/scripts'))
import evaluate_mp_full_gaits as ev  # imports Isaac Gym before torch

from legged_gym.utils import load_run_config
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.inference_forward_hold import ForwardHoldConfig, InferenceForwardHold
from rsl_rl.runners.multi_interoceptive_policy_runner import MultiInteroceptivePolicyRunner


def benchmark(call, loops=300, repeats=5):
    for _ in range(40):
        call()
    samples = []
    for _ in range(repeats):
        ev.torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(loops):
            call()
        ev.torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / loops)
    return sorted(samples)[len(samples) // 2], samples


def main():
    checkpoint = ROOT / 'legged_gym/logs/A1/1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt'
    cfg, train = load_run_config(ev.MotorPrimitivesCfg(), ev.MotorPrimitivesCfgPPO(), checkpoint.parent)
    cfg.env.num_envs = 1
    cfg.env.play = False
    cfg.env.episode_length_s = 12.
    cfg.terrain.scan_period_steps = 5
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows = cfg.terrain.num_cols = 1
    cfg.terrain.selected = cfg.terrain.selected_kwargs = None
    cfg.terrain.terrain_columns = ['flat']
    cfg.terrain.row_difficulties = [.3]
    cfg.terrain.seed = 3
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    params = ev.gymapi.SimParams()
    ev.gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    ev.EvaluationEnv.repeats, ev.EvaluationEnv.test_speeds = 1, [1.5]
    set_seed(321)
    env = ev.EvaluationEnv(cfg, params, ev.gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        runner = MultiInteroceptivePolicyRunner(env, class_to_dict(train),
                                                log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_optimizer=False)
        high, low = runner.get_inference_policy(device=env.device)
        obs, _ = env.reset()
        gate = InferenceForwardHold(ForwardHoldConfig(), env.scan_cache.x,
                                    env.scan_cache.y, 1, env.device)
        gate_inputs = dict(
            episode_steps=ev.torch.full((1,), 3, device=env.device, dtype=ev.torch.long),
            commands=env.commands, lin_vel=env.base_lin_vel, ang_vel=env.base_ang_vel,
            gravity=env.projected_gravity,
            low=env._dense_lower_heights.reshape(1, len(env.scan_cache.x), len(env.scan_cache.y)),
            high=env._dense_upper_heights.reshape(1, len(env.scan_cache.x), len(env.scan_cache.y)),
            known=env.scan_known_mask.reshape(1, len(env.scan_cache.x), len(env.scan_cache.y)),
            scan_age_steps=env.scan_age_steps, scan_period_steps=5, dt=env.dt, period_steps=3)
        ids = ev.torch.zeros(1, device=env.device, dtype=ev.torch.long)

        def high_call():
            states = runner.primitive_selection_reservoir_states
            return high(obs, states) if states is not None else high(obs[:, -1])

        def low_call():
            states = runner.modulation_param_reservoir_states
            return low(obs[:, -1], states) if states is not None else low(obs[:, -1])

        def gate_call():
            return gate.select('modulation', ids, **gate_inputs)

        results = {}
        with ev.torch.inference_mode():
            for name, call in [('gait_actor', high_call), ('modulation_actor', low_call),
                               ('forward_hold_rule', gate_call)]:
                median, samples = benchmark(call)
                results[name] = dict(median_ms=median, samples_ms=samples)
        print(json.dumps(results, indent=2))
        (Path(__file__).parent / 'component_benchmark.json').write_text(
            json.dumps(results, indent=2) + '\n')
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
