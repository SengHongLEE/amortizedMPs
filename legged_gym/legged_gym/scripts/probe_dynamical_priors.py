"""Diagnostic search over constant dynamical parameters on flat and 8 cm stairs."""
import argparse
import itertools
import json
from pathlib import Path
import time

from evaluate_proprio_terrain import (
    ROOT, EvaluationEnv, MotorPrimitivesCfg, MotorPrimitivesCfgPPO, TraversalMetrics,
    load_run_config, class_to_dict, set_seed, gymapi, gymutil, np, torch,
)
from isaacgym.torch_utils import quat_rotate_inverse
from isaacgym import gymtorch
from legged_gym.utils.dynamics.a1.leg_kinematics import foot_kinematics


class PostureProbeEnv(EvaluationEnv):
    """Diagnostic-only equilibrium feedback; production mapping stays unchanged."""
    def _compute_torques(self, actions):
        if getattr(self, 'posture_gains', None) is not None:
            gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            magnitude = torch.linalg.vector_norm(gravity[:, :2], dim=-1, keepdim=True)
            gravity_xy = gravity[:, :2] * (1 - self.posture_deadband / magnitude.clamp_min(1e-6)).clamp_min(0.)
            delta_z = -self.posture_gains * (
                gravity_xy[:, 0:1] * self.ID._FOOT_X[None] +
                gravity_xy[:, 1:2] * self.ID._FOOT_Y[None])
            actions = actions.clone()
            actions[:, 13:17] += delta_z / self.cfg.control.equilibrium.offset_limits[2]
        nominal_torques = super()._compute_torques(actions)
        if getattr(self, 'cartesian_stiffness', None) is None:
            return nominal_torques
        feet, jacobian = foot_kinematics(self.dof_pos.view(self.num_envs, 4, 3))
        velocity = (jacobian @ self.dof_vel.view(self.num_envs, 4, 3, 1)).squeeze(-1)
        desired = self.foot_targets_reachable
        if not hasattr(self, 'previous_foot_target'):
            self.previous_foot_target = desired.clone()
            self.foot_target_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        desired_velocity = torch.where(
            self.foot_target_valid[:, None, None],
            ((desired - self.previous_foot_target) / self.sim_params.dt).clamp(-2., 2.), 0.)
        self.previous_foot_target.copy_(desired)
        self.foot_target_valid.fill_(True)
        force = self.cartesian_stiffness * (desired - feet)
        force += self.cartesian_damping * (self.cartesian_velocity_feedforward * desired_velocity - velocity)
        stance = self.ID._vertical_phase() >= torch.pi
        gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # All probe environments use the nominal URDF mass (randomization is off).
        support = self.robot_mass * abs(self.sim_params.gravity.z) / stance.sum(-1).clamp_min(1)
        force += self.gravity_compensation * support[:, None, None] * gravity[:, None] * stance[..., None]
        torque = (jacobian.transpose(-1, -2) @ force[..., None]).squeeze(-1).flatten(1)
        torque -= .1 * self.dof_vel
        return torque.clamp(-self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids, *args, **kwargs):
        super().reset_idx(env_ids, *args, **kwargs)
        if hasattr(self, 'foot_target_valid'):
            self.foot_target_valid[env_ids] = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='v2 run supplying physical/controller configuration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=256)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--seconds', type=float, default=18.)
    p.add_argument('--posture-scan', action='store_true', help='Test proprioceptive posture feedback and its response limits.')
    p.add_argument('--posture-deadband-scan', action='store_true', help='Test tilt deadband to avoid feedback on flat-gait oscillations.')
    p.add_argument('--stance-extension', type=float, default=.02, help='Diagnostic ground-search extension in metres.')
    p.add_argument('--impedance-scan', action='store_true', help='Test actuator compliance with slow gait parameters.')
    p.add_argument('--validate-posture-best', action='store_true', help='Paired starts for feedback gains 0/1/2/4 around the best screened base.')
    p.add_argument('--task-impedance-scan', action='store_true', help='Test Cartesian impedance and nominal gravity support.')
    args = p.parse_args()
    if sum((args.posture_scan, args.posture_deadband_scan, args.impedance_scan,
            args.validate_posture_best, args.task_impedance_scan)) > 1:
        p.error('Choose one scan mode')
    if args.candidates < 1 or args.repeats < 1 or args.seconds <= 0 or not 0 <= args.stance_extension <= .08:
        p.error('Require positive candidates/repeats/seconds and stance-extension in [0,.08]')
    cfg, _ = load_run_config(MotorPrimitivesCfg(), MotorPrimitivesCfgPPO(), args.run)
    if cfg.env.action_space != 'cpg_equilibrium_v3':
        p.error('A new 18-action v3 configuration is required')
    grid = []
    for gait, scale, duty, amplitude, bx, bz in itertools.product(
            [0, 2], [.4, .55, .7], [0., .1, .2],
            [.4, .6, .8], [-.02, 0., .02], [-.02, 0., .02]):
        base_frequency, base_duty = (6., .6) if gait == 0 else (8., .5)
        actual_duty = min(base_duty + duty, .8)
        if (1 - actual_duty) / (base_frequency * scale) < .06:
            continue
        grid.append(dict(gait=gait, scale=scale, duty=duty, amplitude=amplitude, bx=bx, bz=bz))
    if args.impedance_scan:
        grid = [dict(gait=gait, scale=scale, duty=duty, amplitude=amplitude,
                     bx=0., bz=0., stiffness=stiffness, damping=damping)
                for gait, scale, duty, amplitude, stiffness, damping in itertools.product(
                    [0, 2], [.4, .55], [0., .1, .2], [.6, .8], [20., 40., 60., 100.], [.5, 1., 2.])]
    if args.posture_scan or args.posture_deadband_scan:
        bases = [
            dict(gait=0, scale=.4, duty=.2, amplitude=.8, bx=0., bz=.02),
            dict(gait=0, scale=.55, duty=.1, amplitude=.6, bx=0., bz=0.),
            dict(gait=2, scale=.4, duty=.2, amplitude=.6, bx=0., bz=.02),
            dict(gait=2, scale=.55, duty=.1, amplitude=.8, bx=-.02, bz=.02),
        ]
        grid = [dict(base, posture_gain=gain, stance_z_speed=speed, equilibrium_frequency=frequency)
                for base, gain, speed, frequency in itertools.product(bases, [0., 1., 2., 4.],
                                                                    [.08, .2, .4], [3., 6.])]
        if args.posture_deadband_scan:
            bases.extend([
                dict(gait=0, scale=.55, duty=.1, amplitude=.8, bx=0., bz=0.),
                dict(gait=2, scale=.4, duty=.1, amplitude=.8, bx=0., bz=0.),
            ])
            grid = [dict(base, posture_gain=gain, posture_deadband=deadband,
                         stance_z_speed=speed, equilibrium_frequency=frequency)
                    for base, gain, deadband, speed, frequency in itertools.product(
                        bases, [1., 2., 4.], [0., .1, .2], [.2, .4], [3., 6.])]
    if args.validate_posture_best:
        grid = [dict(gait=2, scale=.55, duty=.1, amplitude=.8, bx=-.02, bz=.02,
                     posture_gain=gain, stance_z_speed=.08, equilibrium_frequency=3.)
                for gain in [0., 1., 2., 4.]]
    if args.task_impedance_scan:
        bases = [
            dict(gait=0, scale=.4, duty=.1, amplitude=.8, bx=0., bz=0.),
            dict(gait=2, scale=.4, duty=.1, amplitude=.8, bx=0., bz=0.),
            dict(gait=2, scale=.55, duty=.1, amplitude=.8, bx=0., bz=0.),
        ]
        grid = [dict(base, foot_stiffness=kp, foot_damping=kd, gravity_compensation=gravity,
                     velocity_feedforward=velocity)
                for base, kp, kd, gravity, velocity in itertools.product(
                    bases, [300., 600., 1000.], [10., 20., 30.], [0., 1.], [0., 1.])]
    rng = np.random.RandomState(args.seed)
    selected = rng.choice(len(grid), size=min(args.candidates, len(grid)), replace=False)
    candidates = [grid[i] for i in selected]
    actions = np.zeros((len(candidates), 19), dtype=np.float32)
    for i, c in enumerate(candidates):
        actions[i, 0] = c['gait']
        actions[i, 1:5] = 2 * c['amplitude'] ** 2 - 1.
        actions[i, 5:9] = c['bx'] / cfg.control.equilibrium.offset_limits[0]
        actions[i, 13:17] = c['bz'] / cfg.control.equilibrium.offset_limits[2]
        actions[i, 17] = (c['scale'] - 1.) / (1. - cfg.control.timing.frequency_scale_range[0])
        actions[i, 18] = c['duty'] / cfg.control.timing.duty_offset_range[1]
    assert np.max(np.abs(actions[:, 1:])) <= 1.
    set_seed(args.seed)
    cfg.env.num_envs = 2 * len(candidates) * args.repeats
    cfg.env.play = False
    cfg.env.episode_length_s = args.seconds
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = cfg.domain_rand.randomize_base_mass = cfg.domain_rand.push_robots = False
    cfg.terrain.curriculum = False
    cfg.terrain.fixed_terrain_level = cfg.terrain.max_init_terrain_level = None
    cfg.terrain.curriculum_min_level, cfg.terrain.curriculum_max_level = 0, None
    cfg.terrain.num_rows, cfg.terrain.num_cols = 2, 1
    cfg.terrain.row_difficulties = [0., 1.]
    cfg.terrain.selected = 'stairs'
    cfg.terrain.selected_kwargs = cfg.terrain.terrain_columns = None
    cfg.terrain.terrain_params['stairs'].update(step_height_range=[0., .08], step_signs=[-1])
    params = gymapi.SimParams()
    gymutil.parse_sim_config(class_to_dict(cfg.sim), params)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    EvaluationEnv.repeats, EvaluationEnv.test_speeds = len(candidates) * args.repeats, [.5]
    env = PostureProbeEnv(cfg, params, gymapi.SIM_PHYSX, 'cuda:0', True)
    try:
        def expand(key):
            return torch.tensor([c[key] for c in candidates], device=env.device).repeat_interleave(
                args.repeats).repeat(2)
        if args.impedance_scan:
            env.p_gains = expand('stiffness')[:, None].expand(-1, 12).clone()
            env.d_gains = expand('damping')[:, None].expand(-1, 12).clone()
        if args.task_impedance_scan:
            env.cartesian_stiffness = expand('foot_stiffness')[:, None, None]
            env.cartesian_damping = expand('foot_damping')[:, None, None]
            env.gravity_compensation = expand('gravity_compensation')[:, None, None]
            env.cartesian_velocity_feedforward = expand('velocity_feedforward')[:, None, None]
        env.ID.max_stance_extension = args.stance_extension
        if args.posture_scan or args.posture_deadband_scan or args.validate_posture_best:
            env.posture_gains = expand('posture_gain')[:, None]
            env.posture_deadband = (expand('posture_deadband')[:, None] if args.posture_deadband_scan
                                    else torch.zeros_like(env.posture_gains))
            env.ID.stance_velocity_limits = env.ID.stance_velocity_limits[None, None].expand(
                env.num_envs, 4, -1).clone()
            env.ID.stance_velocity_limits[:, :, 2] = expand('stance_z_speed')[:, None]
            env.ID.equilibrium_frequency = expand('equilibrium_frequency')[:, None, None]
        set_seed(args.seed)
        env.reset()
        if args.validate_posture_best:
            for row in range(2):
                start_id = row * len(candidates) * args.repeats
                root = env.root_states[start_id:start_id + args.repeats].clone()
                q = env.dof_pos[start_id:start_id + args.repeats].clone()
                dq = env.dof_vel[start_id:start_id + args.repeats].clone()
                for index in range(1, len(candidates)):
                    begin = start_id + index * args.repeats
                    env.root_states[begin:begin + args.repeats] = root
                    env.dof_pos[begin:begin + args.repeats] = q
                    env.dof_vel[begin:begin + args.repeats] = dq
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
            env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
            env.compute_observations()
        env.metrics = TraversalMetrics(env.num_envs, env.device, 3.4, 1.5)
        env.control_integral = torch.zeros(env.num_envs, 6, device=env.device)
        batch = torch.tensor(actions, device=env.device).repeat_interleave(args.repeats, 0).repeat(2, 1)
        start = time.time()
        with torch.inference_mode():
            for step in range(round(args.seconds / env.dt)):
                env.step(batch)
                if step % 100 == 0:
                    print(f'{step * env.dt:.1f}s simulated; {time.time()-start:.1f}s wall; '
                          f'{env.metrics.active.sum().item()} active', flush=True)
                if not env.metrics.active.any():
                    break
        records = env.metrics.records()
        size = len(candidates) * args.repeats
        rows = []
        for i, c in enumerate(candidates):
            begin = i * args.repeats
            flat = records[begin:begin + args.repeats]
            stairs = records[size + begin:size + begin + args.repeats]
            rows.append(dict(parameters=c, actions=actions[i].tolist(), flat=flat, stairs=stairs,
                             flat_success=sum(e['success'] for e in flat),
                             stair_success=sum(e['success'] for e in stairs)))
        rows.sort(key=lambda r: (r['stair_success'], r['flat_success'],
                                -np.mean([e['mean_speed_error'] for e in r['stairs']])), reverse=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'report.json').write_text(json.dumps(
            dict(seed=args.seed, repeats=args.repeats, stance_extension=args.stance_extension,
                 paired_candidates=args.validate_posture_best,
                 effective_config=class_to_dict(cfg),
                 notes='Constant nominal actions with optional proprioceptive posture feedback; '
                       'diagnostic capacity search, not a trained RL policy or validation set.',
                 candidates=rows), indent=2) + '\n')
        for row in rows[:12]:
            print(row['parameters'], 'flat', row['flat_success'], 'stairs', row['stair_success'], flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
