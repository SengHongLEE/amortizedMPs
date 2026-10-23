"""Diagnostic riser-collision signal, separate from vertical support contact."""
import math
import torch


class HorizontalObstacleProbe:
    def __init__(self, controller):
        if hasattr(controller, 'horizontal_obstacle_probe'):
            raise ValueError('A horizontal obstacle probe is already attached')
        self.controller = controller
        self.hit = torch.zeros_like(controller._contact)
        self.original_update = controller._update_reflex
        self.original_reset = controller.reset
        controller._update_reflex = self.update
        controller.reset = self.reset
        controller.horizontal_obstacle_probe = self

    def set_forces(self, force):
        self.hit.copy_((torch.linalg.vector_norm(force[..., :2], dim=-1) > 10.)
                       & (force[..., 2] <= 1.))

    def reset(self, ids):
        self.original_reset(ids)
        self.hit[ids] = False

    def update(self, dt):
        c = self.controller
        previous = c.q_obstacle.clone()
        self.original_update(dt)
        phase = c._vertical_phase()
        normal = c._contact & (phase > .15 * math.pi) & (phase < .73 * math.pi)
        if not c._contact_available:
            normal = torch.zeros_like(normal)
        horizontal = self.hit & (phase > .15 * math.pi) & (phase < math.pi)
        c.q_obstacle = c._approach(
            previous, normal | horizontal, dt,
            torch.as_tensor(c.tau_reflex_on, device=c._device),
            torch.as_tensor(c.tau_reflex_off, device=c._device))
