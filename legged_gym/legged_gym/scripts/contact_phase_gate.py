"""Diagnostic-only contact gate attached to an existing CPG instance."""
import math

import torch


class ContactPhaseGate:
    """Wait at mid-stance for continuous contact, with a bounded deadline.

    The reference clock keeps running. This deliberately isolates a local
    contact gate; it does not add terrain lookahead or alter the policy actions.
    A production version would need to expose the additional state to the actor.
    """
    def __init__(self, controller, maximum_wait, minimum_support=.025):
        if not (math.isfinite(maximum_wait) and maximum_wait >= 0
                and math.isfinite(minimum_support) and minimum_support > 0):
            raise ValueError('Require finite maximum_wait >= 0 and minimum_support > 0')
        if maximum_wait and minimum_support > maximum_wait:
            raise ValueError('minimum_support must not exceed maximum_wait')
        if hasattr(controller, 'contact_phase_gate'):
            raise ValueError('A gate is already attached')
        self.controller = controller
        self.maximum_wait, self.minimum_support = maximum_wait, minimum_support
        self.waiting = torch.zeros_like(controller._contact)
        self.released = torch.zeros_like(self.waiting)
        self.elapsed = torch.zeros_like(controller.X[:, 2])
        self.support = torch.zeros_like(self.elapsed)
        self.gait_ids = controller._gait_ids.clone()
        self.original_step = controller._kuramoto_reflex_substep
        self.original_reset = controller.reset
        controller._kuramoto_reflex_substep = self.step
        controller.reset = self.reset
        controller.contact_phase_gate = self

    def reset(self, ids):
        self.original_reset(ids)
        self.waiting[ids] = False
        self.released[ids] = False
        self.elapsed[ids] = 0.
        self.support[ids] = 0.
        self.gait_ids[ids] = self.controller._gait_ids[ids]

    def step(self, dt):
        c = self.controller
        if self.maximum_wait == 0:
            return self.original_step(dt)
        switched = self.gait_ids != c._gait_ids
        self.waiting[switched] = False
        self.released[switched] = False
        self.elapsed[switched] = 0.
        self.support[switched] = 0.
        self.gait_ids.copy_(c._gait_ids)
        before = c.X[:, 2].clone()
        self.original_step(dt)
        after = c.X[:, 2]
        wrapped = after < before
        self.released[wrapped] = False
        self.elapsed[wrapped] = 0.
        self.support = torch.where(c._contact, self.support + dt, 0.)
        beta = c.selected_duty_factors(c._gait_ids)
        # Vertical phase 1.5*pi: deepest point of the existing stance search.
        gate_phase = math.pi * (2. - 1.5 * beta)
        entering = (before < gate_phase) & (after >= gate_phase) & ~wrapped
        self.waiting |= entering & ~self.released & (self.support < self.minimum_support)
        done = self.waiting & ((self.support >= self.minimum_support)
                               | (self.elapsed >= self.maximum_wait))
        self.released |= done
        self.waiting &= ~done
        after.copy_(torch.where(self.waiting, gate_phase, after))
        c.X_dot[:, 2].masked_fill_(self.waiting, 0.)
        c.omega.masked_fill_(self.waiting, 0.)
        self.elapsed += self.waiting * dt
