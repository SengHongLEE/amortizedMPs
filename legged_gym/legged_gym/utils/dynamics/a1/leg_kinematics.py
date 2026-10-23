"""Hip-frame A1 forward kinematics and geometric Jacobians, FL/FR/RL/RR."""
import torch


def foot_kinematics(joints, hip_length=.0838, thigh_length=.2, calf_length=.2):
    """Return feet [...,4,3] and Jacobians [...,4,xyz,joint] including hip offsets."""
    if joints.shape[-2:] != (4, 3):
        raise ValueError('joints must have trailing shape [4,3]')
    h, t, k = joints.unbind(-1)
    side = joints.new_tensor([1., -1., 1., -1.])
    x = -thigh_length * t.sin() - calf_length * (t + k).sin()
    zp = -thigh_length * t.cos() - calf_length * (t + k).cos()
    y = side * hip_length * h.cos() - zp * h.sin()
    z = side * hip_length * h.sin() + zp * h.cos()
    knee_dx = -calf_length * (t + k).cos()
    knee_dzp = calf_length * (t + k).sin()
    jacobian = torch.stack((
        torch.stack((torch.zeros_like(x), zp, knee_dx), dim=-1),
        torch.stack((-z, x * h.sin(), -knee_dzp * h.sin()), dim=-1),
        torch.stack((y, -x * h.cos(), knee_dzp * h.cos()), dim=-1),
    ), dim=-2)
    return torch.stack((x, y, z), dim=-1), jacobian
