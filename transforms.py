import torch

def cartesian_to_spherical_log(cart_vec):
    x, y = cart_vec.unbind(dim=-1); angle = torch.atan2(y, x); magnitude = torch.log10(torch.sqrt(x**2 + y**2) + 1e-8)
    return torch.stack([magnitude, angle], dim=-1)
def spherical_log_to_cartesian(sph_vec):
    log_magnitude, angle = sph_vec.unbind(dim=-1); magnitude = 10**log_magnitude
    x = magnitude * torch.cos(angle); y = magnitude * torch.sin(angle)
    return torch.stack([x, y], dim=-1)