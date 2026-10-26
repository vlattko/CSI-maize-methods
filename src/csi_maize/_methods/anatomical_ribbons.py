"""Small independent differentiable cubic-Bezier leaf surface prior.

No third-party modelling code or learned weights are used. World Y is upward.
The module represents prescribed leaf organs; it does not infer organ count,
stem attachments or hidden anatomy. Controls use the input tensor's units.
"""

import numpy as np
import torch


def ribbon_faces(samples=48, cross_samples=9):
    """Return (2*(samples-1)*(cross_samples-1),3) leaf-local triangle indices.

    Flatten one leaf's [samples,cross_samples,3] grid in row-major order.
    To join L leaves, add leaf_index*samples*cross_samples to these indices.
    The exact converged tip gives zero-area triangles along the final row;
    keep them for differentiable topology or remove them at export time.
    """
    if samples < 2 or cross_samples < 3 or cross_samples % 2 != 1:
        raise ValueError("Use samples>=2 and an odd cross_samples>=3 for an exact midline")
    i, j = np.meshgrid(np.arange(samples - 1), np.arange(cross_samples - 1), indexing="ij")
    a = (i * cross_samples + j).ravel()
    b = a + cross_samples
    c = b + 1
    d = a + 1
    return (
        np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=1)
        .reshape(-1, 3)
        .astype(np.int64)
    )


def _unit(vector, eps):
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def cubic_ribbons(control_points, widths, twists, samples=48, cross_samples=9):
    """Create explicit leaf strips, differentiable in all organ parameters.

    Args:
        control_points: floating torch tensor [L,4,3], cubic Bezier midribs.
            P0 is the exact stem attachment; P3 is the exact leaf tip.
        widths: positive tensor [L,3], quadratic Bernstein halfwidth controls.
            Profile = quadratic_width*sqrt(sin(pi*t))
                      + .08*width0*(1-t)^4. The tip is exactly zero width.
        twists: tensor [L,3], quadratic Bernstein twist angles in radians,
            applied by Rodrigues rotation about the local midrib tangent.
        samples, cross_samples: longitudinal and odd transverse grid sizes.

    Returns:
        vertices [L,samples,cross_samples,3], midribs [L,samples,3],
        numpy faces [T,3] indexing one leaf's flattened grid.

    Frame convention: horizontal base-to-tip azimuth defines the initial
    transverse direction, projected perpendicular to each unit tangent. If
    that azimuth vanishes, P2 then P1 supplies it. An exactly vertical organ
    has no observable azimuth, so a fixed +X gauge is used. Y-rotation
    equivariance holds away from this unavoidable azimuth degeneracy.
    """
    cp = control_points
    if (
        not torch.is_tensor(cp)
        or not cp.is_floating_point()
        or cp.ndim != 3
        or cp.shape[1:] != (4, 3)
    ):
        raise ValueError("control_points must be floating torch [L,4,3]")
    if widths.shape != (cp.shape[0], 3) or twists.shape != (cp.shape[0], 3):
        raise ValueError("widths and twists must have shape [L,3]")
    if (
        widths.device != cp.device
        or twists.device != cp.device
        or widths.dtype != cp.dtype
        or twists.dtype != cp.dtype
    ):
        raise ValueError("All tensors must share dtype and device")
    if bool((widths <= 0).any()):
        raise ValueError(
            "Halfwidth controls must be positive; use softplus outside this module if needed"
        )
    faces = ribbon_faces(samples, cross_samples)
    eps = 1e-10 if cp.dtype == torch.float64 else 1e-6
    t = torch.linspace(0, 1, samples, device=cp.device, dtype=cp.dtype)
    b = torch.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t**3], dim=-1)
    midribs = torch.einsum("sk,lkd->lsd", b, cp)
    derivative_basis = torch.stack([(1 - t) ** 2, 2 * (1 - t) * t, t * t], dim=-1)
    derivative = 3 * torch.einsum("sk,lkd->lsd", derivative_basis, cp[:, 1:] - cp[:, :-1])
    up = cp.new_tensor([0.0, 1.0, 0.0])
    end_direction = cp[:, 3] - cp[:, 0]
    fallback_tangent = torch.where(
        torch.linalg.vector_norm(end_direction, dim=-1, keepdim=True) > eps,
        _unit(end_direction, eps),
        up.expand_as(end_direction),
    )
    tangent = torch.where(
        torch.linalg.vector_norm(derivative, dim=-1, keepdim=True) > eps,
        _unit(derivative, eps),
        fallback_tangent[:, None, :],
    )
    horizontal = cp.new_tensor([1.0, 0.0, 1.0])
    azimuth = end_direction * horizontal
    for position in [2, 1]:
        candidate = (cp[:, position] - cp[:, 0]) * horizontal
        azimuth = torch.where(
            torch.linalg.vector_norm(azimuth, dim=-1, keepdim=True) > eps, azimuth, candidate
        )
    # Exactly vertical controls have rotational symmetry: an arbitrary gauge
    # is mathematically necessary to construct a finite-width ribbon.
    azimuth = torch.where(
        torch.linalg.vector_norm(azimuth, dim=-1, keepdim=True) > eps,
        _unit(azimuth, eps),
        cp.new_tensor([0.0, 0.0, -1.0]).expand_as(azimuth),
    )
    transverse = torch.linalg.cross(up.expand_as(azimuth), azimuth, dim=-1)
    transverse = transverse[:, None, :].expand_as(tangent)
    projected = transverse - (transverse * tangent).sum(dim=-1, keepdim=True) * tangent
    alternative = (
        azimuth[:, None, :] - (azimuth[:, None, :] * tangent).sum(dim=-1, keepdim=True) * tangent
    )
    projected = torch.where(
        torch.linalg.vector_norm(projected, dim=-1, keepdim=True) > eps, projected, alternative
    )
    width_direction = _unit(projected, eps)
    q = torch.stack([(1 - t) ** 2, 2 * (1 - t) * t, t * t], dim=-1)
    angle = torch.einsum("sk,lk->ls", q, twists)[..., None]
    rotated = (
        width_direction * torch.cos(angle)
        + torch.linalg.cross(tangent, width_direction, dim=-1) * torch.sin(angle)
        + tangent * (tangent * width_direction).sum(dim=-1, keepdim=True) * (1 - torch.cos(angle))
    )
    # t is a fixed sampling grid, so the taper introduces no singular parameter gradients.
    taper = torch.sqrt(torch.sin(torch.pi * t).clamp_min(0))
    taper = torch.where(t == 1, torch.zeros_like(taper), taper)
    profile = (
        torch.einsum("sk,lk->ls", q, widths) * taper[None, :]
        + 0.08 * widths[:, 0, None] * (1 - t)[None, :] ** 4
    )
    profile = torch.where(t[None, :] == 1, torch.zeros_like(profile), profile)
    across = torch.linspace(-1, 1, cross_samples, device=cp.device, dtype=cp.dtype)
    vertices = (
        midribs[:, :, None, :]
        + (profile[:, :, None, None] * across[None, None, :, None]) * rotated[:, :, None, :]
    )
    return vertices, midribs, faces


# Short, explicit alias for callers assembling multiple organ grids.
faces = ribbon_faces
