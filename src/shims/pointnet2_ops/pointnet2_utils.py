"""Pure-PyTorch fallback for the two pointnet2_ops calls used by Mamba3D / PointMamba
classification: furthest_point_sample and gather_operation.

Only used when the compiled `pointnet2_ops` package is NOT importable (this shim dir is
appended to the END of sys.path, so a real install always wins).

Semantics follow erikwijmans/Pointnet2_PyTorch `furthest_point_sampling_kernel`:
  * the first selected index is 0;
  * each step picks the point with the largest distance to the selected set;
  * points with squared norm <= 1e-3 are never selected (the CUDA kernel skips them).
"""
import torch


@torch.no_grad()
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) float -> (B, npoint) int32 indices."""
    assert xyz.dim() == 3 and xyz.size(-1) >= 3, xyz.shape
    xyz = xyz[..., :3].float()
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    skip = (xyz * xyz).sum(-1) <= 1e-3                    # kernel: `if (mag <= 1e-3) continue;`
    batch = torch.arange(B, device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    for i in range(1, npoint):
        c = xyz[batch, far].unsqueeze(1)                  # (B, 1, 3)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.masked_fill(skip, -1.0).argmax(-1)
        idx[:, i] = far
    return idx.int()


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint) -> (B, C, npoint)."""
    idx = idx.long().unsqueeze(1).expand(-1, features.size(1), -1)
    return torch.gather(features, 2, idx)
