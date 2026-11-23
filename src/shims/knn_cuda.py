"""Pure-PyTorch replacement for KNN_CUDA 0.2 (`from knn_cuda import KNN`).

Both models call `KNN(k, transpose_mode=True)(ref, query)` with ref (B, N, 3) and
query (B, M, 3) and only use the returned indices (B, M, k), nearest first.
Exact (cdist + topk), so no JIT CUDA build is needed on the compute nodes.
"""
import torch
import torch.nn as nn

SHIM = True


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    @torch.no_grad()
    def forward(self, ref, query):
        if not self.transpose_mode:                      # (B, 3, N) layout
            ref, query = ref.transpose(1, 2), query.transpose(1, 2)
        d = torch.cdist(query.float().contiguous(), ref.float().contiguous())  # (B, M, N)
        dist, idx = d.topk(self.k, dim=-1, largest=False, sorted=True)
        if not self.transpose_mode:
            dist, idx = dist.transpose(1, 2), idx.transpose(1, 2)
        return dist, idx
