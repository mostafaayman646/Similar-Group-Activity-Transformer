import torch
import torch.nn as nn

class TrajectoryEmbedding(nn.Module):
    def __init__(self, in_features, d_model):
        super().__init__()
        self.rffn = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, d_model))

    def forward(self, x):
        B, T, N, F = x.shape
        x_proj = self.rffn(x) 
        cls_expanded = self.cls_token.expand(B, T, 1, -1)
        return torch.cat([cls_expanded, x_proj], dim=2)