from Transformer.SAB import SAB
from Transformer.PE import PositionalEncoding
from Transformer.Embedding import TrajectoryEmbedding
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, in_features, d_model, num_heads=8, num_sab_layers=2):
        super().__init__()
        self.embedding = TrajectoryEmbedding(in_features, d_model)
        self.pe = PositionalEncoding(d_model)
        self.sabs = nn.ModuleList([SAB(d_model, num_heads) for _ in range(num_sab_layers)])

    def forward(self, x, mask=None):
        B, T, N, F = x.shape
        # Project and add CLS token
        J = self.embedding(x)
        # Flatten: (B, T*(N+1), d_model)
        out = J.view(B, T * (N + 1), -1)
        # Add PE
        out = self.pe(out)
        # Attention blocks
        for sab in self.sabs:
            out = sab(out, M=mask)
        # Reshape back to grid
        return out.view(B, T, N + 1, -1)