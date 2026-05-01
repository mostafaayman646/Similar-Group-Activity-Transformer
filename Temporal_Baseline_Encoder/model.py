import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects mathematical timestamps into the sequence."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing the positional barcodes
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape expected: [Batch, Sequence_Length, Embedding_Dim]
        seq_len = x.size(1)
        # Add the barcode to the input data
        return x + self.pe[:, :seq_len, :].to(x.device)


class TemporalBaselineEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_layers=2, nhead=4):
        super().__init__()
        self.embed_dim = embed_dim

        # The Continuous Spatial Tokenizer (MLP)
        self.movement_mlp = nn.Sequential( #Coordinates
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Role Lookup Table (0=Home, 1=Away, 2=Ball)
        self.role_embedding = nn.Embedding(3, embed_dim)

        # Fusion Layer (Compresses Spatial + Role back to 128d)
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)

        # Time Stamp Injector
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)

        # Temporal Transformer Engine
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, coordinates, roles):
        """
        Input Shapes:
        coordinates: [Batch, Frames, Agents, 2] -> e.g., [8, 100, 23, 2]
        roles:       [Batch, Frames, Agents]    -> e.g., [8, 100, 23, 1]
        """
        B, S, A, _ = coordinates.shape

        # Phase 1: Tokenization
        move_embeds = self.movement_mlp(coordinates)  # Shape: [B, S, A, 128]
        role_embeds = self.role_embedding(roles)  # Shape: [B, S, A, 128]

        # Concatenate on the feature dimension (-1) and fuse
        fused = torch.cat([move_embeds, role_embeds], dim=-1)  # Shape: [B, S, A, 256]
        tokens = self.fusion_layer(fused)  # Shape: [B, S, A, 128]

        # Phase 2: Memory Reshaping
        # We need the Transformer to look at 1 Agent's path over S frames.
        # Swap Agent and Sequence dimensions
        tokens = tokens.permute(0, 2, 1, 3).contiguous()  # Shape: [B, A, S, 128]

        # Merge Batch and Agents so PyTorch treats each player as a separate sequence
        temporal_input = tokens.view(B * A, S, self.embed_dim)  # Shape: [B*23, 100, 128]

        # Phase 3: Temporal Attention
        temporal_input = self.pos_encoder(temporal_input)
        temporal_output = self.temporal_transformer(temporal_input)  # Shape: [B*23, 100, 128]

        # Phase 4: Compression (Pooling)
        # Collapse the timeline: Average the 100 frames into 1 summary per agent
        agent_summaries = temporal_output.mean(dim=1)  # Shape: [B*23, 128]

        # Separate the Batches and Agents again
        agent_summaries = agent_summaries.view(B, A, self.embed_dim)  # Shape: [B, 23, 128]

        # Collapse the team: Average the 23 agents into 1 summary per play
        play_embedding = agent_summaries.mean(dim=1)  # Shape: [B, 128]

        return play_embedding


# Quick test to ensure everything compiles and runs
if __name__ == "__main__":
    print("Testing TemporalBaselineEncoder compilation...")
    model = TemporalBaselineEncoder()

    # Create dummy tensors mirroring our FIFA DataLoader
    dummy_coords = torch.randn(8, 100, 23, 2)  # [Batch=8, Frames=100, Agents=23, (X,Y)]
    dummy_roles = torch.randint(0, 3, (8, 100, 23))  # Random 0, 1, or 2

    output = model(dummy_coords, dummy_roles)
    print(f"Success! Output play_embedding shape: {output.shape}")  # Should be [8, 128]