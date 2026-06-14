import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class HierarchicalPlayEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, frame_layers=2, play_layers=2):
        super().__init__()
        self.d_model = d_model

        # Continuous Spatial Tokenizer
        self.player_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.role_embedding = nn.Embedding(3, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

        # Frame Encoder (Social/Spatial) - No Positional Encoding
        self.frame_cls = nn.Parameter(torch.randn(1, 1, d_model))
        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=1)
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)

        # Play Encoder (Temporal)
        self.pos_encoder = PositionalEncoding(d_model)
        self.play_cls = nn.Parameter(torch.randn(1, 1, d_model))
        play_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )
        self.play_encoder = nn.TransformerEncoder(play_layer, num_layers=play_layers)

    def forward(self, coordinates, roles):
        # coordinates: (Batch, Frames=100, Players=23, 2)
        B, F, P, _ = coordinates.shape

        # Tokenize Features
        loc_embeds = self.player_proj(coordinates)
        role_embeds = self.role_embedding(roles)
        player_tokens = self.fusion(torch.cat([loc_embeds, role_embeds], dim=-1))  # (B, F, P, D)

        # Frame Encoder
        flat_frames = player_tokens.view(B * F, P, self.d_model)  # (B*F, 23, 128)

        # Expand Frame CLS token for all B*F frames
        frame_cls_tokens = self.frame_cls.expand(B * F, -1, -1)  # (B*F, 1, 128)

        # Concatenate CLS to the start of the 23 players
        frame_input = torch.cat([frame_cls_tokens, flat_frames], dim=1)  # (B*F, 24, 128)

        # Pass through Social Transformer (Players looking at players)
        frame_out = self.frame_encoder(frame_input)  # (B*F, 24, 128)

        # Extract the CLS token (Index 0) and un-flatten back to original shape
        frame_embeddings = frame_out[:, 0, :].view(B, F, self.d_model)  # (B, 100, 128)

        # Play Encoder (Temporal)
        temporal_seq = self.pos_encoder(frame_embeddings)  # (B, 100, 128)

        # Expand Play CLS token
        play_cls_tokens = self.play_cls.expand(B, -1, -1)  # (B, 1, 128)
        temporal_input = torch.cat([play_cls_tokens, temporal_seq], dim=1)  # (B, 101, 128)

        # Pass through Temporal Transformer
        play_out = self.play_encoder(temporal_input)  # (B, 101, 128)

        # Extract final Play Embedding from the CLS token
        final_embedding = play_out[:, 0, :]  # (B, 128)

        return final_embedding