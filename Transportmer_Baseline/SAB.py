import torch.nn as nn

class SAB(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)
        self.rffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, M=None):
        attn_output, _ = self.mha(query=X, key=X, value=X, key_padding_mask=M)
        H = self.ln1(X + self.dropout(attn_output))
        return self.ln2(H + self.rffn(H))