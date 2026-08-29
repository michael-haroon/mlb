from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNNEncoder(nn.Module):
    """1D CNN over a team's recent game history sequence."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask, padding):
        # values: (batch, seq_len, feature_dim)
        # mask:   (batch, seq_len, feature_dim) — 1 where value is real, 0 for NaN
        # padding: (batch, seq_len) — 1 for real timesteps, 0 for padded

        x = values * mask
        # Zero out padded timesteps
        x = x * padding.unsqueeze(-1)
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)

        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        x = self.dropout(F.gelu(self.norm3(self.conv3(x))))

        # Masked mean pool over time (only real timesteps)
        pad_mask = padding.unsqueeze(1)  # (batch, 1, seq_len)
        x = (x * pad_mask).sum(dim=2) / pad_mask.sum(dim=2).clamp(min=1.0)
        return x  # (batch, hidden_dim)


class PregameMultiTaskModel(nn.Module):
    """Multi-task pregame model: shared CNN encoders for home/away team histories,
    with distribution heads for each market family."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = _CNNEncoder(feature_dim, hidden_dim, dropout)
        trunk_dim = hidden_dim * 2

        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_home_win = nn.Linear(hidden_dim, 1)
        self.head_yrfi = nn.Linear(hidden_dim, 1)
        self.head_total_runs = nn.Linear(hidden_dim, 2)  # mu, log_sigma
        self.head_run_diff = nn.Linear(hidden_dim, 2)    # mu, log_sigma

    def forward(self, batch: dict) -> dict:
        home_emb = self.encoder(batch["home_values"], batch["home_mask"], batch["home_padding"])
        away_emb = self.encoder(batch["away_values"], batch["away_mask"], batch["away_padding"])

        combined = torch.cat([home_emb, away_emb], dim=-1)
        h = self.trunk(combined)

        total_params = self.head_total_runs(h)
        diff_params = self.head_run_diff(h)

        return {
            "home_win_logit": self.head_home_win(h).squeeze(-1),
            "yrfi_logit": self.head_yrfi(h).squeeze(-1),
            "total_runs_mu": total_params[:, 0],
            "total_runs_sigma": F.softplus(total_params[:, 1]) + 0.1,
            "home_run_diff_mu": diff_params[:, 0],
            "home_run_diff_sigma": F.softplus(diff_params[:, 1]) + 0.1,
        }


class PregamePlayerModel(nn.Module):
    """Player prop model: CNN over player's recent game history."""

    def __init__(
        self,
        feature_dim: int,
        target_count: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        hash_bucket_count: int = 50000,
        player_embed_dim: int = 16,
    ):
        super().__init__()
        self.player_embedding = nn.Embedding(hash_bucket_count, player_embed_dim, padding_idx=0)
        self.encoder = _CNNEncoder(feature_dim, hidden_dim, dropout)

        trunk_input = hidden_dim + player_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Each target gets a count distribution head (log_rate)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(target_count)])

    def forward(self, batch: dict) -> dict:
        seq_emb = self.encoder(batch["values"], batch["mask"], batch["padding"])
        player_emb = self.player_embedding(batch["player_hash"])

        h = torch.cat([seq_emb, player_emb], dim=-1)
        h = self.trunk(h)

        rates = [F.softplus(head(h).squeeze(-1)) + 1e-4 for head in self.heads]
        return {"rates": rates}


class _ResBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.dropout(F.gelu(self.fc1(h)))
        h = self.dropout(self.fc2(h))
        return x + h


class PregameFlatModel(nn.Module):
    """Multi-task MLP for flat game-level feature vectors (no sequence history).

    Takes the full classical feature store (898 features) directly. Temporal
    encoding is already baked in via rolling/EWMA features in the input.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, n_blocks: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.head_home_win = nn.Linear(hidden_dim, 1)
        self.head_yrfi = nn.Linear(hidden_dim, 1)
        self.head_total_runs = nn.Linear(hidden_dim, 2)  # mu, log_sigma
        self.head_run_diff = nn.Linear(hidden_dim, 2)    # mu, log_sigma

    def forward(self, batch: dict) -> dict:
        x = batch["features"]  # (batch, feature_dim)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        total_params = self.head_total_runs(x)
        diff_params = self.head_run_diff(x)

        return {
            "home_win_logit": self.head_home_win(x).squeeze(-1),
            "yrfi_logit": self.head_yrfi(x).squeeze(-1),
            "total_runs_mu": total_params[:, 0],
            "total_runs_sigma": F.softplus(total_params[:, 1]) + 0.1,
            "home_run_diff_mu": diff_params[:, 0],
            "home_run_diff_sigma": F.softplus(diff_params[:, 1]) + 0.1,
        }


class _LSTMEncoder(nn.Module):
    """Bidirectional LSTM over live pitch sequences."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask, padding):
        x = values * mask
        x = x * padding.unsqueeze(-1)
        x = F.gelu(self.proj(x))

        lengths = padding.sum(dim=1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.lstm(packed)
        return self.dropout(h_n[-1])  # last layer hidden state: (batch, hidden_dim)


class LiveGameModel(nn.Module):
    """In-game model: LSTM over pitch sequence prefix for live repricing."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        batter_buckets: int = 50000,
        pitcher_buckets: int = 50000,
        pitch_type_buckets: int = 256,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.batter_embed = nn.Embedding(batter_buckets, embed_dim, padding_idx=0)
        self.pitcher_embed = nn.Embedding(pitcher_buckets, embed_dim, padding_idx=0)
        self.pitch_type_embed = nn.Embedding(pitch_type_buckets, embed_dim, padding_idx=0)

        encoder_input_dim = feature_dim + embed_dim * 3
        self.encoder = _LSTMEncoder(encoder_input_dim, hidden_dim, dropout)

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_home_win = nn.Linear(hidden_dim, 1)
        self.head_yrfi = nn.Linear(hidden_dim, 1)
        self.head_total_runs = nn.Linear(hidden_dim, 2)
        self.head_run_diff = nn.Linear(hidden_dim, 2)

    def forward(self, batch: dict) -> dict:
        batter_emb = self.batter_embed(batch["batter_hashes"])
        pitcher_emb = self.pitcher_embed(batch["pitcher_hashes"])
        ptype_emb = self.pitch_type_embed(batch["pitch_type_hashes"])

        x = torch.cat([batch["values"] * batch["mask"], batter_emb, pitcher_emb, ptype_emb], dim=-1)

        h = self.encoder(x, torch.ones_like(batch["mask"]), batch["padding"])
        h = self.trunk(h)

        total_params = self.head_total_runs(h)
        diff_params = self.head_run_diff(h)

        return {
            "home_win_logit": self.head_home_win(h).squeeze(-1),
            "yrfi_logit": self.head_yrfi(h).squeeze(-1),
            "total_runs_mu": total_params[:, 0],
            "total_runs_sigma": F.softplus(total_params[:, 1]) + 0.1,
            "home_run_diff_mu": diff_params[:, 0],
            "home_run_diff_sigma": F.softplus(diff_params[:, 1]) + 0.1,
        }
