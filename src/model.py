import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 64 # он же hidden_dim - внутрення размерность модели
    n_tracks: int = 50_000 # он же vocab_size, размер словаря модели
    n_users: int = 10_000 # число пользователей
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5


class Recommender(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        user_bias = torch.zeros((2, cfg.n_users))
        user_bias[1] += 1
        self.user_bias = nn.Parameter(user_bias)
        self.tracks = nn.Embedding(cfg.n_tracks, cfg.d_model)

    def forward(self, users: torch.LongTensor, tracks: torch.LongTensor, first_tracks: torch.LongTensor):
        first_track_embs = self.tracks(first_tracks)
        track_embs = self.tracks(tracks)

        x1 = torch.sum(first_track_embs * track_embs, dim=1)

        user_bias = self.user_bias[:, users]
        x2 = (x1 - user_bias[0]) * user_bias[1]
        x3 = nn.functional.sigmoid(x2)
        return x3


def create_model(**kwargs):
    train_config = Config(**kwargs)
    return Recommender(train_config)
