from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    data_path: str
    need_prepare: bool
    batch_size: int
    max_len: int
    n_workers: int
    train_size: float


class Config(BaseModel):
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
