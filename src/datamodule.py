import logging
import os
from typing import Optional, List, Dict

from sklearn.model_selection import train_test_split

import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.dataset import RecDataset


class RecDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.cfg = config

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        if self.cfg.need_prepare:
            split_and_save_datasets(self.cfg.data_path, self.cfg.train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            train_set = read_df(self.cfg.data_path, 'train')
            valid_set = read_df(self.cfg.data_path, 'valid')
            test_set = read_df(self.cfg.data_path, 'test')
            self.train_dataset = RecDataset(train_set, transforms=True)
            self.valid_dataset = RecDataset(valid_set)
            self.test_dataset = RecDataset(test_set)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )


def collate_fn(batch: List[Dict]) -> Dict:
    users = [sample['user'] for sample in batch]
    tracks = [sample['track'] for sample in batch]
    first_track = [sample['first_track'] for sample in batch]
    time = [sample['time'] for sample in batch]

    return {
        'user': torch.LongTensor(users),
        'track': torch.LongTensor(tracks),
        'first_track': torch.LongTensor(first_track),
        'time': torch.Tensor(time),
    }


def split_and_save_datasets(data_path: str, train_fraction: float = 0.9):
    df = pd.read_csv(os.path.join(data_path, "data.csv"))
    logging.info(f'Original dataset: {len(df)}')

    train_df, other_df = train_test_split(df, train_size=train_fraction, shuffle=True)
    valid_df, test_df = train_test_split(other_df, test_size=0.5)

    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
