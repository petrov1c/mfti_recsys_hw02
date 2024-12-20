import random
import pandas as pd

from torch.utils.data import Dataset

from src.constants import PROB_TRANSFORM

class RecDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transforms: bool = False,
    ):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx: int):
        rec = self.data.iloc[idx]
        data = {key: val for key, val in rec.items()}

        if self.transforms:
            do = random.random() < PROB_TRANSFORM
            if do:
                pass

        return data

    def __len__(self) -> int:
        return len(self.data)
