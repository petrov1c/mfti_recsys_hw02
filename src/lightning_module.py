import pytorch_lightning as pl
import torch

from src.model import create_model
from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object


class RecModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = create_model(**self._config.model_kwargs)
        self._losses = get_losses(self._config.losses)
        metrics = get_metrics(
#            task='multilabel',
#            average='macro',
#            threshold=0.5,
        )
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, **kwargs) -> torch.Tensor:
        return self._model(**kwargs)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Считаем лосс.
        """
        pr_time = self(users=batch['user'], tracks=batch['track'], first_tracks=batch['first_track'])
        return self._calculate_loss(pr_time, batch['time'],'train_')

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        pr_time = self(users=batch['user'], tracks=batch['track'], first_tracks=batch['first_track'])

        self._calculate_loss(pr_time, batch['time'],'val_')
        self._valid_metrics(pr_time, batch['time'])

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        pr_time = self(users=batch['user'], tracks=batch['track'], first_tracks=batch['first_track'])
        self._test_metrics(pr_time, batch['time'])

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_time: torch.Tensor,
        time: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            if cur_loss.name in ['KL']:
                loss = cur_loss.loss(pr_time[None, :], time[None, :])
            elif cur_loss.name in ['MSE', 'L1Loss']:
                loss = cur_loss.loss(pr_time, time)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
