import argparse
import logging
import os

import pytorch_lightning as pl
from clearml import OutputModel, Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import RecDM
from src.lightning_module import GPTModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    datamodule = RecDM(config.data_config)
    model = GPTModule(config)

#    task = Task.init(
#        project_name=config.project_name,
#        task_name=f'{config.experiment_name}',
#        auto_connect_frameworks=True,
#    )
#    task.connect(config.dict())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.4f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
#            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)

#    output_model = OutputModel(task=task, name='latest')

    # Сохранение весов модели
#    output_model.update_weights(weights_filename=checkpoint_callback.best_model_path, auto_delete_file=False)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
