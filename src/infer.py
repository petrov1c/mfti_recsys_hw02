import argparse
import json
import logging
import os
from tqdm import tqdm

import torch

from src.config import Config
from src.lightning_module import RecModule


MAX_LEN = 100

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def infer(config: Config):
    model = RecModule.load_from_checkpoint('experiments/recommender/epoch_epoch=59-val_rmse=0.5607.ckpt', weights_only=True)
    model = model.to('cuda')
    model.eval()

    track_count = config.model_kwargs['n_tracks']
    tracks_ids = [id for id in range(track_count)]

    recommendations = []
    with torch.no_grad():
        for user_id in tqdm(range(config.model_kwargs['n_users'])):
            batch = {
                'user': torch.LongTensor([user_id] * track_count).to('cuda'),
                'track': torch.LongTensor(tracks_ids).to('cuda'),
            }

            time = model(users = batch['user'], tracks = batch['track'])

            # треки и пользователи начинаются с нуля
            sorted_time = torch.argsort(time, descending=True)[:MAX_LEN]

            recommendations.append(
                {
                    "user": user_id,
                    "tracks": sorted_time.cpu().tolist(),
                }
            )
            # ToDo Отсортировать по убыванию времени и выкашиванию повторяющихся артистов


    with open(os.path.join(config.data_config.data_path, "recommendations_hw02.json"), "w") as rf:
        for recommendation in tqdm(recommendations):
            rf.write(json.dumps(recommendation) + "\n")


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    infer(config)
