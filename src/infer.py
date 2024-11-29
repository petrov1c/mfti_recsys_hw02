import argparse
import json
import logging
import os
import glob
from tqdm import tqdm

import torch

from src.config import Config
from src.lightning_module import RecModule


MAX_LEN = 100

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def infer_track(config: Config):
    list_of_files = glob.glob('experiments/recommender-track/*.ckpt')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    model = RecModule.load_from_checkpoint(latest_file, weights_only=True)
    model = model.to('cuda')
    model.eval()

    track_count = config.model_kwargs['n_tracks']
    tracks_ids = [id for id in range(track_count)]

    recommendations = []
    with torch.no_grad():
        for track_id in tqdm(range(track_count)):
            batch = {
                'users': torch.LongTensor([0] * track_count).to('cuda'),
                'tracks': torch.LongTensor(tracks_ids).to('cuda'),
                'first_tracks': torch.LongTensor([track_id] * track_count).to('cuda'),
            }

            time = model(**batch)

            sorted_time = torch.argsort(time, descending=True)[:MAX_LEN]

            tracks = sorted_time.cpu().tolist()
            tracks = [id for id in tracks if not id == track_id]
            recommendations.append(
                {
                    "track": track_id,
                    "tracks": tracks,
                }
            )

    with open(os.path.join(config.data_config.data_path, "recommendations_hw02_track.json"), "w") as rf:
        for recommendation in tqdm(recommendations):
            rf.write(json.dumps(recommendation) + "\n")


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    infer_track(config)
