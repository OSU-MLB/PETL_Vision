import argparse
from experiment.build_loader import get_loader
import time
import os
import json
from collections import defaultdict
from engine.trainer import Trainer
from utils.setup_logging import get_logger
logger = get_logger("PETL_vision")
from merge_petl import setup_merge

def main():
    args, default_params, zero_shot_state_dict, original_model, tune_parameters = setup_merge()
    ft_state_dict = {k: v.clone() for k, v in original_model.state_dict().items()}

    start = time.time()
    assert set(zero_shot_state_dict.keys()) == set(ft_state_dict.keys())

    result = defaultdict(list)
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combine = {
            key: (1 - alpha) * zero_shot_state_dict[key] + alpha * ft_state_dict[key]
            for key in zero_shot_state_dict.keys()
        }
        original_model.load_state_dict(combine)
        for test_data in ['fs-imagenet', 'eval_imagenet-r', 'eval_imagenet-s', 'eval_imagenet-a', 'eval_imagenet-v2']:
            default_params.test_data = test_data
            trainer = Trainer(original_model, tune_parameters, default_params)
            _, _, test_loader = get_loader(default_params, logger)
            eval_metrics = trainer.eval_classifier(test_loader, 'test')
            result[test_data].append(eval_metrics['top1'])

    json.dump(result,
        open(os.path.join(default_params.output_dir, 'merge_result.json'), 'w'))
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--test_data', default='eval_imagenet-r')
    parser.add_argument('--data', default='fs-imagenet')
    parser.add_argument('--data_path', default='data_folder')
    parser.add_argument('--tune', default='experiment/config/method/vpt_test.yml')
    parser.add_argument('--default', default='experiment/config/default_fs_imagenet.yml')
    parser.add_argument('--test', action='store_true', help="put results in test folder")
    parser.add_argument('--bs', type=int, default=1024)
    return parser


if __name__ == '__main__':
    main()
