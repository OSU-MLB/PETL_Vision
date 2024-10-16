import argparse
from utils.misc import load_yaml
from experiment.build_loader import get_loader
from utils.misc import set_seed
import time
import os
import json
from collections import defaultdict
from engine.trainer import Trainer
import torch
from experiment.run import update_output_dir
from experiment.build_model import get_model
from utils.setup_logging import get_logger

logger = get_logger("PETL_vision")

def setup_merge():
    args = setup_parser().parse_args()
    default_params = load_yaml(args.default)
    default_params.data = args.data
    default_params.data_path = args.data_path
    default_params.test_batch_size = args.bs
    method_params = load_yaml(args.tune)
    hyper_tune_params = {}
    method_static = {}
    for k, v in method_params.items():
        if isinstance(v, list):
            hyper_tune_params[k] = v
        else:
            method_static[k] = v
    default_params.update(method_static)
    set_seed(default_params.random_seed)
    output_dir, data_name = update_output_dir(default_params, False)
    default_params.output_dir = output_dir
    final_result = json.load(open(os.path.join(default_params.output_dir, 'final_result.json')))
    best_tune = final_result['best_tune']
    default_params.update(best_tune)
    original_model, tune_parameters, _ = get_model(default_params)
    zero_shot_state_dict = {k: v.clone() for k, v in original_model.state_dict().items() if 'head' in k}
    original_model.load_state_dict(torch.load(default_params.output_dir + '/model.pt')['model_state_dict'])
    return args, default_params, zero_shot_state_dict, original_model, tune_parameters

def main():
    args, default_params, zero_shot_state_dict, original_model, tune_parameters = setup_merge()
    ft_state_dict = {k: v.clone() for k, v in original_model.state_dict().items() if 'head' in k}
    start = time.time()
    assert set(zero_shot_state_dict.keys()) == set(ft_state_dict.keys())
    result = defaultdict(list)
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # merge the prediction head
        combine = {
            key: (1 - alpha) * zero_shot_state_dict[key] + alpha * ft_state_dict[key]
            for key in zero_shot_state_dict.keys()
        }
        original_model.load_state_dict(combine, strict=False)
        if args.option == 'all':
            # merge the PETL parameters
            original_model.params.merge_factor = alpha
        for test_data in ['eval_imagenet-a', 'eval_imagenet-r', 'eval_imagenet-s', 'fs-imagenet', 'eval_imagenet-v2']:
            print(alpha, test_data)
            default_params.test_data = test_data
            trainer = Trainer(original_model, tune_parameters, default_params)
            _, _, test_loader = get_loader(default_params, logger)
            eval_metrics = trainer.eval_classifier(test_loader, 'test')
            result[test_data].append(eval_metrics['top1'])
    if args.option == 'all':
        json.dump(result,
                  open(os.path.join(default_params.output_dir, 'merge_result.json'), 'w'))
    elif args.option == 'fc':
        json.dump(result,
                  open(os.path.join(default_params.output_dir, 'merge_fc_result.json'), 'w'))
    else:
        raise NotImplementedError
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--data', default='fs-imagenet')
    parser.add_argument('--data_path', default='data_folder')
    parser.add_argument('--tune', default='experiment/config/method-imagenet/lora_32.yml')
    parser.add_argument('--default', default='experiment/config/clip_fs_imagenet.yml')
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--option', type=str, default='all')
    return parser


if __name__ == '__main__':
    main()
