import argparse
from utils.misc import load_yaml
from experiment.run import evaluate
from utils.setup_logging import get_logger
from utils.misc import set_seed
import time


logger = get_logger("PETL_vision")

def main():
    args = setup_parser().parse_args()
    default_params = load_yaml(args.default)
    default_params.data = args.data
    default_params.test_data = args.test_data
    default_params.data_path = args.data_path
    default_params.test_batch_size = args.bs
    if args.merge_factor is not None:
        default_params.merge_factor = args.merge_factor
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
    start = time.time()
    evaluate(default_params)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--test_data', default='eval_imagenet-r')
    parser.add_argument('--data', default='fs-imagenet')
    parser.add_argument('--data_path', default='data_folder')
    parser.add_argument('--tune',  default='experiment/config/method-imagenet/lora_16.yml')
    parser.add_argument('--default', default='experiment/config/clip_fs_imagenet.yml')
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--merge_factor', type=float, default=None)
    return parser


if __name__ == '__main__':
    main()
