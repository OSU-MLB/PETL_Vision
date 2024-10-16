import argparse
from utils.setup_logging import get_logger
from utils.misc import set_seed
import time
from utils.misc import load_yaml
from experiment.tune import hyper_tune_final_run

logger = get_logger("PETL_vision")

def main():
    args = setup_parser().parse_args()
    default_params = load_yaml(args.default)
    default_params.data = args.data
    method_params = load_yaml(args.tune)
    lr_wd = load_yaml(args.lrwd)
    hyper_tune_params = {}
    method_static = {}
    for k, v in method_params.items():
        if isinstance(v, list):
            hyper_tune_params[k] = v
        else:
            method_static[k] = v
    for k, v in lr_wd.items():
        if isinstance(v, list):
            hyper_tune_params[k] = v
        else:
            method_static[k] = v
    default_params.update(method_static)
    default_params.final_output_name = args.final_output_name
    set_seed(default_params.random_seed)
    if args.test_bs is not None:
        default_params.test_batch_size = args.test_bs
        if args.bs is not None:
            default_params.batch_size = args.bs
    start = time.time()
    hyper_tune_final_run(default_params, hyper_tune_params, test=args.test)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--data', default='processed_vtab-caltech101')
    parser.add_argument('--default', default='experiment/config/default_vtab_processed.yml')
    parser.add_argument('--tune',  default='experiment/config/method/lora.yml')
    parser.add_argument('--lrwd', default='experiment/config/lr_wd_vtab_processed.yml')
    parser.add_argument('--test', action='store_true', help="put results in test folder")
    parser.add_argument('--final_output_name', default='final_result.json')
    parser.add_argument('--test_bs', type=int, default=None)
    parser.add_argument('--bs', type=int, default=None)
    return parser


if __name__ == '__main__':
    main()
