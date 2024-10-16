import argparse
from utils.misc import load_yaml
from experiment.run import collect_prediction
from utils.setup_logging import get_logger
from utils.misc import set_seed
import time


logger = get_logger("PETL_vision")

def main():
    args = setup_parser().parse_args()
    default_params = load_yaml(args.default)
    default_params.data = args.data
    default_params.update(load_yaml(args.tune))
    set_seed(default_params.random_seed)
    start = time.time()
    collect_prediction(default_params)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')

def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision')
    parser.add_argument('--data', default='processed_vtab-caltech101')
    parser.add_argument('--default', default='experiment/config/default_vtab_processed.yml')
    parser.add_argument('--tune', default='experiment/config/method/adaptformer.yml')
    return parser


if __name__ == '__main__':
    main()
