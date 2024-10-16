from utils.misc import load_yaml
from utils.setup_logging import get_logger
import torch
import os
import csv
from sklearn.model_selection import ParameterGrid
from timm.utils import get_outdir
from experiment.build_loader import get_loader
from experiment.run import update_output_dir, final_run, train, result_tracker
from collections import OrderedDict
import time
logger = get_logger("PETL_vision")



def hyper_tune_final_run(default_params, hyper_tune_params, test=False):
    output_dir, data_name = update_output_dir(default_params, test)
    if len(hyper_tune_params) == 0:
        logger.info('No tunable parameters, final run')
        #if not os.path.isfile(os.path.join(default_params.output_dir, 'final_result.json')):
        logger.info('start final run')
        final_run(default_params, {}, result_name=default_params.final_output_name)

    param_grid = list(ParameterGrid(hyper_tune_params))
    already_set = []
    if os.path.exists(os.path.join(default_params.output_dir, 'tune_summary.csv')):
        with open(os.path.join(default_params.output_dir, 'tune_summary.csv'), newline='') as f:
            tune_summary = list(csv.reader(f))[1:]
            already_set = [eval(i[0]) for i in tune_summary]
            if len(tune_summary) == len(param_grid):
                logger.info('all tuning is done, start final run')
                final_run(default_params, result_name=default_params.final_output_name)
                return
    get_outdir(output_dir)
    logger.info(f'Start loading {data_name}')
    default_params.final_run = False
    train_loader, val_loader, _ = get_loader(default_params, logger)
    start_tune = time.time()
    logger.info(f"Total tune sets: {len(param_grid)}")

    for idx, param_set in enumerate(param_grid):
        logger.info('check if this set is run  before')

        if param_set in already_set:
            logger.info('already run')
            logger.info(param_set)
            continue
        else:
            logger.info('need to run this set')
            logger.info(param_set)
        start_this_tune = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Tuning with this set:")
        final_params = default_params
        final_params.update(param_set)
        if isinstance(train_loader, list):
            train_loss, train_top1, train_top5 = 0, 0, 0
            val_loss, val_top1, val_top5 = 0, 0, 0
            val_best_loss, val_best_top1, val_best_top5 = 0, 0, 0
            for i in range(len(train_loader)):
                train_metrics, best_eval_metrics, eval_metrics, _, _ = train(final_params, train_loader[i], val_loader[i], None)
                train_loss += train_metrics['loss']
                train_top1 += train_metrics['top1']
                train_top5 += train_metrics['top5']
                val_loss += eval_metrics['loss']
                val_top1 += eval_metrics['top1']
                val_top5 += eval_metrics['top5']
                val_best_loss += best_eval_metrics['loss']
                val_best_top1 += best_eval_metrics['top1']
                val_best_top5 += best_eval_metrics['top5']
            train_metrics = OrderedDict([('loss', train_loss/5), ('top1', train_top1/5), ('top5',  train_top5/5)])
            eval_metrics = OrderedDict([('loss', val_loss/5), ('top1', val_top1/5), ('top5',  val_top5/5)])
            best_eval_metrics = OrderedDict([('loss', val_best_loss/5), ('top1', val_best_top1/5), ('top5',  val_best_top5/5)])
        else:
            train_metrics, best_eval_metrics, eval_metrics, _, _ = train(final_params, train_loader, val_loader, None)
        # debug purpose
        # train_metrics = eval_metrics = OrderedDict(
        #     [('loss', round(0.123, 2)), ('top1', round(0.123, 2)), ('top5', round(0.123, 2))])
        result_tracker(param_set, train_metrics, eval_metrics, best_eval_metrics,
                       os.path.join(final_params.output_dir, 'tune_summary.csv'), idx == 0)
        end_this_tune = time.time()
        logger.info(f'----------- Tune this set : {(end_this_tune - start_this_tune) / 60} mins-----------')
    end_tune = time.time()
    logger.info(f'----------- Tune total {len(param_grid)} sets: {(end_tune - start_tune) / 60} mins-----------')
    final_run(default_params, result_name=default_params.final_output_name)


