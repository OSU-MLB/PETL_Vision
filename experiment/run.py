import os
from experiment.build_model import get_model
from experiment.build_loader import get_loader
from utils.global_var import OUTPUT_DIR, TUNE_DIR, TUNE_DIR_TEST
from engine.trainer import Trainer
from timm.utils import get_outdir
from utils.log_utils import logging_env_setup
from utils.misc import method_name
from datetime import datetime
import yaml
import torch
from utils.misc import set_seed
from collections import OrderedDict
from statistics import mean
import json
import time
import csv
import numpy as np
from utils.setup_logging import get_logger

logger = get_logger("PETL_vision")


def train(params, train_loader, val_loader, test_loader):
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    trainer = Trainer(model, tune_parameters, params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(train_loader, val_loader, test_loader)
    return train_metrics, best_eval_metrics, eval_metrics, model_grad_params_no_head, trainer.model

def basic_run(params):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = params.data.split("-")[-1]
    dataset_name = params.data.split("-")[0]
    method = method_name(params)
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    output_dir = os.path.join(OUTPUT_DIR, params.pretrained_weights, dataset_name, method, data_name, start_time)
    params.output_dir = get_outdir(output_dir)
    params_text = yaml.safe_dump(params.__dict__, default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(params_text)
    logging_env_setup(params)
    logger.info(f'Start loading {data_name}')
    train_loader, val_loader, test_loader = get_loader(params, logger)

    train(params, train_loader, val_loader, test_loader)


def update_output_dir(default_params, test):
    logger.info(f'start running {default_params.method_name}')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = default_params.data.split("-")[-1]
    dataset_name = default_params.data.split("-")[0]
    method = default_params.method_name
    if test:
        output_dir = os.path.join(TUNE_DIR_TEST, default_params.experiment_name, dataset_name, data_name, method)
    else:
        output_dir = os.path.join(TUNE_DIR, default_params.experiment_name, dataset_name, data_name, method)
    default_params.output_dir = output_dir

    logging_env_setup(default_params)
    return output_dir, data_name




def collect_prediction(default_params):
    output_dir, data_name = update_output_dir(default_params, False)

    if os.path.isfile(os.path.join(default_params.output_dir, 'logits.npy')) and os.path.isfile(os.path.join(default_params.output_dir, 'gt.npy')):
        logger.info('finished, next')
        return

    logger.info(f'-----------------------{default_params.method_name}: {data_name}------------------------------')
    with open(os.path.join(default_params.output_dir, 'tune_summary.csv'), newline='') as f:
        tune_summary = list(csv.reader(f))[1:]

    best_tune = eval(sorted(tune_summary, key=lambda x: float(x[5]))[-1][0])

    logger.info(f'get best tune {best_tune}')
    final_params = default_params
    final_params.update(best_tune)
    final_params.early_patience = 100
    final_params.eval_freq = 100
    final_params.final_run = True
    logger.info(final_params)
    train_loader, _, test_loader = get_loader(default_params, logger)
    logger.info(f'----------- Start final runs-----------')
    train_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_seed(final_params.random_seed)
    model, tune_parameters, model_grad_params_no_head = get_model(final_params)
    trainer = Trainer(model, tune_parameters, final_params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(train_loader, None, test_loader)
    logger.info(
        f'train acc {train_metrics["top1"]}, best acc {best_eval_metrics["top1"]}, last acc {eval_metrics["top1"]}')

    logits, gt = trainer.collect_logits(test_loader)
    with open(os.path.join(final_params.output_dir, 'logits.npy'), 'wb') as f:
        np.save(f, logits)
    with open(os.path.join(final_params.output_dir, 'gt.npy'), 'wb') as f:
        np.save(f, gt)

    train_end = time.time()
    logger.info(f'----------- Total train time : {(train_end - train_start) / 60} mins-----------')




def final_run(default_params, sum_name='final_summary.csv', result_name='final_result.json'):
    if os.path.isfile(os.path.join(default_params.output_dir, result_name)):
        logger.info(f'{result_name} exist, next')
        return
    with open(os.path.join(default_params.output_dir, 'tune_summary.csv'), newline='') as f:
        tune_summary = list(csv.reader(f))[1:]
    if default_params.final_acc_hp:
        # use final acc to select best tune
        best_tune = eval(sorted(tune_summary, key=lambda x: float(x[5]))[-1][0])
    else:
        # use best val acc along training to select best tune
        best_tune = eval(sorted(tune_summary, key=lambda x: float(x[8]))[-1][0])
    final_params = default_params
    final_params.update(best_tune)
    final_params.eval_freq = 1
    final_params.early_patience = 100
    logger.info(final_params)

    final_params.final_run = True
    train_loader, _, test_loader = get_loader(default_params, logger)
    random_seeds = [final_params.random_seed]
    #random_seeds = [final_params.random_seed, final_params.random_seed*5, final_params.random_seed*10]
    logger.info(f'----------- Start final runs: {len(random_seeds)} run(s) to average-----------')
    final_accs = []
    best_accs = []
    train_start = time.time()
    for idx, seed in enumerate(random_seeds):
        train_this_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_params.random_seed = seed
        set_seed(seed)
        train_metrics, best_eval_metrics, eval_metrics, model_grad_params_no_head, model = train(final_params, train_loader, None,
                                                                              test_loader)
        # debug purpose
        # train_metrics = eval_metrics = OrderedDict(
        #     [('loss', round(0.123, 2)), ('top1', round(0.123, 2)), ('top5', round(0.123, 2))])
        result_tracker(idx, train_metrics, eval_metrics, best_eval_metrics,
                       os.path.join(final_params.output_dir, sum_name), idx == 0, 'run_idx', 'test_')
        final_accs.append(eval_metrics['top1'])
        best_accs.append(best_eval_metrics['top1'])
        train_this_end = time.time()
        logger.info(f'----------- train this run : {(train_this_end - train_this_start) / 60} mins-----------')
    json.dump({"avg_acc": mean(final_accs), "avg_best_acc": mean(best_accs), "inserted_parameters": model_grad_params_no_head, 'best_tune': best_tune},
              open(os.path.join(final_params.output_dir, result_name), 'w'))
    train_end = time.time()
    logger.info(f'----------- Total train time : {(train_end - train_start) / 60} mins-----------')




def evaluate(default_params):
    _, _, test_loader = get_loader(default_params, logger)
    if 'eval' in default_params.test_data:
        result_name = f'{default_params.test_data.split("_")[1]}_result.json'
    else:
        result_name = f'{default_params.test_data}_result.json'
    if not os.path.isfile(os.path.join(default_params.output_dir, result_name)):
        if not os.path.isfile(os.path.join(default_params.output_dir, 'final_result.json')):
            logger.info('no final_result.json, the model is not fine-tuned, show model zero shot performance')
            best_tune = ()
            result_name = 'zero_shot_' + result_name
        else:
            result = json.load(open(os.path.join(default_params.output_dir, 'final_result.json')))
            best_tune = result['best_tune']
            default_params.update(best_tune)

        model, tune_parameters, model_grad_params_no_head = get_model(default_params)
        trainer = Trainer(model, tune_parameters, default_params)
        if not os.path.isfile(os.path.join(default_params.output_dir, 'model.pt')):
            assert not os.path.isfile(os.path.join(default_params.output_dir, 'final_result.json'))
            logger.info('no model.pt, shows zero shot performance')
        else:
            trainer.load_weight()
        eval_metrics = trainer.eval_classifier(test_loader, 'test')
        json.dump(
            {"avg_acc": eval_metrics['top1'], "inserted_parameters": model_grad_params_no_head,
             'best_tune': best_tune},
            open(os.path.join(default_params.output_dir, result_name), 'w'))
    else:
        logger.info(f'finish {result_name} for {default_params.method_name}')
    return


def result_tracker(first_col, train_metrics, eval_metrics, best_eval_metrics, filename, write_header=False, first_col_name='param_set',
                   eval_name='val_'):
    rowd = OrderedDict([(first_col_name, first_col)])
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([(eval_name + k, v) for k, v in eval_metrics.items()])
    rowd.update([(eval_name + "best_" + k, v) for k, v in best_eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(rowd)
