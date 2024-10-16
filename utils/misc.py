import torch
import numpy as np
import random
import time
import yaml
from dotwiz import DotWiz

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def method_name(params):
    name = ''
    if params.ft_attn_module:
        name += 'attn_'
        name += params.ft_attn_module + '_'
        name += params.ft_attn_mode + '_'
        if params.ft_attn_mode == 'parallel':
            name += params.ft_attn_ln + '_'
        if params.ft_attn_module == 'adapter':
            name += str(params.adapter_bottleneck) + '_'
            name += params.adapter_init + '_'
            name += str(params.adapter_scaler) + '_'
        elif params.ft_attn_module == 'convpass':
            name += str(params.convpass_xavier_init) + '_'
            name += str(params.convpass_scaler) + '_'
        elif params.ft_mlp_module == 'repadapter':
            name += str(params.repadapter_scaler) + '_'
        else:
            raise NotImplementedError
    if params.ft_mlp_module:
        name += 'mlp_'
        name += params.ft_mlp_module + '_'
        name += params.ft_mlp_mode + '_'
        if params.ft_attn_mode == 'parallel':
            name += params.ft_attn_ln + '_'
        if params.ft_mlp_module == 'adapter':
            name += str(params.adapter_bottleneck) + '_'
            name += params.adapter_init + '_'
            name += str(params.adapter_scaler) + '_'
        elif params.ft_mlp_module == 'convpass':
            name += str(params.convpass_xavier_init) + '_'
            name += str(params.convpass_scaler) + '_'
        elif params.ft_mlp_module == 'repadapter':
            name += str(params.repadapter_scaler) + '_'
        else:
            raise NotImplementedError
    if params.vpt_mode:
        name += 'vpt_'
        name += params.vpt_mode + '_'
        name += str(params.vpt_num) + '_'
        name += str(params.vpt_layer) + '_'
    if params.ssf:
        name += 'ssf_'
    if params.lora_bottleneck > 0:
        name += 'lora_' + str(params.lora_bottleneck) + '_'
    if params.fact_type:
        name += 'fact_' + params.fact_type + '_' + str(params.fact_dim) + '_' + str(params.fact_scaler) + '_'
    if params.bitfit:
        name += 'bitfit_'
    if params.vqt_num > 0:
        name += 'vqt_' + str(params.vqt_num) + '_'
    if params.mlp_index:
        name += 'mlp_' + str(params.mlp_index) + '_' + params.mlp_type + '_'
    if params.attention_index:
        name += 'attn_' + str(params.attention_index) + '_' + params.attention_type + '_'
    if params.ln:
        name += 'ln_'
    if params.difffit:
        name += 'difffit_'
    if params.full:
        name += 'full_'
    if params.block_index:
        name += 'block_' + str(params.block_index) + '_'
    #####if nothing, linear
    if name == '':
        name += 'linear' + '_'
    name += params.optimizer
    return name


def set_seed(random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)

def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            return DotWiz(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)


class EarlyStop:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_metrics = None

    def early_stop(self, eval_metrics):
        '''

        :param val_acc:
        :return: bool(if early stop), bool(if save model)
        '''
        if self.max_metrics is None:
            self.max_metrics = eval_metrics
        if eval_metrics['top1'] > self.max_metrics['top1']:
            self.max_metrics = eval_metrics
            self.counter = 0
            return False, True
        elif eval_metrics['top1'] < (self.max_metrics['top1'] - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        return False, False
