import numpy as np
import copy
import torch
import utils
import functools
from ..vtab_datasets.clevr import CLEVRData
from .tf_dataset import preprocess_fn, to_torch_imgs
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode
from utils.global_var import RETINOPATHY_VAL_SPLIT
from timm.data import IMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD
from utils.global_var import MEAN_STD_DICT


class ClevrDataFull(torch.utils.data.Dataset):
    def __init__(self, params, mode='trainval'):
        dataset_handler = CLEVRData(
                data_dir=params.data_path, task='closest_object_distance')

        def _dict_to_tuple(batch):
            return batch['image'], batch['label']

        tf_data = dataset_handler.get_tf_data(
            batch_size=1,
            drop_remainder=False,
            split_name=mode,
            preprocess_fn=functools.partial(
                preprocess_fn,
                input_range=(0.0, 1.0),
                size=params.crop_size,
            ),
            for_eval=mode!= "train",
            shuffle_buffer_size=1000,
            prefetch=1,
            train_examples=None,
            epochs=1
        ).map(_dict_to_tuple)

        self.images = [t[0].numpy().squeeze() for t in list(tf_data)]
        self.targets = [int(t[1].numpy()[0]) for t in list(tf_data)]
        del tf_data
        mean, std = MEAN_STD_DICT[params.pretrained_weights]
        self.img_mean = torch.tensor(mean).view(3, 1, 1)
        self.img_std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]
        im = to_torch_imgs(
            self.images[index], self.img_mean, self.img_std)
        return im, label


def get_clevr(params, split):
    params.class_num = 6
    if split == 'trainval_combined':
        dataset = ClevrDataFull(params, 'trainval')
        print(f'trainval: {len(dataset)}')
        return dataset, None

    elif split == 'trainval_split':
        train_dataset = ClevrDataFull(params, 'trainval')
        val_dataset = ClevrDataFull(params, 'val')

        return train_dataset, val_dataset

    elif split == 'test':
        dataset_test = ClevrDataFull(params, 'test')
        print(f'test: {len(dataset_test)}')
        return dataset_test, None
    else:
        raise NotImplementedError

