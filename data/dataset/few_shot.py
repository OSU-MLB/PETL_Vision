import os
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms.functional import InterpolationMode
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.global_var import VTAB_DATASETS, MEAN_STD_DICT
from data.dataset.utils import add_samples




class few_shot_dataset(ImageFolder):
    def __init__(self, root, dataset, mode='train', transform=None, shot=16,seed=0,**kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.transform = transform
        self.target_transform = None
        self.samples = []

        if mode == 'train':
            if dataset == 'fs-imagenet':
                train_list_path = 'data/annotations/imagenet/train_shot_100_seed_1992'
                add_samples(self.samples, train_list_path, root + '')
            else:
                raise Exception("Dataset '{}' not supported".format(dataset))
        elif mode == 'val':
            if dataset == 'fs-imagenet':
                val_list_path = 'data/annotations/imagenet/val_size_10_seed_1992'
                add_samples(self.samples, val_list_path, root + '')
            else:
                raise Exception("Dataset '{}' not supported".format(dataset))
        elif mode == 'test':
            if dataset == 'fs-imagenet':
                test_list_path = 'data/annotations/imagenet/val_meta.list'
                add_samples(self.samples, test_list_path, root + '')
            else:
                raise Exception("Dataset '{}' not supported".format(dataset))
        else:
            raise Exception("Mode '{}' not supported".format(mode))


def get_fs(params, mode='train'):
    mean, std = MEAN_STD_DICT[params.pretrained_weights]
    if mode == 'train':
        if 'imagenet' in params.data:
            params.class_num = 1000
            transform = create_transform(
                input_size=params.crop_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                mean=mean,
                std=std
            )

        else:
            raise Exception("Dataset '{}' not supported".format(params.data))
        return few_shot_dataset(params.data_path, params.data, 'train', transform=transform)

    elif mode == 'test' or mode == 'val':
        if 'imagenet' in params.data:
            params.class_num = 1000
            size = int((256 / 224) * params.crop_size)
            transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=3),
                transforms.CenterCrop(params.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            raise Exception("Dataset '{}' not supported".format(params.data))

        if 'test_data' in params:
            return few_shot_dataset(params.data_path, params.test_data, mode, transform=transform)
        else:
            return few_shot_dataset(params.data_path, params.data, mode, transform=transform)