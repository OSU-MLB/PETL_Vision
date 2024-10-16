from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode
from utils.global_var import CIFAR_VAL_SPLIT
from timm.data import IMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD
import numpy as np
from torch.utils.data import Subset

def get_cifar(params, split):
    params.class_num = 100
    if params.pretrained_weights == 'vit_base_patch16_224_in21k':
        mean, std = IMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=str_to_interp_mode('bicubic')),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    if split == 'trainval_combined':
        dataset = CIFAR100(params.data_path, transform=transform_train, train=True, download=True)
        if params.data in ['cifar100-500', 'cifar100-1k', 'cifar100-5k', 'cifar100-10k']:
            indices = np.load(f'data/annotations/cifar100/{params.data}-{split}.npy')
        elif params.data == 'cifar100-full':
            indices = None
        else:
            raise ValueError
        if indices is not None:
            dataset = Subset(dataset, indices)
        return dataset, None
    elif split == 'trainval_split':
        dataset = CIFAR100(params.data_path, transform=transform_val, train=True, download=True)
        if params.data in ['cifar100-500', 'cifar100-1k', 'cifar100-5k', 'cifar100-10k', 'cifar100-full']:
            train_indices = np.load(f'data/annotations/cifar100/{params.data}-{split}_train.npy')
            val_indices = np.load(f'data/annotations/cifar100/{params.data}-{split}_val.npy')
        else:
            raise ValueError

        if params.data == 'cifar100-500':
            train_ret = []
            val_ret = []
            for i in range(5):
                train_ret.append(Subset(dataset, train_indices[i]))
                val_ret.append(Subset(dataset, val_indices[i]))
            return train_ret, val_ret
        else:
            return Subset(dataset, train_indices), Subset(dataset, val_indices)
    elif split == 'test':
        dataset_test = CIFAR100(params.data_path, transform=transform_val, train=False, download=True)
        return dataset_test, None
    else:
        raise NotImplementedError


