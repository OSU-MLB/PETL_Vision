from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode
from data.dataset.utils import add_samples



class ResiscDataset(ImageFolder):
    def __init__(self, root, data_list, transform=None):
        self.data_root = root
        self.loader = default_loader
        self.transform = transform
        self.target_transform = None
        self.samples = []

        add_samples(self.samples, data_list, root)


def get_resisc(params, mode='trainval_split'):
    params.class_num = 45
    if params.pretrained_weights == 'vit_base_patch16_224_in21k':
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        raise Exception('define mean, std')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=str_to_interp_mode('bicubic')),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    if mode == 'trainval_split':

        if params.data in ['resisc-450', 'resisc-225']:
            train_ret = []
            val_ret = []
            # run 5 different splits for more robust tuning
            for i in range(5):
                train_data_list = f'data/annotations/resisc/{params.data}_train_{i}.txt'
                train_ret.append(ResiscDataset(params.data_path, train_data_list, transform_train))
                val_data_list = f'data/annotations/resisc/{params.data}_val_{i}.txt'
                val_ret.append(ResiscDataset(params.data_path, val_data_list, transform_val))
            return train_ret, val_ret
        else:
            train_data_list = f'data/annotations/resisc/{params.data}_train.txt'
            val_data_list = f'data/annotations/resisc/{params.data}_val.txt'
            return ResiscDataset(params.data_path, train_data_list, transform_train), ResiscDataset(params.data_path,
                                                                                                    val_data_list,
                                                                                                    transform_val)
    elif mode == 'trainval_combined':
        train_data_list = f'data/annotations/resisc/{params.data}_combine.txt'
        return ResiscDataset(params.data_path, train_data_list, transform_train)
    elif mode == 'test':
        test_data_list = f'data/annotations/resisc/test.txt'
        return ResiscDataset(params.data_path, test_data_list, transform_val)
    else:
        raise NotImplementedError
