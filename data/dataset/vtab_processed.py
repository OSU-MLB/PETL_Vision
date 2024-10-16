import os
from timm.data.transforms import str_to_interp_mode
from torchvision import transforms
from utils.global_var import VTAB_DATASETS, MEAN_STD_DICT
from data.dataset.utils import ImageFilelist


def get_processed_VTAB(params, split):
    data_name = params.data.split("processed_vtab-")[-1]
    dataset_root = params.data_path + '/' + data_name

    params.class_num = VTAB_DATASETS[data_name]
    mean, std = MEAN_STD_DICT[params.pretrained_weights]
    if params.normalized:
        transform = transforms.Compose([
            transforms.Resize((params.crop_size, params.crop_size), interpolation=str_to_interp_mode('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((params.crop_size, params.crop_size), interpolation=str_to_interp_mode('bicubic')),
            transforms.ToTensor()])

    if split == 'trainval':
        list_path = os.path.join(dataset_root, 'train800val200.txt')
    elif split == 'val':
        list_path = os.path.join(dataset_root, 'val200.txt')
    elif split == 'test':
        list_path = os.path.join(dataset_root, 'test.txt')
    elif split == 'train':
        list_path = os.path.join(dataset_root, 'train800.txt')
    else:
        raise NotImplementedError

    return ImageFilelist(root=dataset_root, flist=list_path, name=data_name,
                            transform=transform)


