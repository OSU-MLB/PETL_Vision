import torch
from data.dataset.tf_dataset import TFDataset
from data.dataset.vtab_processed import get_processed_VTAB
from data.dataset.cifar100 import get_cifar
from data.dataset.clevr import get_clevr
from data.dataset.few_shot import get_fs
from data.dataset.resisc import get_resisc


def get_dataset(data, params, logger):
    dataset_train, dataset_val, dataset_test = None, None, None

    if data.startswith("tfds_vtab-"):
        logger.info("Loading TFDS vtab data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for vtab)...")
            dataset_train = TFDataset(params, 'trainval')
            dataset_test = TFDataset(params, 'test')
        else:
            logger.info("Loading training and validation  data (tuning for vtab)...")
            dataset_train = TFDataset(params, 'train')
            dataset_val = TFDataset(params, 'val')
    elif data.startswith("processed_vtab-"):
        logger.info("Loading processed vtab data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for vtab)...")
            dataset_train = get_processed_VTAB(params, 'trainval')
            dataset_test = get_processed_VTAB(params, 'test')
        else:
            logger.info("Loading training and validation  data (tuning for vtab)...")
            dataset_train = get_processed_VTAB(params, 'train')
            dataset_val = get_processed_VTAB(params, 'val')
    elif data.startswith('cifar100'):
        logger.info("Loading cifar100 ...")
        if params.final_run:
            logger.info("Loading training data (final training data for cifar100)...")
            dataset_train, _ = get_cifar(params, 'trainval_combined')
            dataset_test, _ = get_cifar(params, 'test')
        else:
            logger.info("Loading training and validation  data (tuning for cifar100)...")
            dataset_train, dataset_val = get_cifar(params, 'trainval_split')
    elif data.startswith('clevr'):
        logger.info("Loading clevr ...")
        if params.final_run:
            logger.info("Loading training data (final training data for cleevr)...")
            dataset_train, _ = get_clevr(params, 'trainval_combined')
            dataset_test, _ = get_clevr(params, 'test')
        else:
            logger.info("Loading training and validation data (tuning for retinopathy)...")
            dataset_train, dataset_val = get_clevr(params, 'trainval_split')
    elif data.startswith('fs'):
        logger.info("Loading training and test data (tuning for fs imagenet)...")
        dataset_train = get_fs(params, mode='train')
        dataset_val = get_fs(params, mode='val')
        dataset_test = get_fs(params, mode='test')
    elif data.startswith('eval'):
        logger.info("Loading test data for fs imagenet DG ...")
        dataset_test = get_fs(params, mode='test')
    elif data.startswith('resisc'):
        logger.info("Loading resisc ...")
        if params.final_run:
            logger.info("Loading training data (final training data for resisc)...")
            dataset_train = get_resisc(params, 'trainval_combined')
            dataset_test = get_resisc(params, 'test')
        else:
            logger.info("Loading training and validation  data (tuning for resisc)...")
            dataset_train, dataset_val = get_resisc(params, 'trainval_split')
    else:
        raise Exception("Dataset '{}' not supported".format(data))
    return dataset_train, dataset_val, dataset_test


def get_loader(params, logger):
    if 'test_data' in params:
        dataset_train, dataset_val, dataset_test = get_dataset(params.test_data, params, logger)
    else:
        dataset_train, dataset_val, dataset_test = get_dataset(params.data, params, logger)

    if isinstance(dataset_train, list):
        train_loader, val_loader, test_loader = [], [], []
        for i in range(len(dataset_train)):
            tmp_train, tmp_val, tmp_test = gen_loader(params, dataset_train[i], dataset_val[i], None)
            train_loader.append(tmp_train)
            val_loader.append(tmp_val)
            test_loader.append(tmp_test)
    else:
        train_loader, val_loader, test_loader = gen_loader(params, dataset_train, dataset_val, dataset_test)

    logger.info("Finish setup loaders")
    return train_loader, val_loader, test_loader


def gen_loader(params, dataset_train, dataset_val, dataset_test):
    train_loader, val_loader, test_loader = None, None, None
    if params.debug:
        num_workers = 1
    else:
        num_workers = 4
    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True

        )
    return train_loader, val_loader, test_loader
