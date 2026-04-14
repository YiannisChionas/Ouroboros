import os
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import SVHN as TorchVisionSVHN
from torchvision.transforms import InterpolationMode

from . import base_dataset_v2 as basedat
from . import memory_dataset_v2 as memd

_NORMALIZE_PRESETS = {
    'in1k':  ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'in21k': ((0.5, 0.5, 0.5),       (0.5, 0.5, 0.5)),
}

def get_loaders(args: dict):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    """ Getting values from args and setting defaults if missing. """
    dataset            = args.get('dataset',            'cifar100')
    data_path          = args.get('data_path',          './data')
    increment          = args.get('increment',          10)
    classes_first_task = args.get('classes_first_task', None)
    batch_size         = args.get('batch_size',         64)
    num_workers        = args.get('num_workers',        4)
    pin_memory         = args.get('pin_memory',         True)
    validation         = args.get('validation',         0.1)
    resize             = args.get('resize',             224)
    crop               = args.get('crop',               None)
    pad                = args.get('pad',                None)
    flip               = args.get('flip',               True)
    normalize          = args.get('normalize',          'in1k')
    extend_channel     = args.get('extend_channel',     None)

    normalize = _NORMALIZE_PRESETS[normalize]

    train_loader, validation_loader, test_loader = [], [], []

    # transformations
    train_transform, test_transform = get_transforms(
                                                     resize = resize,
                                                     pad = pad,
                                                     crop = crop,
                                                     flip = flip,
                                                     normalize=normalize,
                                                     extend_channel=extend_channel
                                                     )

    # datasets
    train_dataset, validation_dataset, test_dataset, classes_per_task = get_dataset(
                                                                                    dataset = dataset,
                                                                                    path = data_path,
                                                                                    increment = increment,
                                                                                    classes_first_task = classes_first_task,
                                                                                    validation = validation,
                                                                                    train_transformation = train_transform,
                                                                                    test_transformation = test_transform
                                                                                    )

    # loaders
    for task in range(len(classes_per_task)):
        train_loader.append(data.DataLoader(
                                            train_dataset[task],
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory
                                            ))
        validation_loader.append(data.DataLoader(
                                                 validation_dataset[task],
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory
                                                 ))
        test_loader.append(data.DataLoader(
                                           test_dataset[task],
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory
                                           ))

    return train_loader, validation_loader, test_loader, classes_per_task


def get_dataset(dataset, path, increment, classes_first_task, validation, train_transformation, test_transformation, class_order=None):
    """Extract dataset and create Dataset class"""

    train_dataset, validation_dataset, test_dataset = [], [], []

    if dataset == 'mnist':
        tvmnist_train = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_test  = TorchVisionMNIST(path, train=False, download=True)
        train_data = {'x': tvmnist_train.data.numpy(), 'y': tvmnist_train.targets.tolist()}
        test_data  = {'x': tvmnist_test.data.numpy(),  'y': tvmnist_test.targets.tolist()}
        # compute splits
        all_data, classes_per_task, class_indices = memd.get_data(
                                                                  train_data,
                                                                  test_data,
                                                                  classes_first_task=classes_first_task,
                                                                  increment=increment,
                                                                  validation=validation,
                                                                  shuffle_classes=class_order is None,
                                                                  class_order=class_order
                                                                  )
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'cifar100':
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        train_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        test_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, classes_per_task, class_indices = memd.get_data(
                                                                  train_data,
                                                                  test_data,
                                                                  classes_first_task=classes_first_task,
                                                                  increment=increment,
                                                                  validation=validation,
                                                                  shuffle_classes=class_order is None,
                                                                  class_order=class_order
                                                                  )
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        train_data = {'x': tvsvhn_trn.data.transpose(0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        test_data = {'x': tvsvhn_tst.data.transpose(0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # Notice that SVHN in Torchvision has an extra training set in case needed
        # tvsvhn_xtr = TorchVisionSVHN(path, split='extra', download=True)
        # xtr_data = {'x': tvsvhn_xtr.data.transpose(0, 2, 3, 1), 'y': tvsvhn_xtr.labels}

        # compute splits
        all_data, classes_per_task, class_indices = memd.get_data(
                                                                  train_data,
                                                                  test_data,
                                                                  classes_first_task=classes_first_task,
                                                                  increment=increment,
                                                                  validation=validation,
                                                                  shuffle_classes=class_order is None,
                                                                  class_order=class_order
                                                                  )
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'imagenet_32':
        import pickle
        # load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # labels from 0 to 999
        # reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        train_data = {'x': x_trn, 'y': y_trn}
        test_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, classes_per_task, class_indices = memd.get_data(
                                                                  train_data,
                                                                  test_data,
                                                                  classes_first_task=classes_first_task,
                                                                  increment=increment,
                                                                  validation=validation,
                                                                  shuffle_classes=class_order is None,
                                                                  class_order=class_order
                                                                  )
        # set dataset type
        Dataset = memd.MemoryDataset

    else:
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, classes_per_task, class_indices = basedat.get_data(
                                                                     path,
                                                                     classes_first_task=classes_first_task,
                                                                     increment=increment,
                                                                     validation=validation,
                                                                     shuffle_classes=class_order is None,
                                                                     class_order=class_order
                                                                     )
        # set dataset type
        Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(len(classes_per_task)):
        all_data[task]['train']['y'] = [label + offset for label in all_data[task]['train']['y']]
        all_data[task]['validation']['y'] = [label + offset for label in all_data[task]['validation']['y']]
        all_data[task]['test']['y'] = [label + offset for label in all_data[task]['test']['y']]
        train_dataset.append(Dataset(all_data[task]['train'], train_transformation, class_indices))
        validation_dataset.append(Dataset(all_data[task]['validation'], test_transformation, class_indices))
        test_dataset.append(Dataset(all_data[task]['test'], test_transformation, class_indices))
        offset += classes_per_task[task][1]

    return train_dataset, validation_dataset, test_dataset, classes_per_task


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    train_transformation_list = []
    test_transformation_list = []

    # resize
    if resize is not None:
        # BICUBIC (interpolation=3)
        train_transformation_list.append(transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC))
        test_transformation_list.append(transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC))

    # padding
    if pad is not None:
        train_transformation_list.append(transforms.Pad(pad))
        test_transformation_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        train_transformation_list.append(transforms.RandomResizedCrop(crop))
        test_transformation_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        train_transformation_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    train_transformation_list.append(transforms.ToTensor())
    test_transformation_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        train_transformation_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        test_transformation_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        train_transformation_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        test_transformation_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(train_transformation_list), transforms.Compose(test_transformation_list)