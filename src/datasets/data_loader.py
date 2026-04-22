import os
import zipfile
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR10 as TorchVisionCIFAR10
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import FashionMNIST as TorchVisionFashionMNIST
from torchvision.datasets import SVHN as TorchVisionSVHN
from torchvision.datasets.utils import download_url
from torchvision.transforms import InterpolationMode

from . import base_dataset as basedat
from . import memory_dataset as memd

_NORMALIZE_PRESETS = {
    'in1k':  ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'in21k': ((0.5, 0.5, 0.5),       (0.5, 0.5, 0.5)),
}


def _gray_to_rgb(arr):
    """Convert (N, H, W) uint8 grayscale array to (N, H, W, 3)."""
    return np.stack([arr, arr, arr], axis=-1)


def _load_notmnist(path):
    """Download (if needed) and load NotMNIST. Returns (trn_x, trn_y, tst_x, tst_y).

    Images are stored as (H, W, 3) uint8 numpy arrays (already RGB).
    Labels are 0-9 (A=0, B=1, ..., J=9).
    Source: https://github.com/facebookresearch/Adversarial-Continual-Learning
    """
    url      = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
    zip_path = os.path.join(path, 'notMNIST.zip')
    extract  = os.path.join(path, 'notMNIST')

    if not os.path.isfile(zip_path):
        print('Downloading NotMNIST from', url)
        download_url(url, path, 'notMNIST.zip')

    if not os.path.isdir(extract):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path)

    def _load_split(split_dir):
        X, Y = [], []
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = ord(folder) - 65   # A→0, B→1, …, J→9
            for fname in os.listdir(folder_path):
                try:
                    img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
                    X.append(np.array(img))
                    Y.append(label)
                except Exception:
                    pass
        return X, Y

    trn_x, trn_y = _load_split(os.path.join(extract, 'Train'))
    tst_x, tst_y = _load_split(os.path.join(extract, 'Test'))
    return trn_x, trn_y, tst_x, tst_y

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

    elif dataset == 'five_datasets':
        # 5-datasets benchmark: SVHN, MNIST, CIFAR-10, NotMNIST, FashionMNIST
        # Each dataset = 1 task, labels offset to be globally unique (0-49).
        # Images stored as (H,W,3) uint8 — different sizes per dataset, same within each task.
        # get_data iterates element-by-element so mixed sizes are fine; np.asarray() per task
        # works because each task contains only one dataset (uniform shape).

        pieces_trn_x, pieces_trn_y = [], []
        pieces_tst_x, pieces_tst_y = [], []

        # Task 0: SVHN (labels 0-9)
        svhn_trn = TorchVisionSVHN(path, split='train', download=True)
        svhn_tst = TorchVisionSVHN(path, split='test',  download=True)
        svhn_trn_x = svhn_trn.data.transpose(0, 2, 3, 1)   # (N,C,H,W) → (N,H,W,C)
        svhn_tst_x = svhn_tst.data.transpose(0, 2, 3, 1)
        svhn_trn_y = np.array(svhn_trn.labels, dtype=np.int64)
        svhn_tst_y = np.array(svhn_tst.labels, dtype=np.int64)
        np.place(svhn_trn_y, svhn_trn_y == 10, 0)           # SVHN labels digit 0 as 10
        np.place(svhn_tst_y, svhn_tst_y == 10, 0)
        pieces_trn_x.append(svhn_trn_x);  pieces_trn_y.append(svhn_trn_y + 0)
        pieces_tst_x.append(svhn_tst_x);  pieces_tst_y.append(svhn_tst_y + 0)

        # Task 1: MNIST (labels 10-19)
        mnist_trn = TorchVisionMNIST(path, train=True,  download=True)
        mnist_tst = TorchVisionMNIST(path, train=False, download=True)
        pieces_trn_x.append(_gray_to_rgb(mnist_trn.data.numpy()))
        pieces_trn_y.append(np.array(mnist_trn.targets) + 10)
        pieces_tst_x.append(_gray_to_rgb(mnist_tst.data.numpy()))
        pieces_tst_y.append(np.array(mnist_tst.targets) + 10)

        # Task 2: CIFAR-10 (labels 20-29)
        cifar10_trn = TorchVisionCIFAR10(path, train=True,  download=True)
        cifar10_tst = TorchVisionCIFAR10(path, train=False, download=True)
        pieces_trn_x.append(np.array(cifar10_trn.data))
        pieces_trn_y.append(np.array(cifar10_trn.targets) + 20)
        pieces_tst_x.append(np.array(cifar10_tst.data))
        pieces_tst_y.append(np.array(cifar10_tst.targets) + 20)

        # Task 3: NotMNIST (labels 30-39)
        not_trn_x, not_trn_y, not_tst_x, not_tst_y = _load_notmnist(path)
        pieces_trn_x.append(not_trn_x);  pieces_trn_y.append(np.array(not_trn_y) + 30)
        pieces_tst_x.append(not_tst_x);  pieces_tst_y.append(np.array(not_tst_y) + 30)

        # Task 4: FashionMNIST (labels 40-49)
        fmnist_trn = TorchVisionFashionMNIST(path, train=True,  download=True)
        fmnist_tst = TorchVisionFashionMNIST(path, train=False, download=True)
        pieces_trn_x.append(_gray_to_rgb(fmnist_trn.data.numpy()))
        pieces_trn_y.append(np.array(fmnist_trn.targets) + 40)
        pieces_tst_x.append(_gray_to_rgb(fmnist_tst.data.numpy()))
        pieces_tst_y.append(np.array(fmnist_tst.targets) + 40)

        # Flatten into lists — mixed image shapes (28x28 / 32x32) require list, not np.concatenate.
        # get_data iterates element-by-element so this is fine; filtering is a no-op (all labels valid).
        trn_x_all, trn_y_all = [], []
        tst_x_all, tst_y_all = [], []
        for px, py in zip(pieces_trn_x, pieces_trn_y):
            trn_x_all.extend(list(px)); trn_y_all.extend(py.tolist())
        for px, py in zip(pieces_tst_x, pieces_tst_y):
            tst_x_all.extend(list(px)); tst_y_all.extend(py.tolist())

        train_data = {'x': trn_x_all, 'y': trn_y_all}
        test_data  = {'x': tst_x_all, 'y': tst_y_all}

        # Fixed order — each task = one dataset, no class shuffling
        class_order = list(range(50))
        all_data, classes_per_task, class_indices = memd.get_data(
            train_data, test_data,
            classes_first_task=None,
            increment=10,
            validation=validation,
            shuffle_classes=False,
            class_order=class_order,
        )
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