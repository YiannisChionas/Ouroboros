import math
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert('RGB')
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(path, classes_first_task, increment, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    classes_per_task = []

    # read filenames and labels
    train_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
    test_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)
    if class_order is None:
        number_of_classes = len(np.unique(train_lines[:, 1]))
        class_order = list(range(number_of_classes))
    else:
        number_of_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and number_of_tasks
    if classes_first_task is None:
        number_of_tasks = math.ceil(number_of_classes / increment)
        cpertask = np.array([increment] * (number_of_tasks - 1) + [number_of_classes - increment * (number_of_tasks - 1)])
    else:
        assert classes_first_task < number_of_classes, "first task wants more classes than exist"
        remaining_classes = number_of_classes - classes_first_task
        number_of_tasks = 1 + math.ceil(remaining_classes / increment)
        cpertask = np.array([classes_first_task] + [increment] * (number_of_tasks - 2) + [remaining_classes - increment * (number_of_tasks - 2)])

    assert number_of_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for task in range(number_of_tasks):
        data[task] = {}
        data[task]['name'] = 'task-' + str(task)
        data[task]['train'] = {'x': [], 'y': []}
        data[task]['validation'] = {'x': [], 'y': []}
        data[task]['test'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    for this_image, this_label in train_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['train']['x'].append(this_image)
        data[this_task]['train']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_image, this_label in test_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['test']['x'].append(this_image)
        data[this_task]['test']['y'].append(this_label - init_class[this_task])

    # check classes
    for task in range(number_of_tasks):
        data[task]['ncla'] = len(np.unique(data[task]['train']['y']))
        assert data[task]['ncla'] == cpertask[task], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for task in data.keys():
            for cc in range(data[task]['ncla']):
                class_index = list(np.where(np.asarray(data[task]['train']['y']) == cc)[0])
                random_image = random.sample(class_index, int(np.round(len(class_index) * validation)))
                random_image.sort(reverse=True)
                for image in range(len(random_image)):
                    data[task]['validation']['x'].append(data[task]['train']['x'][random_image[image]])
                    data[task]['validation']['y'].append(data[task]['train']['y'][random_image[image]])
                    data[task]['train']['x'].pop(random_image[image])
                    data[task]['train']['y'].pop(random_image[image])

    # other

    n = 0
    for t in data.keys():
        classes_per_task.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, classes_per_task, class_order
