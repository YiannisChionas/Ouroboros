import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def last_layer_analysis(heads, heads_dist, last_task_index, classes_per_task, y_lim=False, sort_weights=False):
    """Plot last layer weight and bias analysis"""
    print('Plotting last layer analysis...')
    number_of_classes = sum([x for (_, x) in classes_per_task])
    weights, weights_dist, biases, biases_dist, indexes = [], [], [], [], []
    class_id = 0
    with torch.no_grad():
        for task in range(last_task_index + 1):
            number_of_classes_of_task = classes_per_task[task][1]
            indexes.append(np.arange(class_id, class_id + number_of_classes_of_task))
            if type(heads) == torch.nn.Linear:  # Single head
                biases.append(heads.bias[class_id: class_id + number_of_classes_of_task].detach().cpu().numpy())
                weights.append((heads.weight[class_id: class_id + number_of_classes_of_task] ** 2).sum(1).sqrt().detach().cpu().numpy())
            else:  # Multi-head
                if heads_dist:
                    weights_dist.append((heads_dist[task].weight ** 2).sum(1).sqrt().detach().cpu().numpy())
                    biases_dist.append(heads_dist[task].bias.detach().cpu().numpy())
                weights.append((heads[task].weight ** 2).sum(1).sqrt().detach().cpu().numpy())
                if type(heads[task]) == torch.nn.Linear:
                    biases.append(heads[task].bias.detach().cpu().numpy())
                else:
                    biases.append(np.zeros(weights[-1].shape))  # For LUCIR
            class_id += number_of_classes_of_task

    # Figure weights
    f_weights = plt.figure(dpi=300)
    ax = f_weights.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, weights), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Weights L2-norm", fontsize=11, fontfamily='serif')
    if number_of_classes is not None:
        ax.set_xlim(0, number_of_classes)
    if y_lim:
        ax.set_ylim(0, 5)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    # Figure biases
    f_biases = plt.figure(dpi=300)
    ax = f_biases.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, biases), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Bias values", fontsize=11, fontfamily='serif')
    if number_of_classes is not None:
        ax.set_xlim(0, number_of_classes)
    if y_lim:
        ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

    if weights_dist:

        # Figure weights_dist
        f_weights_dist = plt.figure(dpi=300)
        ax = f_weights_dist.subplots(nrows=1, ncols=1)
        for i, (x, y) in enumerate(zip(indexes, weights_dist), 0):
            if sort_weights:
                ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
            else:
                ax.bar(x, y, label="Task {}".format(i))
        ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
        ax.set_ylabel("Weights L2-norm", fontsize=11, fontfamily='serif')
        if number_of_classes is not None:
            ax.set_xlim(0, number_of_classes)
        if y_lim:
            ax.set_ylim(0, 5)
        ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

        # Figure biases_dist
        f_biases_dist = plt.figure(dpi=300)
        ax = f_biases_dist.subplots(nrows=1, ncols=1)
        for i, (x, y) in enumerate(zip(indexes, biases_dist), 0):
            if sort_weights:
                ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
            else:
                ax.bar(x, y, label="Task {}".format(i))
        ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
        ax.set_ylabel("Bias values", fontsize=11, fontfamily='serif')
        if number_of_classes is not None:
            ax.set_xlim(0, number_of_classes)
        if y_lim:
            ax.set_ylim(-1.0, 1.0)
        ax.legend(loc='upper left', fontsize='11') #, fontfamily='serif')

        return f_weights, f_biases, f_weights_dist, f_biases_dist
    else:
        return f_weights, f_biases
