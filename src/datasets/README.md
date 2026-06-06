# Datasets
The following datasets are actively used and tested:

| `"dataset"` value | Description | Tasks |
|---|---|---|
| `"cifar100"` | CIFAR-100 (100 classes) — TorchVision (auto-download) | 10×10 |
| `"food101"` | Food-101 (100 of 101 classes) — custom path-based | 10×10 |
| `"inat200"` | iNaturalist-200 (200 animal classes) — custom path-based | 20×10 |
| any other string | Custom dataset (path-based) | configured via `increment` |

> **Note — `"five_datasets"`**: The 5-datasets benchmark (SVHN + MNIST + CIFAR-10 + NotMNIST + FashionMNIST) is implemented in `data_loader.py` but is **not ready to use** — configs and SLURM scripts are not yet provided.

## Main usage
All dataset configuration is done through the JSON config file. The core keys are:

| Key | Description | Default |
|-----|-------------|---------|
| `"dataset"` | Dataset identifier (see table above) | `"cifar100"` |
| `"data_path"` | Path to the dataset root folder | `"./data"` |
| `"increment"` | Number of classes per task | `10` |
| `"classes_first_task"` | Override class count for task 0 (null = same as `increment`) | `null` |
| `"batch_size"` | Samples per batch | `64` |
| `"num_workers"` | DataLoader worker processes | `2` |
| `"pin_memory"` | Copy tensors into CUDA pinned memory | `false` |
| `"use_valid_only"` | Use the validation split as test set | `false` |
| `"start_at_task"` | First task to run (for resuming) | `0` |
| `"stop_at_task"` | Last task to run exclusive (0 = run all) | `0` |

Example:
```json
{
    "dataset": "cifar100",
    "data_path": "../data/cifar100",
    "increment": 10,
    "batch_size": 64,
    "num_workers": 2
}
```

## Transforms
Transforms are also configured via top-level JSON keys:

| Key | Train | Eval | Default |
|-----|-------|------|---------|
| `"resize"` | Resize (BICUBIC) | Resize (BICUBIC) | `224` |
| `"crop"` | RandomResizedCrop | CenterCrop | `null` |
| `"pad"` | Pad | Pad | `null` |
| `"flip"` | RandomHorizontalFlip (p=0.5) | — | `true` |
| `"normalize"` | Normalize with preset mean/std | same | `"in1k"` |

Normalize presets:
* `"in1k"` — ImageNet-1k standard: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`
* `"in21k"` — ImageNet-21k: mean `(0.5, 0.5, 0.5)`, std `(0.5, 0.5, 0.5)`

### Exemplars
For approaches that use exemplars, configure them under `"exemplars_args"`:
```json
"exemplars_args": {
    "num_exemplars": 2000,
    "exemplar_selection": "random"
}
```

| Key | Description |
|-----|-------------|
| `"num_exemplars"` | Fixed total exemplar budget across all seen classes |
| `"num_exemplars_per_class"` | Growing memory: exemplars per class (mutually exclusive with `"num_exemplars"`) |
| `"exemplar_selection"` | Strategy: `"random"`, `"herding"`, `"entropy"`, `"distance"` |

## Adding new datasets
To add a new dataset, follow this:

1. **In-memory datasets** (small enough to fit in RAM, e.g. CIFAR-100): add a new branch in `get_dataset` inside [data_loader.py](data_loader.py) following the existing `cifar100` or `mnist` examples. Load train/test as `{'x': ..., 'y': ...}` numpy arrays and call `memd.get_data(...)`.

2. **Custom path-based datasets** (too large for memory, e.g. Food-101, iNat200): no code change needed. Use any string that is not already a built-in keyword as `"dataset"`. Place `train.txt` and `test.txt` in `data_path`. Each line must contain the image path and class label separated by a space:
   ```
   /data/train/sample1.jpg 0
   /data/train/sample2.jpg 1
   /data/train/sample3.jpg 2
   ...
   ```

## Dataset types
* **`MemoryDataset`** — loads all images into RAM at startup. Used for MNIST, CIFAR-100, SVHN, and similar small datasets.
* **`BaseDataset`** — loads image paths into RAM; images are read from disk on demand via the DataLoader. Used for large datasets (Food-101, iNat200, ImageNet, etc.).
