import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .incremental_learning_v3 import Incremental_Learning_Approach

# Joint training baseline — upper bound for continual learning.
# Trains on the union of all datasets seen so far at each task.
#
# Epoch-based resume is supported via save_progress/load_progress:
# the accumulated trn/val datasets are serialised as numpy arrays so they
# survive across SLURM jobs.  The scheduler and optimizer state are also
# checkpointed so the cosine curve is not reset mid-task.


class Appr(Incremental_Learning_Approach):

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.trn_datasets = []
        self.val_datasets = []

    def _get_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

    # ------------------------------------------------------------------
    # Override train() to accept start_epoch / stop_epoch
    # ------------------------------------------------------------------

    def train(self, task, train_loader, val_loader, start_epoch=0, stop_epoch=0):
        self.pre_train_process(task, train_loader)
        self.train_loop(task, train_loader, val_loader,
                        start_epoch=start_epoch, stop_epoch=stop_epoch)
        # post_train_process only runs when the full task is done
        if stop_epoch == 0 or stop_epoch == self.nepochs:
            self.post_train_process(task, train_loader)

    # ------------------------------------------------------------------
    # train_loop with epoch resume
    # ------------------------------------------------------------------

    def train_loop(self, task, trn_loader, val_loader, start_epoch=0, stop_epoch=0):
        import time
        stop_epoch = stop_epoch if stop_epoch != 0 else self.nepochs

        # Accumulate datasets (only on first epoch-job for this task)
        if start_epoch == 0:
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)

        # Patch transforms: load_progress restores datasets with transform=None
        for ds in self.trn_datasets:
            if ds.transform is None:
                ds.transform = trn_loader.dataset.transform
        for ds in self.val_datasets:
            if ds.transform is None:
                ds.transform = val_loader.dataset.transform

        joint_trn = DataLoader(JointDataset(self.trn_datasets),
                               batch_size=trn_loader.batch_size,
                               shuffle=True,
                               num_workers=trn_loader.num_workers,
                               pin_memory=trn_loader.pin_memory)
        joint_val = DataLoader(JointDataset(self.val_datasets),
                               batch_size=val_loader.batch_size,
                               shuffle=False,
                               num_workers=val_loader.num_workers,
                               pin_memory=val_loader.pin_memory)

        if self.freeze_backbone:
            self.model.freeze_backbone()

        # Restore or create optimizer + scheduler
        if start_epoch > 0:
            self.optimizer = self._get_optimizer()
            self._load_optimizer_state(task)
            scheduler = self._create_scheduler()
            self._load_scheduler_state(task, scheduler)
            best_loss = self._load_best_loss(task)
            best_model = self._load_best_model(task)
        else:
            self.optimizer = self._get_optimizer()
            scheduler = self._create_scheduler()
            best_loss = float('inf')
            best_model = self.model.get_copy()
        self._log_trainable_params()

        for e in range(start_epoch, stop_epoch):
            clock0 = time.time()
            self.train_epoch(task, joint_trn)
            clock1 = time.time()

            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(task, joint_trn)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=task, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=task, iter=e + 1, name="acc",  value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(task, joint_val)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=task, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=task, iter=e + 1, name="acc",  value=100 * valid_acc, group="valid")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                print(' *', end='')

            if scheduler is not None:
                scheduler.step(e + 1, metric=valid_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(f' lr={lr:.1e}')
            self.logger.log_scalar(task=task, iter=e + 1, name="lr", value=lr, group="train")

            if self.lr_min is not None and lr < self.lr_min:
                break

        # Always restore best model found so far and checkpoint it
        self.model.set_state_dict(best_model)
        self._save_optimizer_state(task)
        if scheduler is not None:
            self._save_scheduler_state(task, scheduler)
        self._save_best_loss(task, best_loss)
        self._save_best_model(task, best_model)

    # ------------------------------------------------------------------
    # Criterion
    # ------------------------------------------------------------------

    def criterion(self, t, outputs, targets):
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

    # ------------------------------------------------------------------
    # save_progress / load_progress  (datasets + epoch state)
    # ------------------------------------------------------------------

    def save_progress(self, results_path, task):
        """Persist accumulated datasets as numpy arrays."""
        save_dir = os.path.join(results_path, "joint_progress")
        os.makedirs(save_dir, exist_ok=True)
        for split, datasets in [("trn", self.trn_datasets), ("val", self.val_datasets)]:
            all_x, all_y = [], []
            for ds in datasets:
                xs, ys = _extract_xy(ds)
                all_x.append(xs)
                all_y.append(ys)
            np.save(os.path.join(save_dir, f"{split}_x_task{task}.npy"), np.concatenate(all_x, axis=0))
            np.save(os.path.join(save_dir, f"{split}_y_task{task}.npy"), np.concatenate(all_y, axis=0))
        print(f"[joint] Saved accumulated datasets up to task {task} → {save_dir}")

    def load_progress(self, results_path, task):
        """Restore accumulated datasets from disk."""
        save_dir = os.path.join(results_path, "joint_progress")
        transform    = None  # set below from first dataset if needed
        class_indices = None

        for split in ("trn", "val"):
            x = np.load(os.path.join(save_dir, f"{split}_x_task{task}.npy"))
            y = np.load(os.path.join(save_dir, f"{split}_y_task{task}.npy"))
            ds = MemoryDataset(x, y, transform=transform)
            if split == "trn":
                self.trn_datasets = [ds]
            else:
                self.val_datasets = [ds]
        print(f"[joint] Loaded accumulated datasets up to task {task} from {save_dir}")

    # ------------------------------------------------------------------
    # Epoch-level checkpoint helpers
    # ------------------------------------------------------------------

    def _epoch_ckpt_dir(self, task):
        d = os.path.join(self.logger.exp_path, "joint_epoch_ckpt", f"task{task}")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_optimizer_state(self, task):
        path = os.path.join(self._epoch_ckpt_dir(task), "optimizer.pt")
        torch.save(self.optimizer.state_dict(), path)

    def _load_optimizer_state(self, task):
        path = os.path.join(self._epoch_ckpt_dir(task), "optimizer.pt")
        if os.path.isfile(path):
            self.optimizer.load_state_dict(torch.load(path, map_location=self.device))

    def _save_scheduler_state(self, task, scheduler):
        path = os.path.join(self._epoch_ckpt_dir(task), "scheduler.pt")
        torch.save(scheduler.state_dict(), path)

    def _load_scheduler_state(self, task, scheduler):
        path = os.path.join(self._epoch_ckpt_dir(task), "scheduler.pt")
        if os.path.isfile(path) and scheduler is not None:
            scheduler.load_state_dict(torch.load(path, map_location=self.device))

    def _save_best_loss(self, task, best_loss):
        path = os.path.join(self._epoch_ckpt_dir(task), "best_loss.npy")
        np.save(path, np.array([best_loss]))

    def _load_best_loss(self, task):
        path = os.path.join(self._epoch_ckpt_dir(task), "best_loss.npy")
        if os.path.isfile(path):
            return float(np.load(path)[0])
        return float('inf')

    def _save_best_model(self, task, best_model):
        path = os.path.join(self._epoch_ckpt_dir(task), "best_model.pt")
        torch.save(best_model, path)

    def _load_best_model(self, task):
        path = os.path.join(self._epoch_ckpt_dir(task), "best_model.pt")
        if os.path.isfile(path):
            return torch.load(path, map_location=self.device)
        return self.model.get_copy()


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

class JointDataset(Dataset):
    """Concatenates multiple task datasets into one."""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum(len(d) for d in datasets)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)


class MemoryDataset(Dataset):
    """Simple dataset wrapping raw numpy arrays, used for resume."""

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        from PIL import Image
        import numpy as np
        x = self.x[index]
        if isinstance(x, (str, np.str_)):
            img = Image.open(x).convert('RGB')
        else:
            img = Image.fromarray(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[index])


def _extract_xy(dataset):
    """Extract raw (x, y) arrays from a dataset, handling common wrappers."""
    # Try common attribute names used in this codebase
    if hasattr(dataset, 'x') and hasattr(dataset, 'y'):
        return np.array(dataset.x), np.array(dataset.y)
    if hasattr(dataset, 'images') and hasattr(dataset, 'labels'):
        return np.array(dataset.images), np.array(dataset.labels)
    if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        return np.array(dataset.data), np.array(dataset.targets)
    # Fallback: iterate (slow but safe)
    xs, ys = [], []
    for x, y in dataset:
        xs.append(np.array(x))
        ys.append(y)
    return np.stack(xs), np.array(ys)
