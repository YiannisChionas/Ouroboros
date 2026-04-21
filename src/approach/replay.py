import os
import warnings
import torch
import torch.nn.functional as F

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Experience Replay baseline — finetuning + exemplar replay.

    Training: current task data concatenated with stored exemplars (t > 0).
    Loss:     flat CE across all seen heads.
    Inference: standard logit-based (no NCM, no KD).

    Exemplar args (set via args['exemplars_args'] in the trainer):
        num_exemplars           (int) — total exemplar budget
        num_exemplars_per_class (int) — per-class budget (mutually exclusive)
        exemplar_selection      (str) — 'herding', 'random', 'entropy', 'distance'
    """

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def train_loop(self, t, trn_loader, val_loader):
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )
        super().train_loop(t, trn_loader, val_loader)
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        return F.cross_entropy(torch.cat(outputs[:t + 1], dim=1), targets)

    def save_progress(self, results_path, task):
        torch.save(
            {'images': self.exemplars_dataset.images, 'labels': self.exemplars_dataset.labels},
            os.path.join(results_path, f"task{task}_exemplars.pth"),
        )

    def load_progress(self, results_path, task):
        exemplars_file = os.path.join(results_path, f"task{task}_exemplars.pth")
        if os.path.isfile(exemplars_file):
            state = torch.load(exemplars_file, weights_only=False)
            self.exemplars_dataset.images = state['images']
            self.exemplars_dataset.labels = state['labels']
            print(f"Loaded {len(self.exemplars_dataset.images)} exemplars from {exemplars_file}")
        else:
            warnings.warn(f"Exemplars file NOT found at {exemplars_file}!")
