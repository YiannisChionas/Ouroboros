import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .incremental_learning_v3 import Incremental_Learning_Approach
from datasets.exemplars_selection import override_dataset_transform


class Appr(Incremental_Learning_Approach):
    """Simple Class Incremental Learning (SimpleCIL)
    https://arxiv.org/pdf/2303.07338
    No training — backbone frozen, classification via cosine nearest-prototype.

    Prototype computation uses test transforms (no augmentation), matching the
    original RevisitingCIL implementation.
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self.prototypes: dict[int, torch.Tensor] = {}

    def _get_optimizer(self):
        return None

    def train_loop(self, t, trn_loader, val_loader):
        print(f"\n[Task {t}] Computing prototypes ...", flush=True)
        self.model.eval()
        self.model.freeze_all()

        test_transform = val_loader.dataset.transform
        with override_dataset_transform(trn_loader.dataset, test_transform) as ds:
            proto_loader = DataLoader(ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            self.compute_prototypes(t, proto_loader)

    def compute_prototypes(self, t, loader):
        feat_sum = {}
        count = {}

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                feats = self.model.model(images)
                for i in range(targets.shape[0]):
                    lbl = int(targets[i].item())
                    if lbl not in feat_sum:
                        feat_sum[lbl] = feats[i].detach()
                        count[lbl] = 1
                    else:
                        feat_sum[lbl] += feats[i].detach()
                        count[lbl] += 1

        for cls_idx, s in feat_sum.items():
            proto = s / float(count[cls_idx])
            proto = F.normalize(proto, p=2, dim=0)
            self.prototypes[int(cls_idx)] = proto

        print(f"   -> Stored {len(feat_sum)} prototypes. Total seen classes: {len(self.prototypes)}", flush=True)

    def eval(self, t, val_loader):
        """Global cosine NCM evaluation using prototypes."""
        self.model.eval()

        sorted_class_ids = sorted(self.prototypes.keys())
        W = torch.stack([self.prototypes[c] for c in sorted_class_ids]).to(self.device)
        class_map = torch.tensor(sorted_class_ids, device=self.device)

        # TAw: indices restricted to current task
        offset = int(self.model.task_offset[t].item()) if hasattr(self.model.task_offset[t], 'item') else int(self.model.task_offset[t])
        t_cls = int(self.model.task_cls[t].item()) if hasattr(self.model.task_cls[t], 'item') else int(self.model.task_cls[t])
        task_range = set(range(offset, offset + t_cls))
        taw_indices = torch.tensor([i for i, c in enumerate(sorted_class_ids) if c in task_range],
                                   device=self.device, dtype=torch.long)

        total_num, correct_taw, correct_tag = 0, 0, 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                feats = F.normalize(self.model.model(images), p=2, dim=1)
                logits = feats @ W.T

                pred_tag = class_map[torch.argmax(logits, dim=1)]
                correct_tag += int((pred_tag == targets).sum().item())

                if taw_indices.numel() > 0:
                    pred_taw = class_map[taw_indices[torch.argmax(logits[:, taw_indices], dim=1)]]
                    correct_taw += int((pred_taw == targets).sum().item())

                total_num += int(targets.shape[0])

        acc_taw = correct_taw / float(total_num) if total_num > 0 else 0.0
        acc_tag = correct_tag / float(total_num) if total_num > 0 else 0.0
        return 0.0, acc_taw, acc_tag

    def save_progress(self, results_path, task):
        """Save prototypes so resume works correctly."""
        torch.save(self.prototypes, os.path.join(results_path, f"task{task}_prototypes.pth"))

    def load_progress(self, results_path, task):
        """Restore prototypes on resume."""
        proto_file = os.path.join(results_path, f"task{task}_prototypes.pth")
        if os.path.isfile(proto_file):
            self.prototypes = torch.load(proto_file, weights_only=False)
            print(f"Loaded {len(self.prototypes)} prototypes from {proto_file}")
        else:
            import warnings
            warnings.warn(f"Prototypes file NOT found at {proto_file}!")
