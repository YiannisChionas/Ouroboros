import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .incremental_learning import Incremental_Learning_Approach
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Incremental_Learning_Approach):
    """Deep Model Consolidation (DMC)
    https://arxiv.org/abs/1903.07864

    No exemplars. Uses an auxiliary dataset for model consolidation.

    Per task (t > 0):
      Phase 1 — New task: train model_new (fresh pretrained backbone) on new task data with CE
      Phase 2 — Student: train main model on auxiliary data with double distillation (MSE)
                         Target = normalized concat(model_old logits[:t], model_new logits[t])

    Approach-specific args are read from args['approach_args']:
        aux_data_path  (str,   required)              — path to auxiliary dataset (ImageFolder format)
        aux_batch_size (int,   default 128)           — batch size for auxiliary loader
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        aargs = args.get('approach_args', {})
        self.aux_data_path  = aargs.get('aux_data_path')
        self.aux_batch_size = aargs.get('aux_batch_size', 128)

        assert self.aux_data_path is not None, "DMC requires aux_data_path in approach_args"

        # Save pretrained backbone weights so we can restore them for each new task
        self.pretrained_state = deepcopy(self.model.model.state_dict())

        self.model_old = None
        self.model_new = None

        # Build aux loaders — same transforms as main dataset (resize/normalize from args)
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
                    if args.get('normalize', 'in1k') == 'in1k' \
                    else ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        aux_transform = transforms.Compose([
            transforms.Resize(args.get('resize', 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.get('crop', 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        aux_full = datasets.ImageFolder(root=self.aux_data_path, transform=aux_transform)
        n_val = max(1, int(0.1 * len(aux_full)))
        n_trn = len(aux_full) - n_val
        trn_ds, val_ds = random_split(
            aux_full, [n_trn, n_val],
            generator=torch.Generator().manual_seed(args.get('seed', 0))
        )
        nw = args.get('num_workers', 2)
        pm = args.get('pin_memory', False)
        self.aux_trn_loader = DataLoader(trn_ds, batch_size=self.aux_batch_size, shuffle=True,  num_workers=nw, pin_memory=pm)
        self.aux_val_loader = DataLoader(val_ds, batch_size=self.aux_batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    @staticmethod
    def exemplars_dataset_class():
        return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def pre_train_process(self, t, trn_loader):
        """For t > 0: restore pretrained backbone and create model_new."""
        if t == 0:
            return

        # Restore pretrained backbone — no training history
        self.model.model.load_state_dict(deepcopy(self.pretrained_state))
        # Reset all heads to fresh random init
        for h in self.model.heads:
            h.reset_parameters()

        # model_new = fresh copy with old heads zeroed + frozen (only trains on new task)
        self.model_new = deepcopy(self.model)
        for h in self.model_new.heads[:-1]:
            with torch.no_grad():
                h.weight.zero_()
                h.bias.zero_()
            for p in h.parameters():
                p.requires_grad = False

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0:
            super().train_loop(t, trn_loader, val_loader)
            return

        # Phase 1: train model_new on new task data — swap model temporarily
        print('=' * 108)
        print('DMC Phase 1: new task training')
        print('=' * 108)
        self.model, self.model_new = self.model_new, self.model
        super().train_loop(t, trn_loader, val_loader)   # base class CE loop on model_new
        self.model, self.model_new = self.model_new, self.model
        self.model_new.eval()
        self.model_new.freeze_all()

        # Phase 2: train student (self.model) on auxiliary dataset
        print('=' * 108)
        print('DMC Phase 2: student training')
        print('=' * 108)
        self._student_loop(t)

    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # ------------------------------------------------------------------
    # Student training
    # ------------------------------------------------------------------

    def _student_loop(self, t):
        """Train student (self.model) on aux data with double distillation MSE loss."""
        best_loss  = np.inf
        best_state = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        self._log_trainable_params()
        scheduler = self._create_scheduler()

        for e in range(self.nepochs):
            clock0 = time.time()
            self.model.train()
            for images, _ in self.aux_trn_loader:
                images = images.to(self.device)
                with torch.no_grad():
                    out_old = self.model_old(images)
                    out_new = self.model_new(images)
                outputs = self.model(images)
                loss = self._student_criterion(t, outputs, out_old, out_new)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
                self.optimizer.step()
            clock1 = time.time()

            val_loss = self._eval_student(t)
            clock2   = time.time()

            print(f'| [Student] Epoch {e+1:3d}, time={clock1-clock0:.1f}s/{clock2-clock1:.1f}s'
                  f' | val_loss={val_loss:.4f} |', end='')
            self.logger.log_scalar(task=t, iter=e+1, name="student_loss", value=val_loss, group="valid")

            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = self.model.get_copy()
                print(' *', end='')

            if scheduler:
                scheduler.step(e+1, metric=val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            print(f' lr={lr:.1e}')
            self.logger.log_scalar(task=t, iter=e+1, name="lr", value=lr, group="train")

            if self.lr_min and lr < self.lr_min:
                break

        self.model.set_state_dict(best_state)

    def _eval_student(self, t):
        total_loss, total_num = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for images, _ in self.aux_val_loader:
                images = images.to(self.device)
                out_old = self.model_old(images)
                out_new = self.model_new(images)
                outputs = self.model(images)
                loss    = self._student_criterion(t, outputs, out_old, out_new)
                total_loss += loss.item() * len(images)
                total_num  += len(images)
        return total_loss / total_num

    def _student_criterion(self, t, outputs, out_old, out_new):
        """Eq. 3: Double Distillation Loss — MSE against normalized concat of teacher logits."""
        with torch.no_grad():
            # Eq. 4: concat old-task logits + new-task logits, subtract column mean
            target = torch.cat(out_old[:t] + [out_new[t]], dim=1)
            target = target - target.mean(0)
        return F.mse_loss(torch.cat(outputs, dim=1), target.detach())

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def save_progress(self, results_path, task):
        """Save pretrained backbone state — required for correct model_new init on resume."""
        import os
        torch.save(self.pretrained_state,
                   os.path.join(results_path, 'pretrained_backbone.pth'))

    def load_progress(self, results_path, task):
        """Restore model_old and pretrained backbone state on resume."""
        import os
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        pt_file = os.path.join(results_path, 'pretrained_backbone.pth')
        if os.path.isfile(pt_file):
            self.pretrained_state = torch.load(pt_file, weights_only=True)
            print(f"Loaded pretrained backbone state from {pt_file}")
        else:
            warnings.warn(
                f"pretrained_backbone.pth not found at {pt_file} — "
                "pretrained_state will be the currently loaded (trained) weights, "
                "which may cause incorrect model_new initialization."
            )
