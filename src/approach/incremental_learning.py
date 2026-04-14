import os
import time
import torch
import warnings
import numpy as np
from copy import deepcopy
import timm.optim

from timm.scheduler import create_scheduler_v2

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset


class Incremental_Learning_Approach:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self,
                 args,
                 model,
                 logger: ExperimentLogger = None,
                 exemplars_dataset: ExemplarsDataset = None):

        self.model =             model
        self.device =            args['device']
        self.nepochs =           args['nepochs']
        self.lr_scheduler =      args['lr_scheduler']
        self.optimizer_name =    args['optimizer_name']
        self.lr =                args['lr']
        self.lr_min =            args['lr_min']
        self.lr_factor =         args['lr_factor']
        self.lr_patience =       args['lr_patience']
        self.lr_warmup_epochs =  args['lr_warmup_epochs']
        self.clipping =          args['clipping']
        self.momentum =          args['momentum']
        self.weight_decay =      args['weight_decay']
        self.multi_softmax =     args['multi_softmax']
        self.logger =            logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs =     args['warmup_nepochs']
        self.warmup_lr_factor =  self.lr * args['warmup_lr_factor']
        self.warmup_loss =       torch.nn.CrossEntropyLoss()
        self.fix_bn =            args['fix_bn']
        self.freeze_backbone =   args['freeze_backbone']
        self.eval_on_train =     args['eval_on_train']
        self.optimizer =         None

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        return timm.optim.create_optimizer_v2(
                                              params,
                                              opt=self.optimizer_name,
                                              lr=self.lr,
                                              weight_decay=self.weight_decay,
                                              momentum=self.momentum,
                                              nesterov=False,
                                              )

    def _log_trainable_params(self):
        """Log trainable vs total parameter counts for the current optimizer."""
        params_all = list(self.model.parameters())
        params = [p for p in params_all if p.requires_grad]
        total_numel     = sum(p.numel() for p in params_all)
        trainable_numel = sum(p.numel() for p in params)
        print(f"[params] total:{total_numel:,}  trainable:{trainable_numel:,}"
              f"  tensors:{len(params_all)}  trainable_tensors:{len(params)}")

    def train(self, task, train_loader, validation_loader, **kwargs):
        """Main train structure. Extra kwargs (e.g. start_epoch, stop_epoch) are ignored by default."""
        self.pre_train_process(task, train_loader)
        self.train_loop(task, train_loader, validation_loader)
        self.post_train_process(task, train_loader)

    def pre_train_process(self, task, train_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and task > 0:
            self.optimizer = timm.optim.create_optimizer_v2(
                                                            self.model.heads[-1].parameters(),
                                                            opt=self.optimizer_name,
                                                            lr=self.warmup_lr_factor,
                                                            weight_decay=self.weight_decay,
                                                            momentum=self.momentum,
                                                            )
            # Loop epochs -- train warm-up head
            for epoch in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in train_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[task], targets.to(self.device) - self.model.task_offset[task])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipping)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in train_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[task], targets.to(self.device) - self.model.task_offset[task])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(train_loader.dataset.labels)
                train_loss, train_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    epoch + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, train_loss, 100 * train_acc))
                self.logger.log_scalar(task=task, iter=epoch + 1, name="loss", value=train_loss, group="warmup")
                self.logger.log_scalar(task=task, iter=epoch + 1, name="acc", value=100 * train_acc, group="warmup")

    def _create_scheduler(self):
        """Create a timm scheduler from self.lr_scheduler and related hyperparams.

        timm's create_scheduler_v2 accepts the scheduler name and all relevant
        kwargs — we just forward what we have and let timm handle the rest.
        Passing None values is safe; timm ignores them when not applicable.
        """
        if not self.lr_scheduler:
            return None

        scheduler, _ = create_scheduler_v2(
                                          self.optimizer,
                                          sched=self.lr_scheduler,        # 'cosine', 'plateau', 'step', …
                                          num_epochs=self.nepochs,
                                          min_lr=self.lr_min,
                                          warmup_epochs=self.lr_warmup_epochs,
                                          warmup_lr=self.warmup_lr_factor,
                                          # plateau-specific (ignored by other schedulers)
                                          decay_rate=self.lr_factor,
                                          patience_epochs=self.lr_patience,
                                          )
        return scheduler

    def train_loop(self, task, train_loader, validation_loader):
        """Contains the epochs loop"""
        best_loss = np.inf
        best_model = self.model.get_copy()

        if self.freeze_backbone:
            self.model.freeze_backbone()
        if self.fix_bn and task > 0:
            self.model.freeze_bn()
        self.optimizer = self._get_optimizer()
        self._log_trainable_params()
        scheduler = self._create_scheduler()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(task, train_loader)
            clock1 = time.time()

            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(task, train_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=task, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=task, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(task, validation_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=task, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=task, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Track best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                print(' *', end='')

            # Scheduler step — timm handles cosine/plateau/etc. uniformly
            if scheduler is not None:
                scheduler.step(e + 1, metric=valid_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(f' lr={lr:.1e}')
            self.logger.log_scalar(task=task, iter=e + 1, name="lr", value=lr, group="train")

            # Early stopping: halt if lr has decayed below minimum
            if self.lr_min is not None and lr < self.lr_min:
                break

        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def save_checkpoint_state(self, models_dir, task):
        # Base implementation does nothing.
        # Override in subclass to persist approach-specific state (e.g. exemplars, fisher).
        pass

    def load_progress(self, results_path, task):
        # Base implementation does nothing.
        # Override in subclass to restore approach-specific state on resume.
        pass

    def save_progress(self, results_path, task):
        # Base implementation does nothing.
        # Override in subclass if your approach needs to persist state on pause
        # (e.g. exemplars, model_old, fisher matrices) so that resume works correctly.
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            # One liner to find task index through label
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            # Local label + offset
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        # Get hits and turn them to float
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            # log_softmax normalizes logits within each head before concatenation —
            # necessary when heads have different numbers of classes (different scales)
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
