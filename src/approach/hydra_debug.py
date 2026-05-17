import torch
import torch.nn.functional as F

from .hydra import Appr as HydraAppr


class Appr(HydraAppr):
    """Hydra + feature-drift logging.

    Captures dist_features on the first training batch before and after
    each task and prints cosine/L2 distances.
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        super().__init__(args, model, logger, exemplars_dataset)
        self._drift_images   = None
        self._drift_pre_feat = None

    def pre_train_process(self, t, train_loader):
        super().pre_train_process(t, train_loader)
        images, _ = next(iter(train_loader))
        self.model.eval()
        with torch.no_grad():
            out = self.model(images.to(self.device), return_features=True)
        self._drift_images   = images
        self._drift_pre_feat = out['dist_features'].detach()

    def post_train_process(self, t, trn_loader):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._drift_images.to(self.device), return_features=True)
        post_feat = out['dist_features'].detach()

        cos_dist = (1 - F.cosine_similarity(self._drift_pre_feat, post_feat, dim=-1)).mean().item()
        l2_dist  = (self._drift_pre_feat - post_feat).norm(dim=-1).mean().item()
        print(f'[drift] task {t} | cos_dist={cos_dist:.4f} | l2_dist={l2_dist:.4f}')

        super().post_train_process(t, trn_loader)
