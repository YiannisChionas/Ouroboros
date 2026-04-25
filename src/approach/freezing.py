from .finetuning import Appr as FinetuningAppr


class Appr(FinetuningAppr):
    """Frozen backbone baseline — identical to Finetuning but backbone is frozen by default.

    Equivalent to running Finetuning with freeze_backbone=True and fix_bn=True.
    Only the linear heads are trained; the pretrained ViT weights are not updated.
    """

    def __init__(self, args, model, logger=None, exemplars_dataset=None):
        args.setdefault('freeze_backbone', True)
        args.setdefault('fix_bn', True)
        super().__init__(args, model, logger, exemplars_dataset)
