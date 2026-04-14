import torch
from torch import nn
from copy import deepcopy

class LLL_Net_Distilled(nn.Module):
    """Basic class for implementing distilation networks"""

    def __init__(self, model, remove_existing_head=False):

        # Set head_var e.g. fc or head
        head_var = model.head_var
        assert type(head_var) == str
        
        # If remove_existing_head=True Check that model has attribute with the given name
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        # If remove_existing_head=True Check if head is either nn.Linear or nn.Sequential or nn.Identity
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear, nn.Identity], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        
        # Contstructor call
        super(LLL_Net_Distilled, self).__init__()

        # Set model and classification head
        self.model = model

        self.out_size = model.num_features

        # Module List to store heads for tasks.
        self.heads = nn.ModuleList()
        self.heads_dist = nn.ModuleList()
        # Accounting for classes per task
        self.task_cls = []
        # Offset for classes
        self.task_offset = []
        # TODO Fix weights initialisation placemant 
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        self.heads_dist.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):

        outputs_backbone = self.model(x)
        assert len(self.heads) > 0, "Cannot access any head"

        # cls_features = outputs_backbone["cls_features"]
        # dist_features = outputs_backbone["dist_features"]
        cls_features, dist_features = outputs_backbone

        cls_logits=[]
        dist_logits=[]

        for head in self.heads:
            cls_logits.append(head(cls_features))

        for head_dist in self.heads_dist:
            dist_logits.append(head_dist(dist_features))

        if return_features:
            return {
                "cls_logits": cls_logits,
                "dist_logits": dist_logits,
                "cls_features": cls_features,
                "dist_features": dist_features,
            }
        return {
            "cls_logits": cls_logits,
            "dist_logits": dist_logits,
        }

    # Used for knowledge distilation methods
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
