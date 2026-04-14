import torch
from torch import nn
from copy import deepcopy


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):

        # Set head_var e.g. fc or head
        head_var = model.head_var
        assert type(head_var) == str
        
        # If remove_existing_head=True Check that model has attribute with the given name
        # assert not remove_existing_head or hasattr(model, head_var), \
        #     "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or has_recursive_attr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        # If remove_existing_head=True Check if head is either nn.Linear or nn.Sequential or nn.Identity
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear, nn.Identity], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        
        # Contstructor call
        super(LLL_Net, self).__init__()

        # Set model and classification head
        self.model = model
        # last_layer = getattr(self.model, head_var)
        last_layer = get_recursive_attr(self.model, head_var)

        if remove_existing_head:
            # Case 1: Head is a Sequential block (e.g., VGG style)
            # We locate the last layer (classifier), save its input dimension (feature size), 
            # and remove it to turn the model into a feature extractor.
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                del last_layer[-1]
            # Case 2: Head is a single Linear layer (e.g., ResNet/ViT style)
            # We save the input dimension (feature size) and replace the layer 
            # with an Identity mapping (empty Sequential).
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # Replace the Linear layer with an empty Sequential block.
                # Note: An empty nn.Sequential() acts exactly like nn.Identity() 
                # (passes input through unchanged) but is compatible with older PyTorch versions (<1.2).
                # setattr(self.model, head_var, nn.Sequential())
                set_recursive_attr(self.model, head_var, nn.Sequential())
            elif type(last_layer) == nn.Identity:
                if hasattr(model, 'num_features'):
                    self.out_size = model.num_features
                else:
                    raise ValueError("Model has Identity head but no 'num_features' or 'embed_dim' attribute to determine output size.")
        else:
            # Case 3: Keep the existing head
            # We simply store the output dimension (number of classes).
            self.out_size = model.num_features

        # Module List to store heads for tasks.
        self.heads = nn.ModuleList()
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
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        # Calls forward function of backbone. 
        # Since 'fc' is replaced by an empty Sequential, this returns FEATURES (768), not logits.
        features = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        logits = []
        for head in self.heads:
            logits.append(head(features))
        if return_features:
            return logits, features
        else:
            return logits

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

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False
        n_bb = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_heads = sum(p.numel() for h in self.heads for p in h.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DEBUG] trainable params -> backbone:{n_bb} heads:{n_heads} total:{n_all}")

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass


def get_recursive_attr(obj, attr):
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj

def set_recursive_attr(obj, attr, value):
    parts = attr.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

def has_recursive_attr(obj, attr):
    try:
        for part in attr.split('.'):
            obj = getattr(obj, part)
        return True
    except AttributeError:
        return False