# Approaches
We include the following baselines:
* Finetuning ~ Finetuning the backbone along with the heads in each task without forgetting limitations.
* Freezing   ~ Finetuning the heads while keeping the backbone frozen. Serves as an indicator of the backbone's pretraining quality while using a LinearLayer classification head.
* SimpleCIL  ~ Keeps the backbone frozen while using a nearest class mean classifier on the features provided by the backbone for a specific task. Serves as an indicator of the backbone's pretraining quality.
* Replay     ~ Finetuning with the addition of exemplars. Serves as the lower bound for replay methods.
* Joint      ~ Incremental Joint Training. Serves as the upper bound.

The regularization-based approaches available:
* EWC
* LwF
* DMC (Not Ready)

The rehearsal approaches available:
* iCaRL
* EEIL

The bias-correction approaches available:
* BiC
* LUCIR

The prompting approaches available:
* L2P
* DualPrompt (Not Ready)

The distillation token approaches introduced in this repository:
* LWF Dual
* Hydra

## Main usage
All configuration is done through JSON config files. The approach is selected with:
```json
"approach": "approach_name"
```
Each approach is loaded by its respective `*.py` filename. All approaches inherit from
`Incremental_Learning_Approach`, which reads the following top-level config keys:

| Key | Description | Default |
|-----|-------------|---------|
| `"nepochs"` | Number of training epochs per task | `50` |
| `"optimizer_name"` | Optimizer name (timm-compatible, e.g. `"sgd"`, `"adamw"`) | `"sgd"` |
| `"lr_scheduler"` | LR scheduler (`"cosine"`, `"plateau"`, `"step"`, or `null`) | `null` |
| `"lr"` | Initial learning rate | `0.1` |
| `"lr_min"` | Minimum LR — training stops early when reached | `null` |
| `"lr_patience"` | Patience epochs for plateau scheduler | `null` |
| `"clipping"` | Gradient norm clip value | `10000` |
| `"momentum"` | Optimizer momentum | `0.0` |
| `"weight_decay"` | Weight decay (L2 penalty) | `0.0` |
| `"warmup_nepochs"` | Head-only warm-up epochs before full training (task > 0 only) | `0` |
| `"warmup_lr_factor"` | LR multiplier used during warm-up | `1.0` |
| `"multi_softmax"` | Apply per-head softmax before TAg argmax | `false` |
| `"fix_bn"` | Freeze BatchNorm layers after task 0 | `false` |
| `"eval_on_train"` | Log training loss/accuracy each epoch | `false` |

Approach-specific hyperparameters are defined under `"approach_args"`:
```json
"approach_args": {
    "lamb": 1.0,
    "T": 2
}
```

### Allowing rehearsal
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
| `"exemplar_selection"` | Selection strategy: `"random"`, `"herding"`, `"entropy"`, `"distance"` |

## Adding new approaches
To add a new approach, follow this:

1. Create a new file similar to [finetuning.py](finetuning.py). The filename (without `.py`) is the value used for `"approach"` in the config.
2. Implement the method by subclassing `Incremental_Learning_Approach` and overriding the necessary methods (`criterion`, `train_epoch`, `pre_train_process`, `post_train_process`).
3. Read approach-specific arguments from `args.get('approach_args', {})` in `__init__`. Do not modify `calculate_metrics()` unless strictly necessary to keep metrics comparable.

## Baselines

### Finetuning
```json
"approach": "finetuning"
```
Learning approach which learns each task incrementally while not using any data or knowledge from previous tasks.
By default, weights corresponding to the outputs of previous classes are not updated. This can be changed by
setting `"all_outputs": true` inside `"approach_args"`. This approach allows the use of exemplars.

### Freezing
```json
"approach": "freezing"
```
Learning approach which freezes the backbone after training the first task so only the heads are learned.
Set `"fix_bn": true` and `"freeze_backbone": true` in the top-level config to activate this behaviour.

### SimpleCIL
```json
"approach": "simplecil"
```
Learning approach which uses a pretrained frozen backbone as a feature extractor. Classification is achieved
using a nearest class mean classifier on the extracted features.

### Incremental Joint Training
```json
"approach": "joint"
```
Learning approach which has access to all data from all tasks and serves as an upper-bound baseline.

## Approaches

### Learning without Forgetting
```json
"approach": "lwf"
```
[arxiv](https://arxiv.org/abs/1606.09282)
| [TPAMI 2017](https://ieeexplore.ieee.org/document/8107520)

```json
"approach_args": {
    "lamb": 1.0,
    "T": 2
}
```
* `"lamb"`: forgetting-intransigence trade-off (default=1.0)
* `"T"`: softmax temperature for knowledge distillation (default=2)

### iCaRL
```json
"approach": "icarl"
```
[arxiv](https://arxiv.org/abs/1611.07725)
| [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)

```json
"approach_args": {
    "lamb": 1.0
},
"exemplars_args": {
    "num_exemplars": 2000,
    "exemplar_selection": "random"
}
```
* `"lamb"`: forgetting-intransigence trade-off (default=1.0)

### Elastic Weight Consolidation
```json
"approach": "ewc"
```
[arxiv](http://arxiv.org/abs/1612.00796)
| [PNAS 2017](https://www.pnas.org/content/114/13/3521)

```json
"approach_args": {
    "lamb": 5000,
    "alpha": -1,
    "fi_sampling_type": "true",
    "fi_num_samples": -1
}
```
* `"lamb"`: EWC regularization strength (default=5000)
* `"alpha"`: trade-off for fusing old and new Fisher matrices (`-1` = no fusion) (default=-1)
* `"fi_sampling_type"`: Fisher sampling type (default=`"true"`)
* `"fi_num_samples"`: number of samples for Fisher computation (`-1` = all) (default=-1)

### End-to-End Incremental Learning
```json
"approach": "eeil"
```
[arxiv](https://arxiv.org/abs/1807.09536)
| [ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf)

```json
"approach_args": {
    "lamb": 1.0,
    "T": 2,
    "lr_finetuning_factor": 0.01,
    "nepochs_finetuning": 40,
    "noise_grad": false
},
"exemplars_args": {
    "num_exemplars": 2000,
    "exemplar_selection": "random"
}
```
* `"lamb"`: forgetting-intransigence trade-off (default=1.0)
* `"T"`: softmax temperature (default=2)
* `"lr_finetuning_factor"`: LR multiplier for the balanced finetuning phase (default=0.01)
* `"nepochs_finetuning"`: epochs for balanced finetuning phase (default=40)
* `"noise_grad"`: add noise to gradients (default=false)

### Deep Model Consolidation (Not Ready)
```json
"approach": "dmc"
```
[arxiv](https://arxiv.org/abs/1903.07864)
| [WACV 2020](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Class-incremental_Learning_via_Deep_Model_Consolidation_WACV_2020_paper.pdf)

```json
"approach_args": {
    "aux_data_path": "/path/to/aux_dataset",
    "aux_batch_size": 128
}
```
* `"aux_data_path"`: path to auxiliary dataset (ImageFolder format) used for distillation (required)
* `"aux_batch_size"`: batch size for auxiliary dataset (default=128)

### Bias Correction
```json
"approach": "bic"
```
[arxiv](https://arxiv.org/abs/1905.13260)
| [CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf)

```json
"approach_args": {
    "lamb": -1,
    "T": 2,
    "val_exemplar_percentage": 0.1,
    "num_bias_epochs": 200
},
"exemplars_args": {
    "num_exemplars": 2000,
    "exemplar_selection": "random"
}
```
* `"lamb"`: forgetting-intransigence trade-off (`-1` = moving schedule `known/total`) (default=-1)
* `"T"`: softmax temperature (default=2)
* `"val_exemplar_percentage"`: fraction of exemplars reserved for bias-layer validation (default=0.1)
* `"num_bias_epochs"`: epochs for training the bias correction layer (default=200)

### Learning a Unified Classifier Incrementally via Rebalancing
```json
"approach": "lucir"
```
[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf)

```json
"approach_args": {
    "lamb": 5.0,
    "lamb_mr": 1.0,
    "dist": 0.5,
    "K": 2
},
"exemplars_args": {
    "num_exemplars": 2000,
    "exemplar_selection": "random"
}
```
* `"lamb"`: distillation loss weight (default=5.0)
* `"lamb_mr"`: margin ranking loss weight (default=1.0)
* `"dist"`: margin threshold for the MR loss (default=0.5)
* `"K"`: number of hard negative new-class embeddings for MR loss (default=2)

### Learning to Prompt
```json
"approach": "l2p"
```
[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_to_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf)

Requires a prompt-enabled backbone (`"network": "vit_small_patch16_224_prompt"` or `"vit_base_patch16_224_prompt"`).

```json
"approach_args": {
    "lamb": 0.5
}
```
* `"lamb"`: weight for the prompt-similarity pull loss (default=0.5)

### DualPrompt (Not Ready)

<!-- TODO -->

### LWF Dual
```json
"approach": "lwf_dual"
```
Dual-head LwF: soft KL distillation on both the cls and dist heads against the same teacher cls logits.
Inference uses cls logits only. Requires a distilled backbone (`"distilled": true`).

```json
"approach_args": {
    "lamb": 1.0,
    "T": 2
}
```
* `"lamb"`: KD loss weight (default=1.0)
* `"T"`: softmax temperature (default=2)

### Hydra
```json
"approach": "hydra"
```
Dual-head CIL with role separation: the cls head is the in-task classifier, the dist head acts as a
frozen task identifier. After each task the dist head is frozen and used for task retrieval at inference.
Requires a distilled backbone (`"distilled": true`).

```json
"approach_args": {
    "lamb": 1.0,
    "lamb_cos": 1.0,
    "lamb_dist": 1.0,
    "lamb_dist_kd": 1.0,
    "T": 2
}
```
* `"lamb"`: KD weight on old cls logits (default=1.0)
* `"lamb_cos"`: cosine feature loss weight on dist features (default=1.0)
* `"lamb_dist"`: CE weight on current-task dist head (default=1.0)
* `"lamb_dist_kd"`: KD weight on old dist logits (default=1.0)
* `"T"`: softmax temperature (default=2)
