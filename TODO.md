# TODO — Ouroboros

---

## Framework

### `export TMPDIR=/tmp` σε όλα τα sbatch scripts

Τώρα υπάρχει μόνο στο `lwf_hydra_v2.sh` και `lwf_hydra_v3.sh`. Χωρίς αυτό ο PyTorch DataLoader δημιουργεί `pymp-*` dirs στο CWD (project root).

- [ ] `find slurm/ -name "*.sh" | xargs sed -i 's/conda activate "\$CONDA_ENV"/conda activate "\$CONDA_ENV"\nexport TMPDIR=\/tmp/'`

### `requirements.txt` + `environment.yml`

- [ ] Δημιουργία στη ρίζα του repo — αναφέρονται ήδη στο README με TODO marker

### `network_args` support στον trainer + L2P PILOT re-run

Τώρα: οι παράμετροι `pool_size`, `top_k`, `prompt_len` του L2P είναι hardcoded στο `vit_prompt.py`.

- [ ] Πρόσθεσε `network_args` key στο JSON config → pass ως kwargs στο network factory (ανάλογο με `approach_args`)
- [ ] Re-run L2P με PILOT hyperparameters (`pool_size=10`, `prompt_len=5`, Adam, `lr=0.001875`, `epochs=5`, `batch_size=16`, `λ=0.1`)

Αποτέλεσμα τώρα (non-PILOT config): TAg=83.0% vs paper=83.83%.

### ConvNeXt — πιθανή προσθήκη αρχιτεκτονικής

- [ ] `convnext_small.fb_in1k` (Small, IN-1k) — αντίστοιχο ViT-Small
- [ ] `convnext_base.fb_in22k` (Base, IN-21k) — αντίστοιχο ViT-Base
- Σκοπός: να δούμε αν modernized CNN (χωρίς attention) συμπεριφέρεται διαφορετικά στο CIL

### Refactor hydra network selection στον trainer

Αντί για boolean flags (`hydra_v2: true`, `hydra_v3: true`) + if/elif chain, χρήση `network_class` key στο config + dict mapping:

```python
HYDRA_NETS = {"hydra": LLL_Net_Hydra, "hydra_v2": LLL_Net_Hydra_v2, ...}
net = HYDRA_NETS[args['network_class']](...)
```

- [ ] Αντικατάσταση if/elif στο `cil_trainer.py`
- [ ] Ενημέρωση configs (αφαίρεση flags, προσθήκη `network_class`)

### DMC — Auxiliary dataset

- [ ] Χρειάζεται ImageNet-1k subset στο cluster (ImageFolder format, 100k–200k εικόνες)
- imagenet_32 απορρίφθηκε (.npz format + 32×32 resolution)

### Smoke tests

Το FACILCUSTOM είχε integration smoke tests — δεν μεταφέρθηκαν (δεμένα με παλιό CLI interface).

- [ ] Γράψε smoke tests για Ouroboros με JSON-config interface
- Tiny dataset (CIFAR-10 subset ή synthetic) + μικρό ViT, 1–2 epochs, CPU
- Δοκίμαζε: finetuning, lwf, bic (με exemplars), simplecil, l2p
- Ελέγχει: δεν κρασάρει, TAg/TAw metrics εκτυπώνονται, resume από task N δουλεύει
