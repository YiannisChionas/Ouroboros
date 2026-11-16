---
name: hpc
description: Διαχείριση του HUA HPC cluster (SLURM) για το Ouroboros — κανόνες compliance, υποβολή/αλυσίδες jobs, έλεγχος κατάστασης, sync logs HPC→local, εξαγωγή αποτελεσμάτων, καθαρισμός logs/checkpoints, νέα datasets. Χρησιμοποίησέ το για ΟΠΟΙΑΔΗΠΟΤΕ ενέργεια στο cluster ή στα logs του (ssh hpc, sbatch, squeue, "δες πως πάνε τα jobs", "βάλτο να τρέξει", "τι αποτελέσματα βγάλαμε").
---

# HPC (HUA cluster) — Ouroboros

> **Παραβίαση του οδηγού = process kill χωρίς προειδοποίηση, επανάληψη = BAN** → μήνες πίσω.
> Πριν από **κάθε** ενέργεια στο HPC ελέγχω ότι είναι συμβατή με τους κανόνες §1. Αν έχω αμφιβολία, **ρωτάω τον χρήστη πριν**.
> Επίσημος οδηγός: https://gitlab.hua.gr/hpc/hpc-cluster-guide (clone: `git@gitlab.hua.gr:hpc/hpc-cluster-guide.git`).

Scripts του skill (τρέχουν **μόνο τοπικά**, `S=.claude/skills/hpc/scripts`):

| Script | Τι κάνει |
|---|---|
| `bash $S/status.sh [SINCE]` | running/pending jobs, αποτυχίες από SINCE, φόρτος κόμβων, θέση μας στην ουρά (1 ssh) |
| `bash $S/sync_logs.sh <regex>\|--all` | rsync logs HPC (flat) → local (structured), ξαναφέρνει ελλιπή `.out` |
| `python3 $S/extract_results.py <dirs> [--mean]` | TAw / TAg / TAw Forg / TAg Forg ανά run, + mean±std (seeds) |
| `python3 $S/cleanup_list.py <dirs> [--apply-local]` | περιττά `.out/.err` ανά task + έτοιμη εντολή `rm` για το HPC |

---

## 1. Κανόνες (compliance)

### SSH
`ssh hpc` — key-based auth (`~/.ssh/config`). Καταλήγει σε **login node** (submission node): μόνο για submit jobs, εντολές στο απολύτως ελάχιστο.

### Επιτρέπεται στον login node μόνο
- `git pull --rebase`
- `sbatch`, `squeue`, `sacct`, `scancel`, `scontrol show job`, `scontrol update job` (dependency/partition δικών μας jobs), `sinfo`, `scontrol show partition|node`, `sacctmgr show assoc`
- `mkdir -p` για log dirs
- `ls`, `tail`, `grep`, `head` σε συγκεκριμένα αρχεία (σύντομες, one-shot) — **όχι** `du`/`find` σε μεγάλα dirs (π.χ. datasets)
- `rm` για logs/results με **λίστα που αποφασίστηκε τοπικά** — μόνο λίγα/μεγάλα αρχεία. Φάκελοι με χιλιάδες μικρά αρχεία (datasets, `.cache` του HF download) σβήνονται με **CPU SLURM job** στο obiwan (`sbatch --wrap='rm -rf ...'`), γιατί στο NFS παίρνουν δεκάδες λεπτά.
- `rsync` HPC → local (τρέχει **από το local**)

### Απαγορεύεται στον login node
- `python` (parsing logs, cleanup, ανάλυση, imports torch κλπ)
- interactive sessions, `tail -f`, loops/watch, μακροσκελή `find`/`grep -r`
- `pip install` / αλλαγές στο conda env χωρίς ρητή άδεια του χρήστη

**Κάθε ανάλυση γίνεται τοπικά**: rsync τα logs → Python εδώ → αν χρειάζεται διαγραφή στο HPC, στέλνω μόνο έτοιμη λίστα με `ssh hpc "rm ..."`.
**Εργασίες χωρίς GPU** (downloads, extraction, preprocessing, αναλύσεις): **μόνο ως SLURM job χωρίς `--gres`**, ποτέ στον login node.

### Κώδικας
**ΠΟΤΕ** rsync για αλλαγές στον κώδικα του repo. Πάντα: `git commit` + `git push` τοπικά → `ssh hpc "cd /home/it219111/git/Ouroboros && git pull --rebase"`.

### Όρια και δίκαιη χρήση
- Account `basic`, QoS `normal`: **max 2 jobs ταυτόχρονα ανά χρήστη σε όλα τα partitions μαζί**, max **5h** ανά job. Μακριά runs = αλυσίδες jobs + checkpoint.
- Ζητάω **μόνο ό,τι χρειάζεται**: `--gres=gpu:1`, `--cpus-per-task=4`, `--ntasks=1` πάντα.
- Jobs με >1 GPU (και οι MIG slices μετράνε ως GPU) ή πολλά CPUs: **μόνο μετά από έγκριση admins** (`hpc-admin@hua.gr` ή issue στο repo του οδηγού). Το ότι ένα partition φαίνεται ελεύθερο **δεν** το δικαιολογεί.
- Compute nodes: **CUDA 11.8** — συμβατά PyTorch builds.

### Partitions (επαληθευμένο 2026-11-14 με `sacctmgr`/`scontrol show partition`)

| Partition | Κόμβος | GPUs | CPUs / RAM | Εμείς; | Σημείωση |
|---|---|---|---|---|---|
| **yoda** | hpc02 | 4 ολόκληρες | 112 / 980GB | ✓ | Το μόνο για κανονικό GPU training (DeiT-B κλπ) |
| **obiwan** (default) | hpc01 | 4 × MIG `1g.20gb` | 64 / 183GB | ✓ | Για CPU jobs (downloads) και ελαφριά GPU jobs. Κάθε slice ≈ 1/7 GPU, δεν ενώνονται |
| **leia** | hpc03 | 2 × RTX A5000 | 16 / 30GB | ✓ | DRAIN/INVALID από 2026-09-18 (δεν βλέπει τις GPUs) |
| solo | hpc01–02 | — | — | ✗ | Μόνο account `cpu-only` |
| chewbacca | hpc04–05 | 7 | 64 | ✗ | Μόνο με έγκριση admins |
| anakin(_low) / luke(_low) | hpc06 / hpc07 | 8×H200 / 4×H100 | 96 | ✗ | Μόνο `god-account` — ποτέ submit |

### Storage
- `/home` (NFS, **quota 200G από 2026-11-16**) — code, conda envs, logs. Το `quota`/`df` δεν δείχνουν το όριο· ο server μετρούσε >200G ενώ το `du` έβγαζε 115G. Σύμπτωμα υπέρβασης: `Disk quota exceeded` / `could not close .git/index.lock`, και jobs που αποτυγχάνουν σε δευτερόλεπτα με **άδεια** `.out`/`.err` (exit 1:0). Τότε jobs μέτρησης/καθαρισμού γράφουν output στο `/opt/storage/users/${USER}`.
- `/opt/storage/datasets`, `/opt/storage/users/${USER}` — NFS, **αργά**.
- `/var/local/storage/${USER}` — **γρήγορος τοπικός δίσκος του compute node** (quota 500G). Τα job scripts **αντιγράφουν** τα δεδομένα εκεί αν δεν υπάρχουν (με `flock`, βλ. §5) και διαβάζουν από εκεί. Διαγραφή όταν τελειώσουν τα πειράματα.
- Κρατάμε μόνο ό,τι χρειάζεται. **Τα αρχεία μπορεί να διαγραφούν από τους admins χωρίς προειδοποίηση** → logs γίνονται rsync τοπικά, checkpoints μακριών runs (pretraining) αντιγράφονται τοπικά περιοδικά.
- **ΠΟΤΕ** μη σβήσεις `results/pretraining/vit_hydra_full/vit_hydra_final.pth`.

---

## 2. Δομή HPC vs Local

| Τι | Πού | Μορφή |
|---|---|---|
| HPC logs (μόνιμη πηγή αλήθειας, τα `.out` κρατιούνται πάντα) | `/home/it219111/experiments/Ouroboros/logs/` | flat: `{approach}_{dataset}_{backbone}/` |
| Local logs | `Ouroboros/logs/` | `{backbone}/{dataset}/{approach}/` · χωρίς dataset στο όνομα → `logs/analysis/` ή `logs/_misc/` |
| HPC results (προσωρινά: checkpoints, exemplars, per-task .txt) | `/home/it219111/git/Ouroboros/results/{network}/{dataset}/{approach}/10/` ή `experiment_path` του config | — |
| Datasets στο HPC | `/home/it219111/git/data/{cifar100,food101,inat200,inatm200,inat500,...}` | — |

- Dataset anchors για το parsing του flat ονόματος: `cifar100`, `food101`, `inatm200`, `inat500`, `inat200`, `five_datasets`.
  - `bic_cifar100_resnet50_in1k` → backbone=`resnet50_in1k`, dataset=`cifar100`, approach=`bic`
  - `lwf_dt_cos_v2_cifar100_deit_small_in1k` → backbone=`deit_small_in1k`, dataset=`cifar100`, approach=`lwf_dt_cos_v2`
- Checkpoints: το `disk_logger.save_model` κρατά **μόνο το τελευταίο** `task{t}.ckpt` (atomic save, σβήνει τα παλιότερα). Το resume φορτώνει `task{start_task-1}.ckpt`.
- Αν δεν υπάρχουν results/ για ένα approach: κοίτα πρώτα τα HPC logs — μάλλον έτρεξε και σβήστηκαν τα results.
- **Κρίσιμο**: πάντα κοιτάω απευθείας στο HPC πριν πω τι λείπει — όχι μόνο στα local logs.

---

## 3. Υποβολή jobs

> **Το `sbatch` τρέχει ΠΑΝΤΑ από το project root** `/home/it219111/git/Ouroboros` (relative paths `../data`, `./results`). Αλλιώς `FileNotFoundError`.
> **ΠΑΝΤΑ** πριν το πρώτο submit νέου job name: `mkdir -p /home/it219111/experiments/Ouroboros/logs/<job_name>/` — αλλιώς το SLURM δεν γράφει `.out` και το job αποτυγχάνει αμέσως.

### Ροή για νέο run
1. Τοπικά: approach (νέο versioned αρχείο, όχι flags σε υπάρχον), `configs/<backbone>/<dataset>/<name>.json`, `slurm/<backbone>/<dataset>/<name>.sh` (αντιγραφή από υπάρχον script, `--job-name=<name>_<dataset>_<backbone>`).
2. `git add` συγκεκριμένα αρχεία → `commit` → `push`.
3. Έλεγχος slots: `bash $S/status.sh` (ή `squeue -u it219111`). **Αν υπάρχει ελεύθερο slot, το χρησιμοποιώ** — κρίμα να χάνεται χρόνος.
4. `ssh hpc "cd /home/it219111/git/Ouroboros && git pull --rebase && mkdir -p <logdir> && bash slurm/general.sh <backbone>/<dataset>/<name>.sh <NUM_TASKS> 0"`
5. Αναφέρω στον χρήστη first/last job id κάθε αλυσίδας.

### Αλυσίδες
- `bash slurm/general.sh <script> [NUM_TASKS] [START_FROM]` — 1 job ανά task, `afterok` μεταξύ τους. Resume: `START_FROM=k`.
- `bash slurm/general_serial.sh <script> [NUM_TASKS] [START_FROM] [INITIAL_DEP]` — το 1ο job με `afterany:INITIAL_DEP`. **Μόνο όταν τρέχουν ήδη 2 approaches** και θέλω 3ο σε αναμονή (INITIAL_DEP = last job μιας lane). Τυπώνει `first_job=` / `last_job=`.
- Μεμονωμένο job: `sbatch --export=ALL,START_TASK=0,STOP_TASK=1 slurm/<...>.sh`.
- `afterok` vs `afterany`: με `afterok`, αν αποτύχει ή ακυρωθεί το προηγούμενο, το επόμενο μένει `DependencyNeverSatisfied` για πάντα. Για jobs άλλων projects πίσω από τις αλυσίδες μας → `afterany`.
- Αλλαγή εξάρτησης/partition pending job: `scontrol update jobid=<id> dependency=afterok:<id2>` / `dependency=''` / `partition=obiwan`.
- **Πριν ακυρώσω αλυσίδα** ελέγχω αν κάποιο job (π.χ. άλλου project, `hest_*`) εξαρτάται από αυτήν: `squeue -u it219111 -h -o '%i %j %E' | grep <last_job>`. Αν ναι, το ξαναδένω πριν το `scancel`, αλλιώς με `afterany` ξεκινά αμέσως.
- Προτεραιότητα: όσο περιμένει ένα job ανεβαίνει (age). Ακύρωση + επανυποβολή = μηδενισμός.

### Joint (`slurm/general_joint.sh`) — epoch-splitting ανά task
```bash
bash slurm/general_joint.sh <job_script> <num_tasks> <nepochs> [SPLIT1] [SPLIT2]
```
| Tasks | Jobs/task | Epochs/job |
|---|---|---|
| 0 … SPLIT1-1 | 1 | nepochs |
| SPLIT1 … SPLIT2-1 | 2 | nepochs/2 |
| SPLIT2 … end | 4 | nepochs/4 |

Defaults SPLIT1=5, SPLIT2=8 (10-task datasets). inat200 (20 tasks): `... 20 90 10 15`. Τα partial-epoch jobs δεν καλούν `save_model`.

---

## 4. Έλεγχος, sync, αποτελέσματα, καθαρισμός

### "Δες πως πάνε τα jobs"
1. `bash $S/status.sh` → running, αποτυχίες, φόρτος, θέση στην ουρά.
2. Αν κάτι απέτυχε: `ssh hpc "tail -n 40 <HPC_LOGS>/<job_name>/<job_name>-<id>.err"` (one-shot, ΟΧΙ `tail -f`).
3. Για ολοκληρωμένα / σε εξέλιξη runs: `bash $S/sync_logs.sh '<regex>'` → `python3 $S/extract_results.py logs/<bb>/<ds>/<appr> ...`.
4. Αν περιμένουν με `Priority`: ποιος πιάνει τον κόμβο → `ssh hpc "squeue -p yoda -t R -o '%.8i %.10u %.4C %.12b %.10M'"`.
5. Αναφορά: αριθμοί/πίνακες, χωρίς συμπεράσματα αν δεν ζητηθούν.

### Αποτελέσματα από τα .out
Κάθε `.out` έχει στο τέλος 4 sections: `TAw Acc`, `TAg Acc`, `TAw Forg`, `TAg Forg` — πίνακας μέχρι το `STOP_TASK` του job, οι γραμμές των tasks που δεν έχουν εκπαιδευτεί είναι 0. Το τελικό αποτέλεσμα = τελευταίο `Avg.:` κάθε section στο `.out` με το μεγαλύτερο `STOP_TASK` (το `extract_results.py` το κάνει αυτό και σημειώνει `!` αν το run δεν έχει τελειώσει). Τα seeds αλλάζουν και τη σειρά των κλάσεων (`np.random.shuffle(class_order)`).

### Καθαρισμός logs (μετά από ολοκλήρωση run)
- Κανόνας: για κάθε task κρατάμε μόνο το πιο πρόσφατο (υψηλότερο job id) `.out` με `TAg Acc` + το `.err` του. Τα υπόλοιπα είναι αποτυχημένα/ακυρωμένα jobs.
- `python3 $S/cleanup_list.py logs/<bb>/<ds>/<appr> [--apply-local]` → λίστα + έτοιμο `ssh hpc "cd ... && rm ..."` που το στέλνω μετά από έλεγχο.
- **Ποτέ** σε runs που ακόμα τρέχουν ή έχουν pending jobs. **Ποτέ** στο `joint` (πολλά `.out` ανά task είναι σωστά).

### Καθαρισμός results/checkpoints
Μετά από ολοκλήρωση run και αφού διαβαστούν τα `.out`: `rm -rf ~/git/Ouroboros/results/<network>/<dataset>/<approach>` — εκτός αν χρειάζονται checkpoints για ανάλυση (drift, CKA κλπ): ρωτάω πρώτα.

---

## 5. Δεδομένα / νέο dataset

- Download/extraction = **CPU SLURM job στο obiwan** (χωρίς `--gres`, 4 CPUs, 8G), resumable. Παράδειγμα: `slurm/data/download_inatm200.sh` (`hf download ... --local-dir /home/it219111/git/data/<name>`, στο τέλος ελέγχει ότι υπάρχουν όλα τα αρχεία των `train.txt`/`test.txt`).
- Το HF download κατεβάζει ~7 αρχεία/s (100k εικόνες ≈ 4h). Αν μπορεί να ξεπεράσει τις 5h: δεύτερο download job με `afterany` στο πρώτο, και τα training chains `afterok` στο **δεύτερο**.
- Format που διαβάζει ο loader (generic branch του `get_dataset`): `train.txt` / `test.txt` με `<relative_path> <class_id>` και `classes.txt`, relative στο `data_path`. Το όνομα του dataset δεν έχει ειδική μεταχείριση στον κώδικα — χρησιμοποιείται στα results paths και τα job names.
- Τα training scripts αντιγράφουν τα δεδομένα στον τοπικό δίσκο του κόμβου μία φορά (το config έχει `"data_path": "/var/local/storage/it219111/<name>"`):
  ```bash
  SRC_DATA="${BASE_DIR}/git/data/<name>"
  LOCAL_DATA="/var/local/storage/${USER}/<name>"
  mkdir -p "/var/local/storage/${USER}"
  flock "/var/local/storage/${USER}/<name>.lock" bash -c "
    if [ ! -f '${LOCAL_DATA}/.complete' ]; then
      mkdir -p '${LOCAL_DATA}' && rsync -a '${SRC_DATA}/' '${LOCAL_DATA}/' && touch '${LOCAL_DATA}/.complete'
    fi"
  ```
  Παράδειγμα: `slurm/deit_base_in1k/inatm200/lwf_dual_v9.sh`.
- Datasets στο HPC: `inat200` (παλιό, πτυχιακή) και `inatm200` (iNatM-200, https://huggingface.co/datasets/YiannisChionas/iNatM-200 — ίδιες 200 κλάσεις/σειρά με το inat200, 74.578/100.000 κοινές εικόνες, 400/100 ανά κλάση). `inat500` (iNat-500, https://huggingface.co/datasets/YiannisChionas/iNat-500 — 500 κλάσεις, 400/100 ανά κλάση, 250.000 εικόνες → 50 tasks των 10, download ≈ 10h = 3 jobs σε σειρά).
