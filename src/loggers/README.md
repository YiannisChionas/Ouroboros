# Loggers

Loggers are configured via the `"log"` key in the JSON config:
```json
"log": ["disk"]
```
or, to enable both:
```json
"log": ["disk", "tensorboard"]
```

The output directory is controlled by `"experiment_path"` (optional). If not set, it is automatically
derived as `{network}/{dataset}/{approach}/{increment}`. Results are written under `"results_path"` (default `"./results"`).

## Disk logger
The disk logger outputs the following file and folder structure:
- **figures/**: folder where generated figures are logged.
- **models/**: folder where model weight checkpoints are saved.
- **results/**: folder containing the results.
  - **acc_tag**: task-agnostic accuracy table.
  - **acc_taw**: task-aware accuracy table.
  - **avg_acc_tag**: task-agnostic average accuracies.
  - **avg_acc_taw**: task-aware average accuracies.
  - **forg_tag**: task-agnostic forgetting table.
  - **forg_taw**: task-aware forgetting table.
  - **wavg_acc_tag**: task-agnostic weighted average accuracies.
  - **wavg_acc_taw**: task-aware weighted average accuracies.
- **raw_log**: JSON file containing all logged metrics (easily read with e.g. `pandas`).
- stdout: a copy of the standard output.
- stderr: a copy of the error output.

## TensorBoard logger
The TensorBoard logger outputs analogous metrics to the disk logger separated into different tabs
according to the task and different graphs according to the data splits.

Screenshot for a 10-task experiment, showing the last task plots:
<p align="center">
<img src="/docs/_static/tb2.png" alt="Tensorboard Screenshot" width="920"/>
</p>
