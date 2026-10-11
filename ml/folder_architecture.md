# PoisonTrace — Folder Architecture

## Repository overview

The current repository root contains `ml/` and `docs/`; `docs/ml_notes/` is empty. The implementation is almost entirely under `ml/`. It is a Python/PyTorch Fashion-MNIST experiment for creating contributor data shards, injecting synthetic poisoning scenarios, training a CNN, measuring clean accuracy and a known-trigger attack-success rate (ASR), triaging contributors, and performing leave-one-out/group attribution.

This document describes files and relationships that currently exist. Terms such as **legacy** and **engine** are evidence-based descriptions of duplicate code paths, not a claim that one is necessarily executed by an external application.

## Complete folder tree

The tree below lists the meaningful current repository contents. `.venv/`, `__pycache__/`, and `.pytest_cache/` are generated environment/cache areas; their many internal files are intentionally collapsed. `ml/.pytest_cache` could not be enumerated by the filesystem sandbox, so its internal contents are uncertain.

```text
poisontrace/
├── docs/
│   └── ml_notes/                                  [empty]
└── ml/
    ├── .gitignore                                 [empty]
    ├── requirements.txt
    ├── ML_MODEL_AUDIT_REPORT.md
    ├── folder_architecture.md
    ├── .venv/                                     [generated virtual environment]
    ├── .pytest_cache/                              [generated pytest cache; inaccessible]
    ├── data/
    │   ├── poisoned/
    │   │   ├── C2_collusion.npz
    │   │   ├── C3_collusion.npz
    │   │   ├── C3_poisoned.npz
    │   │   ├── C5_noisy.npz
    │   │   ├── collusion_metadata.json
    │   │   └── scenario_metadata.json
    │   ├── processed/
    │   │   ├── clean/                             [empty]
    │   │   ├── metadata/                          [empty]
    │   │   ├── poisoned/                          [empty]
    │   │   ├── test/                              [empty]
    │   │   ├── validation/                        [empty]
    │   │   └── train/contributors/
    │   │       ├── C1.npz
    │   │       ├── C2.npz
    │   │       ├── C3.npz
    │   │       ├── C4.npz
    │   │       └── C5.npz
    │   └── raw/
    │       ├── downloaded/                        [empty]
    │       ├── original/                          [empty]
    │       └── FashionMNIST/raw/
    │           ├── train-images-idx3-ubyte
    │           ├── train-images-idx3-ubyte.gz
    │           ├── train-labels-idx1-ubyte
    │           ├── train-labels-idx1-ubyte.gz
    │           ├── t10k-images-idx3-ubyte
    │           ├── t10k-images-idx3-ubyte.gz
    │           ├── t10k-labels-idx1-ubyte
    │           └── t10k-labels-idx1-ubyte.gz
    ├── experiments/
    │   ├── configs/j2_multiseed.yaml
    │   ├── logs/                                  [empty]
    │   └── results/
    │       ├── final_verdicts.json
    │       ├── forensic_report.md
    │       ├── leave_one_out_results.json
    │       ├── multi_seed_leave_one_out.json
    │       └── engine/
    │           ├── engine_test.json
    │           ├── final_forensic_report.md
    │           ├── forensic_report.json
    │           ├── group_attribution_engine.json
    │           ├── leave_one_out_engine.json
    │           ├── multi_seed_analysis.json
    │           ├── multi_seed_engine.json
    │           ├── multi_seed_forensic_verdict.json
    │           └── multi_seed_group_attribution.json
    ├── models/
    │   ├── checkpoints/                           [empty]
    │   ├── final/                                 [empty]
    │   └── scenario_all_contributors.pt
    ├── notebooks/01_data_exploration.ipynb
    ├── src/
    │   ├── attacks/
    │   │   ├── backdoor.py
    │   │   ├── create_collusion_scenario.py
    │   │   ├── create_scenario.py
    │   │   └── trigger.py
    │   ├── attribution/
    │   │   ├── analyze_results.py
    │   │   ├── collusion.py
    │   │   ├── collusion_individual.py
    │   │   ├── collusion_verdict.py
    │   │   ├── prove.py
    │   │   ├── triage.py
    │   │   └── verdict.py
    │   ├── data/
    │   │   ├── create_shards.py
    │   │   ├── dataset.py
    │   │   ├── download_dataset.py
    │   │   ├── preprocessing.py
    │   │   ├── splitting.py
    │   │   └── verify_shards.py
    │   ├── engine/
    │   │   ├── __init__.py
    │   │   ├── analyze_multiseed.py
    │   │   ├── multiseed.py
    │   │   ├── pipeline.py
    │   │   ├── report.py
    │   │   ├── results.py
    │   │   ├── attacks/__init__.py, backdoor.py
    │   │   ├── probes/__init__.py, metrics.py
    │   │   ├── prove/__init__.py, group_multiseed_verdict.py,
    │   │   │   group_test.py, leave_one_out.py, multi_seed.py,
    │   │   │   multi_seed_group_test.py
    │   │   ├── trainer/__init__.py, trainer.py
    │   │   ├── triage/__init__.py, triage.py
    │   │   └── verdict/__init__.py, analyze_multiseed.py,
    │   │       forensic_report.py, group_verdict.py,
    │   │       multiseed_verdict.py, verdict.py
    │   ├── models/baseline.py
    │   ├── training/evaluate.py, train.py
    │   └── utils/config.py, metrics.py
    └── tests/
        ├── conftest.py
        ├── test_attacks.py
        ├── test_detector.py
        ├── test_environment.py
        ├── test_statistical_analysis.py
        └── test_triage.py
```

## Folder-by-folder explanation

| Folder | Current responsibility |
|---|---|
| `docs/ml_notes/` | Present but empty; no project documentation currently resides here. |
| `ml/data/raw/` | Local Fashion-MNIST source data. The actual downloaded IDX files are under `FashionMNIST/raw/`; `downloaded/` and `original/` are empty. |
| `ml/data/processed/` | Intended processed-data hierarchy. Only `train/contributors/` is populated; all other visible subdirectories are empty. |
| `ml/data/poisoned/` | Generated S1/S2 and collusion scenario arrays plus compact metadata. |
| `ml/models/` | Model artifact area. One baseline checkpoint exists; named checkpoint/final subdirectories are empty. |
| `ml/notebooks/` | Exploratory Jupyter material; one data-exploration notebook. |
| `ml/experiments/configs/` | Experiment configuration artifact; the one YAML is not loaded by current Python source. |
| `ml/experiments/logs/` | Intended logs location; empty. |
| `ml/experiments/results/` | Saved legacy/non-engine experiment metrics and human-readable report. |
| `ml/experiments/results/engine/` | Saved outputs from the engine-oriented serializers, leave-one-out, multi-seed analysis, report generation, and group attribution. |
| `ml/src/data/` | Downloading, sharding, storage access, and standalone helper routines. |
| `ml/src/attacks/` | Canonical scenario/trigger implementation used by the training and evaluation paths. |
| `ml/src/models/` | CNN definition. |
| `ml/src/training/` | Standalone training/evaluation implementation and dataset selection logic. |
| `ml/src/attribution/` | Earlier direct scripts for triage, causal analysis, collusion, and verdict logic. |
| `ml/src/engine/` | A more structured wrapper layer: reusable training, probes, triage, proof engines, result persistence, verdict engines, and reports. |
| `ml/src/utils/` | ML root paths, seed constant, and generic sklearn classification metrics helper. |
| `ml/tests/` | Pytest suite. |

## File-by-file explanation

### Top-level ML files

| File | Actual content/purpose |
|---|---|
| `.gitignore` | Empty file; no ignore rules are defined. |
| `requirements.txt` | Unpinned dependency names: NumPy, pandas, scikit-learn, plotting/notebook tools, PyYAML, Torch/Torchvision, and pytest. |
| `ML_MODEL_AUDIT_REPORT.md` | Existing detailed audit of the ML system, its artifacts, results, and limitations. |
| `folder_architecture.md` | This architecture document. |

### Source: data, attacks, model, and training

| File | Actual behavior |
|---|---|
| `src/utils/config.py` | Resolves `ml/` from its own location and defines data/model/results paths plus `RANDOM_SEED=42`. |
| `src/utils/metrics.py` | Provides weighted sklearn accuracy, precision, recall, and F1 helper; it is not used by the training/probe flow. |
| `src/data/download_dataset.py` | Downloads torchvision Fashion-MNIST train/test data to `data/raw/` if run. |
| `src/data/create_shards.py` | Selects 2,000 examples per Fashion-MNIST class (20,000 total), shuffles with seed 42, and writes five 4,000-example contributor NPZ shards. |
| `src/data/dataset.py` | `ContributorDataset` loads clean, S1/S2 poisoned, or S3 collusion NPZ data, scales images to 0–1, adds channel dimension, and returns Torch tensors. |
| `src/data/verify_shards.py` | Prints shard shapes and class counts. It does not calculate overlap despite its name. |
| `src/data/preprocessing.py` | Standalone array normalization helper; not called by the active dataset loader. |
| `src/data/splitting.py` | Standalone stratified sklearn train/test split helper; not called by the experiment path. |
| `src/attacks/backdoor.py` | Canonical functional backdoor: selects non-target labels, overwrites a lower-right 3×3 patch with 255, and relabels selected samples to target class 0. |
| `src/attacks/trigger.py` | Torch inference-time counterpart that puts a value-1.0 lower-right 3×3 patch into normalized NCHW images. |
| `src/attacks/create_scenario.py` | Generates S1 C3 backdoor artifact and S2 C5 label-noise/brightness decoy artifact, then writes scenario metadata. |
| `src/attacks/create_collusion_scenario.py` | Generates S3 artifacts by applying a complete C2 and C3 backdoor independently, then writes collusion metadata. |
| `src/models/baseline.py` | Defines `BaselineCNN`: three convolutions (1→32→64→128), two 2×2 pools, then 6,272→256→10 classifier with ReLU and 0.2 dropout. `extract_features()` returns the 256-D pre-final representation. |
| `src/training/train.py` | Standalone deterministic CPU train loop: Adam, cross-entropy, default 8 epochs/128 batch/0.001 LR. Builds a concatenated contributor dataset and its main block saves the one checkpoint. |
| `src/training/evaluate.py` | Uses official Fashion-MNIST test data for clean accuracy and targeted ASR over non-class-0 examples after trigger application. |

### Source: legacy/direct attribution scripts

| File | Actual behavior |
|---|---|
| `src/attribution/prove.py` | Directly runs all-contributor and leave-one-out models for seeds 42, 123, 2024 and writes `results/multi_seed_leave_one_out.json`. |
| `src/attribution/analyze_results.py` | Reads that legacy multi-seed JSON, calculates mean/sample-SD and a custom pooled SD, prints verdict-style analysis; it does not save output. |
| `src/attribution/triage.py` | Earlier checkpoint-driven loss and KMeans activation clustering prints. Unlike engine triage, it has no trigger-response signal or combined score persistence. |
| `src/attribution/verdict.py` | Small functional individual verdict helper: 50% ASR causal threshold, 10% cleared threshold, `2×pooled_std` condition. |
| `src/attribution/collusion.py` | Direct script for full collusion baseline and removal of a selected group; prints metrics only. |
| `src/attribution/collusion_individual.py` | Direct script that removes C2 then C3 singly in collusion scenario and prints metrics; it does not save them. |
| `src/attribution/collusion_verdict.py` | Functional group verdict helper using only 50%/10% ASR thresholds. |

### Source: engine layer

| File | Actual behavior |
|---|---|
| `src/engine/__init__.py` | Empty package marker. |
| `src/engine/attacks/__init__.py` | Empty package marker. |
| `src/engine/attacks/backdoor.py` | `BackdoorAttack` class duplicate of functional `src/attacks/backdoor.py`; no current import uses it. |
| `src/engine/trainer/__init__.py` | Empty package marker. |
| `src/engine/trainer/trainer.py` | `EngineTrainer`, a near duplicate of standalone training that delegates dataset construction to `build_training_dataset`. |
| `src/engine/probes/__init__.py` | Empty package marker. |
| `src/engine/probes/metrics.py` | Duplicate clean-accuracy/ASR probe implementation used by engine proof code. |
| `src/engine/triage/__init__.py` | Empty package marker. |
| `src/engine/triage/triage.py` | `TriageEngine`: per-contributor loss, KMeans minority fraction, known-trigger probability shift, ranks, normalized 50/50 loss+cluster suspicion score, and confidence. Trigger score is deliberately excluded from the final score. |
| `src/engine/prove/__init__.py` | Re-exports `LeaveOneOutEngine` and `GroupAttributionEngine`. |
| `src/engine/prove/leave_one_out.py` | Trains full and one-contributor-removed models, calculates ΔASR/clean drop, applies single-seed verdicts, and can write `leave_one_out_engine.json`. |
| `src/engine/prove/multi_seed.py` | Object-oriented multi-seed leave-one-out runner/aggregator; returns results but does not persist them in its main block. |
| `src/engine/prove/group_test.py` | Group baseline/removal engine. `test_group()` writes `group_attribution_engine.json`. |
| `src/engine/prove/multi_seed_group_test.py` | Runs C2+C3 group tests for three fixed seeds, aggregates with `statistics.stdev`, and writes `multi_seed_group_attribution.json`. |
| `src/engine/prove/group_multiseed_verdict.py` | Defines a multi-seed group verdict class but no source imports/calls it. |
| `src/engine/verdict/__init__.py` | Re-exports individual and group verdict engines. |
| `src/engine/verdict/verdict.py` | Single-run individual classification using ΔASR, causal/cleared fractions, and `2×pooled_std`; no clean-accuracy guard. |
| `src/engine/verdict/multiseed_verdict.py` | Multi-seed individual classification with 50% and 10% fractions, `2×std(ΔASR)`, and mean clean-drop ≤0.02 guard. |
| `src/engine/verdict/group_verdict.py` | Single-run group classification using only 50% and 10% ASR fractions. |
| `src/engine/verdict/analyze_multiseed.py` | Reads `multi_seed_engine.json`, calculates population means/SDs, applies `MultiSeedVerdictEngine`, and writes `multi_seed_forensic_verdict.json`. |
| `src/engine/verdict/forensic_report.py` | Reads multi-seed final verdict JSON and creates `final_forensic_report.md`. |
| `src/engine/results.py` | JSON persistence helper. Wraps engine data in `created_at` and `data` and writes under `results/engine/`. |
| `src/engine/report.py` | Combines the saved *single-seed* leave-one-out and group JSON into `forensic_report.json`. |
| `src/engine/pipeline.py` | Calls `build_forensic_report()` and prints its individual and group results; it does not run training or proof itself. |
| `src/engine/multiseed.py` | Procedural three-seed leave-one-out runner. Applies single-seed verdicts and writes `multi_seed_engine.json`. |
| `src/engine/analyze_multiseed.py` | Older engine analysis script reading `multi_seed_engine.json` and writing `multi_seed_analysis.json`; it independently reconstructs the multi-seed rule. |

### Data, checkpoints, experiment artifacts, and notebook

| File/group | Actual content/purpose |
|---|---|
| `data/raw/FashionMNIST/raw/*` | Local Fashion-MNIST raw train/test images and labels in both IDX and gzip form. |
| `data/processed/train/contributors/C1.npz` … `C5.npz` | Generated clean contributor shards; each contains `images` and `labels`. |
| `data/poisoned/C3_poisoned.npz` | S1 replacement C3 shard with backdoor modifications. |
| `data/poisoned/C5_noisy.npz` | S2 replacement C5 shard with 10% relabel/brightness modifications. |
| `data/poisoned/C2_collusion.npz`, `C3_collusion.npz` | S3 replacement collusion shards, each independently backdoored. |
| `data/poisoned/scenario_metadata.json` | S1/S2 declared poison/noise parameters and counts. |
| `data/poisoned/collusion_metadata.json` | S3 declared participant IDs and backdoor parameters. |
| `models/scenario_all_contributors.pt` | Saved Torch `state_dict` from standalone `train.py`; it has no embedded scenario/config metadata. |
| `notebooks/01_data_exploration.ipynb` | Two executed code cells that load/configure Fashion-MNIST exploration. It does not implement the pipeline or generate saved experiment outputs. |
| `experiments/configs/j2_multiseed.yaml` | Declares seeds, contributors, CPU training hyperparameters, attack/decoy settings, and attribution thresholds. Current source does not read it. |
| `results/multi_seed_leave_one_out.json` | Legacy raw three-seed S1/S2 all/LOO metrics from `attribution/prove.py`. |
| `results/leave_one_out_results.json` | Compact seed-42 all/LOO metric derivative. |
| `results/final_verdicts.json` | Rounded legacy S1/S2 verdict summary. |
| `results/forensic_report.md` | Human-readable legacy S1/S2 report. |
| `results/engine/engine_test.json` | Result-serialization smoke artifact, not an ML experiment result. |
| `results/engine/leave_one_out_engine.json` | Engine seed-42 LOO run, including single-seed placeholder pooled SD/verdict fields. |
| `results/engine/multi_seed_engine.json` | Engine raw three-seed LOO records, generated by `engine/multiseed.py`. |
| `results/engine/multi_seed_analysis.json` | Derived multi-seed statistical analysis from the engine raw result. |
| `results/engine/multi_seed_forensic_verdict.json` | Final structured individual multi-seed verdicts. |
| `results/engine/final_forensic_report.md` | Markdown rendering of those final individual verdicts. |
| `results/engine/group_attribution_engine.json` | Seed-42 C2+C3 group-removal record. |
| `results/engine/multi_seed_group_attribution.json` | Three-seed C2+C3 group-removal aggregate and per-seed metrics. |
| `results/engine/forensic_report.json` | Combined report that joins S1/S2 seed-42 individual result with S3 seed-42 group result. |

## Architecture and data flow

Only source-supported relationships are shown:

```text
FashionMNIST raw data
  -> src/data/create_shards.py
  -> processed/train/contributors/C1..C5.npz
  -> create_scenario.py: C3_poisoned + C5_noisy (S1/S2)
     OR create_collusion_scenario.py: C2_collusion + C3_collusion (S3)
  -> ContributorDataset / build_training_dataset
  -> train_model or EngineTrainer -> BaselineCNN
  -> evaluate.py or engine/probes/metrics.py
       -> clean accuracy + known-trigger ASR
  -> optional TriageEngine (loss, clustering, trigger response; no saved triage run)
  -> leave-one-out / group retraining
  -> individual/group verdict engines
  -> JSON and Markdown result/report artifacts
```

For any non-`"collusion"` scenario, `build_training_dataset()` loads C3 and C5 using `scenario="poisoned"`, and clean files for other contributors. Thus its default `scenario="original"` actually means the S1+S2 mixture, not all-clean input. For `scenario="collusion"`, only C2/C3 use collusion artifacts.

## Entry points and commands

The modules below have `if __name__ == "__main__"` blocks. From `ml/`, the source imports expect `src` to be importable; use `python -m ...` where possible.

```powershell
# Data and scenario generation
python -m src.data.download_dataset
python -m src.data.create_shards
python -m src.data.verify_shards
python -m src.attacks.create_scenario
python -m src.attacks.create_collusion_scenario

# Standalone model path
python -m src.training.train

# Legacy/direct attribution path
python -m src.attribution.prove
python -m src.attribution.analyze_results
python -m src.attribution.triage
python -m src.attribution.collusion
python -m src.attribution.collusion_individual
python -m src.attribution.collusion_verdict

# Engine path
python -m src.engine.trainer.trainer
python -m src.engine.prove.leave_one_out
python -m src.engine.multiseed
python -m src.engine.analyze_multiseed
python -m src.engine.verdict.analyze_multiseed
python -m src.engine.verdict.forensic_report
python -m src.engine.prove.group_test
python -m src.engine.prove.multi_seed_group_test
python -m src.engine.pipeline

# Tests
python -m pytest -q
```

Some commands are order-dependent: reporting modules expect their input JSON to exist; `pipeline.py` expects the two single-seed engine result files; training/evaluation requires the local Fashion-MNIST files. `src/engine/prove/multi_seed.py` has a main block but only prints/returns a result, so it does not itself create a result file. `j2_multiseed.yaml` is configuration by content only: no inspected code loads it.

## Experiments/results structure

There are two parallel S1/S2 result families. The legacy direct path writes `results/multi_seed_leave_one_out.json`; the engine path writes `results/engine/multi_seed_engine.json` then generates analysis/verdict/report derivatives. Their saved core values agree for seeds 42, 123, and 2024. The engine area additionally contains S3 group outputs.

`multi_seed_group_attribution.json` confirms a saved three-seed C2+C3 group experiment now exists: mean ASR all `0.994037`, mean ASR without group `0.014148`, and mean ΔASR `0.979889`. It is a data artifact from `multi_seed_group_test.py`; `MultiSeedGroupVerdictEngine` is not wired to consume it, so no matching saved multi-seed group verdict report exists.

## Tests

| Test file | What it covers |
|---|---|
| `tests/conftest.py` | Adds `ml/` to `sys.path` for `src.*` imports. |
| `tests/test_attacks.py` | Backdoor count, target-label exclusion, copied input, patch application, and deterministic fixed-seed output. |
| `tests/test_detector.py` | Two fixture-based `VerdictEngine` classifications for C5 cleared and C3 implicated. |
| `tests/test_environment.py` | Checks that Torch, NumPy, pandas, and sklearn expose version strings. |
| `tests/test_statistical_analysis.py` | Local duplicate classification function and fixtures for C3, C5, and a failing clean-accuracy guard. |
| `tests/test_triage.py` | Triage output/fields/determinism using a dummy model and actual local shards. |

No obvious tests directly exercise a full training run, official-test ASR calculation, shard generation/overlap, S2 decoy construction, collusion scenario construction, group verdicts, YAML usage, persistence/report generation, or checkpoint provenance.

## Legacy and duplicate components

* `src/training/train.py` and `src/engine/trainer/trainer.py` implement materially the same CPU training loop.
* `src/training/evaluate.py` and `src/engine/probes/metrics.py` implement the same clean-accuracy and targeted ASR probes.
* `src/attacks/backdoor.py::add_backdoor` and `src/engine/attacks/backdoor.py::BackdoorAttack.apply` implement the same NumPy attack; the class is not imported by other inspected source.
* `src/attribution/` has direct scripts and functional helpers; `src/engine/` adds classes, result wrapping, and report generation. The code does not explicitly deprecate the former, so "legacy" remains an inference from overlap and artifact paths.
* `src/engine/analyze_multiseed.py` and `src/engine/verdict/analyze_multiseed.py` both aggregate `multi_seed_engine.json`; they create different derivative JSON formats. The latter applies the dedicated `MultiSeedVerdictEngine` and writes the final individual verdict artifact.
* `src/attribution/verdict.py`, `src/engine/verdict/verdict.py`, and test-local `classify()` functions duplicate individual threshold logic. `src/attribution/collusion_verdict.py` and `src/engine/verdict/group_verdict.py` duplicate group threshold logic.

## Architectural observations

* **Scenario naming is inconsistent:** `scenario="original"` routes to poisoned C3/C5 data, while `ContributorDataset` only recognizes `clean`, `poisoned`, and `collusion` directly.
* **Configuration is disconnected:** the YAML describes an experiment but all inspected experiments use Python constants/defaults instead of parsing it.
* **Generated/storage layout is partly unused:** `processed/clean`, `processed/metadata`, `processed/poisoned`, `processed/test`, `processed/validation`, raw `downloaded`/`original`, experiment `logs`, and model `checkpoints`/`final` are empty; actual artifacts use sibling locations.
* **Report scope is mixed:** `engine/forensic_report.json` joins individual seed-42 S1/S2 results with group seed-42 S3 results. It is not a single scenario run.
* **Single- and multi-seed verdicts differ:** single `VerdictEngine` has no clean-accuracy guard and calculates a so-called pooled SD across contributors in `LeaveOneOutEngine`; multi-seed individual verdicts use per-seed ΔASR population SD plus a clean-drop guard. Group multi-seed data exists but is not routed through `MultiSeedGroupVerdictEngine`.
* **Artifact provenance is incomplete:** the lone `.pt` file does not record its seed, scenario, metrics, package versions, or exact source revision. Scenario metadata does not preserve selected poison indices.
* **Notebook provenance is uncertain:** it has executed cells but no recorded link to a particular source/data version or experiment output.
* **Documentation is sparse:** beyond result reports and the pre-existing audit, no README/command guide is present in the currently visible repository.
