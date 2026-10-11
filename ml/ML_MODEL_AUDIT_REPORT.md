# PoisonTrace ML Model Audit

**Audit scope:** entire `ml/` tree as present on 2026-10-11. This is an evidence audit, not a design specification. Source, dataset artifacts, tests, checkpoint, notebook, and result files were inspected. No ML implementation was modified and no training was rerun.

## Executive finding

PoisonTrace implements a small Fashion-MNIST provenance-forensics experiment: five contributor shards are combined, C3 is backdoored, C5 is a noisy decoy, and leave-one-out retraining measures whether removing a contributor removes the backdoor. The strongest saved evidence is the three-seed S1+S2 experiment: C3 has mean `ΔASR=0.986815` and is **CAUSALLY IMPLICATED**; C1, C2, C4, and C5 are **CLEARED**. The source and artifacts support that conclusion for this narrowly constructed scenario.

The repository also implements a C2+C3 collusion scenario and has one saved *single-seed* group result. It does **not** contain the output expected from its multi-seed group script, nor saved individual-removal results for collusion. Thus the claim that collusion has been robustly validated across seeds is not supported.

## Repository map and artifact status

| Path / item | Purpose and status |
|---|---|
| `src/data/` | Download, shard construction, loading, normalization helper, split helper, and basic shard-print verifier. Active data path is `dataset.py` + `create_shards.py`; `preprocessing.py` and `splitting.py` are not imported by the train path. |
| `src/attacks/` | Canonical NumPy backdoor, trigger, S1/S2 scenario creation, and S3 collusion creation. Populated and used by training/evaluation. |
| `src/models/baseline.py` | Active `BaselineCNN`. |
| `src/training/` | Legacy/standalone training and evaluation path; active and duplicated by the engine trainer/probes. |
| `src/attribution/` | Older scripts for triage, multi-seed leave-one-out, analysis, collusion, and verdict helpers. Mostly superseded by `src/engine/`, but its `prove.py` produced the non-engine multi-seed JSON. |
| `src/engine/` | Reusable trainer, probes, triage, leave-one-out, verdicts, reporting, and group test. Active source for `experiments/results/engine/*`. |
| `data/raw/FashionMNIST/raw/` | Local IDX Fashion-MNIST train/test files, both uncompressed and `.gz`; populated. |
| `data/processed/train/contributors/` | Five populated contributor `.npz` files. |
| `data/poisoned/` | Four scenario `.npz` artifacts and two metadata JSONs. Populated. |
| `models/scenario_all_contributors.pt` | One populated state dict. No accompanying seed/config/metric metadata. |
| `experiments/configs/j2_multiseed.yaml` | Populated apparent experiment configuration, but no code loads it. |
| `experiments/results/` | Legacy and engine result/report families; populated, with duplicated representations of the S1/S2 evidence. |
| `notebooks/01_data_exploration.ipynb` | Two code cells; execution counts 2 and 3, one output. Exploratory and not part of pipeline. |
| `tests/` | Six source test files (including `conftest.py`) plus generated `__pycache__`; 12 tests passed in this audit. |
| `.gitignore` | **Empty (0 bytes).** |
| `.venv`, `.pytest_cache`, `__pycache__` | Generated environment/cache artifacts. The source tree contains many tracked/present bytecode artifacts; their currency is not verifiable. |

No empty directories were found in the recursively visible `ml/` tree. The only empty regular file found was `ml/.gitignore`.

## Actual architecture

```text
Fashion-MNIST local raw files (60,000 train / 10,000 test)
        |
create_shards.py: balanced 20,000-sample contributor pool
        |
C1..C5 clean 4,000-sample NPZ shards (disjoint by exact image bytes)
        |
create_scenario.py                 create_collusion_scenario.py
 C3: 8% backdoor; C5: 10% noise      C2+C3: each 8% backdoor
        |                                      |
build_training_dataset() -> ConcatDataset -> CPU Adam training -> BaselineCNN
        |
official 10,000-image test probes: clean accuracy + triggered non-class-0 ASR
        |
TriageEngine (optional suspicion ranking; no saved run)
        |
LeaveOneOutEngine / legacy prove.py: retrain all, then retrain without each C
        |
multi-seed aggregation and MultiSeedVerdictEngine (S1+S2 saved)
        |
single-seed GroupAttributionEngine (C2+C3 saved) -> group verdict
```

Important actual behavior: the value `scenario="original"` does **not** mean clean. In `src/training/train.py::build_training_dataset`, every non-`"collusion"` scenario loads poisoned C3 and noisy C5, with clean data for the other contributors. The label is therefore misleading; the saved "original" leave-one-out runs are the S1+S2 scenario.

## Data and contributor audit

`src/data/download_dataset.py::download_fashion_mnist()` uses `torchvision.datasets.FashionMNIST(..., download=True)`. Local raw IDX artifacts exist, so current evaluation/training uses `download=False`. Fashion-MNIST is 28×28 grayscale (`uint8`) with 10 classes; the canonical distribution is 60,000 train and 10,000 test. The code uses no validation split. `create_shards.py` selects 2,000 shuffled examples from each training class (20,000 total), then shuffles and partitions five equal shards. The remaining 40,000 training examples are neither used nor materialized as validation/tuning data.

`ContributorDataset.__getitem__()` converts each stored `uint8` image to `float32 / 255.0` and adds a channel dimension, yielding `[1,28,28]`. There is no centering/standard-deviation normalization and no augmentation. `preprocessing.normalize_features()` and `splitting.split_dataset()` exist but are not on the actual pipeline; the latter would make a stratified 80/20 split if called.

Direct NPZ inspection found all files have 4,000 unique images, range 0–255, and no exact image-byte overlap between any pair of clean contributor shards. This verifies disjointness at image-byte level, but does not prove source-index disjointness or semantic near-duplicate absence.

| Contributor | Clean class counts (0..9) | S1/S2 role | S3 role |
|---|---|---|---|
| C1 | 431,383,380,398,409,387,387,393,406,426 | clean | clean |
| C2 | 406,379,393,398,403,409,400,411,434,367 | clean | 320 backdoored images/labels |
| C3 | 369,413,403,395,406,399,402,389,389,435 | 320 backdoored images/labels | 320 backdoored images/labels |
| C4 | 366,419,416,420,406,397,405,405,371,395 | clean | clean |
| C5 | 428,406,408,389,376,408,406,402,400,377 | 400 noisy images/labels | clean |

Scenario artifact label counts corroborate the intended target-0 increase: C3 S1 has `[689,384,369,365,365,365,375,344,350,394]`; C2 and C3 S3 have respectively `[726,350,362,365,361,374,368,370,396,328]` and `[689,388,370,370,361,362,372,350,354,384]`. `C5_noisy.npz` has `[424,410,394,386,385,409,404,410,396,382]`. Metadata contains aggregate counts/parameters only—no selected index list—so the exact intended poison membership cannot be recovered from metadata alone. Comparing clean and generated artifacts directly verifies 320 changed images/labels in each backdoor artifact and 400 in C5; all four retain 4,000 unique stored images.

## Attack implementation

`src/attacks/backdoor.py::add_backdoor()` copies arrays, creates `np.random.default_rng(seed)`, restricts candidates to labels not equal to target 0, selects `int(len(images)*0.08)` without replacement, writes `255` to `images[idx, -3:, -3:]`, and sets those labels to 0. It is deterministic for identical input/seed. Test coverage verifies count, non-target candidate eligibility, copied input, 3×3 patch, label replacement, and determinism.

`create_scenario.py` applies that attack to C3 with seed 42; it writes `C3_poisoned.npz` and metadata declaring 320 poisons. It separately selects exactly 400 C5 examples with RNG seed 42; each gets a uniformly sampled alternative label in 0–9 (resampling if unchanged), and only selected images are multiplied by 1.10 then clipped to `[0,255]`. This is label noise plus brightness change, not a backdoor.

`create_collusion_scenario.py` independently backdoors C2 and C3 with seeds 102 and 103, respectively; each uses the same 8%, target 0, lower-right 3×3 patch. Both participants individually contain a full backdoor; this is duplicated malicious evidence, not a split-trigger or jointly necessary attack. `apply_trigger()` writes `1.0` to the corresponding patch on normalized NCHW tensors.

## Model and training

`BaselineCNN` is PyTorch and has **1,701,130 trainable parameters** (computed from the implementation).

| Stage | Operation / output |
|---|---|
| Input | `N×1×28×28` |
| Features 1–2 | Conv 1→32, 3×3 pad 1 + ReLU; Conv 32→64, 3×3 pad 1 + ReLU → `N×64×28×28` |
| Pool | MaxPool 2 → `N×64×14×14` |
| Features 3 | Conv 64→128, 3×3 pad 1 + ReLU → `N×128×14×14` |
| Pool | MaxPool 2 → `N×128×7×7` |
| Classifier | Flatten 6,272; Linear 6,272→256 + ReLU; Dropout 0.2; Linear 256→10 logits |

`extract_features()` returns the 256-dimensional ReLU representation before dropout/final linear layer. The compact CNN is suitable for a controlled 28×28 proof-of-concept, but there is no comparison with clean-only training, alternate architecture, real federated aggregation, adversarially adaptive attack, or calibration analysis.

`train_model()` and `EngineTrainer.train()` are near-duplicates: CPU only, Adam (`lr=0.001`), cross-entropy, 8 epochs, batch size 128, shuffled `DataLoader(num_workers=0)`, and a freshly initialized model each call. Seeds set Python, NumPy, Torch CPU and CUDA RNGs plus cuDNN deterministic/benchmark flags. CPU use and fixed worker count improve repeatability; no `torch.use_deterministic_algorithms`, version lock, hardware capture, or persisted training log exists. The standalone main alone saves `models/scenario_all_contributors.pt`; engine/leave-one-out training saves JSON metrics but no checkpoints.

## Metrics and causal methodology

Clean accuracy is `correct / 10,000` on the official test set. ASR is computed over only test examples whose true label is not 0 (9,000 examples in balanced Fashion-MNIST): apply the 3×3 trigger, then `predicted_class==0 / eligible`. It is a targeted trigger-response rate, not accuracy on triggered labels. `src/training/evaluate.py` and `src/engine/probes/metrics.py` implement the same calculation.

For each contributor C, leave-one-out retrains from scratch with C absent:

`ΔASR(C) = ASR(all contributors) − ASR(without C)`.

This is a counterfactual causal attribution test within the fixed dataset/training protocol. It cannot establish culpability beyond the synthetic supplied data or distinguish correlated contributors absent appropriate group testing.

The authoritative multi-seed engine decision (`MultiSeedVerdictEngine`) uses mean baseline ASR `B`, mean and population-standard-deviation (`ddof=0`) of per-seed ΔASR, and mean clean-accuracy drop:

* implicated if `mean ΔASR >= .50B`, `mean ΔASR > 2*std(ΔASR)`, and clean drop `<= .02`;
* cleared if `mean ΔASR < .10B`; otherwise inconclusive.

The **clean guard is absent** from `VerdictEngine` (single-seed) and group verdicts. Also, neither `2×std` condition is a hypothesis test, confidence interval, p-value, correction for repeated testing, nor a conventional pooled standard deviation. With only three fixed seeds, it is a heuristic stability gate.

## Saved S1+S2 experimental evidence

`experiments/results/engine/multi_seed_engine.json` and the legacy `multi_seed_leave_one_out.json` agree numerically: separate full/LOO models were trained for seeds 42, 123, and 2024. Baseline: clean accuracy `0.904700 ± 0.001485` and ASR `0.994000 ± 0.001485` (population SD in engine output). The legacy Markdown rounds clean SD differently (`0.19%` vs engine `0.16%`), so it is a presentation derivative rather than the authority.

| Contributor | ASR all mean | ASR without mean | ΔASR mean ± SD | Clean drop mean | Multi-seed verdict |
|---|---:|---:|---:|---:|---|
| C1 | 0.994000 | 0.995259 | -0.001259 ± 0.003039 | 0.010600 | CLEARED |
| C2 | 0.994000 | 0.992000 | 0.002000 ± 0.001996 | 0.010600 | CLEARED |
| C3 | 0.994000 | 0.007185 | 0.986815 ± 0.000962 | 0.008633 | CAUSALLY IMPLICATED |
| C4 | 0.994000 | 0.994296 | -0.000296 ± 0.001588 | 0.011033 | CLEARED |
| C5 | 0.994000 | 0.988074 | 0.005926 ± 0.003738 | 0.007967 | CLEARED |

The saved final verdict at `engine/multi_seed_forensic_verdict.json`, derivative `engine/multi_seed_analysis.json`, `final_verdicts.json`, and two Markdown reports all support the same S1/S2 conclusion. C3 is the only backdoor contributor in that scenario and removal collapses ASR to roughly 0.7%; C5 has noisy data but does not produce a causal-size ASR change.

**Single-seed caution:** `LeaveOneOutEngine.apply_verdicts()` calculates its `pooled_std` as `np.std` over the five *contributors’* ΔASRs, which yields about 0.395; it is not a within-contributor/run uncertainty estimate. The code labels it a placeholder, but `leave_one_out_engine.json` and `forensic_report.json` expose it as a threshold (`0.789794`). Do not use that quantity as statistical proof. Its C3 conclusion happens to agree with the later multi-seed analysis.

## Triage audit

`src/engine/triage/triage.py::TriageEngine` loads C3/C5 from poisoned artifacts and others clean, then computes: per-example cross-entropy mean/std; two-cluster KMeans (`random_state=42`, `n_init=10`) over 256-D features; and mean target-0 probability change after applying the known trigger. It ranks loss and minority-cluster fraction descending; lower rank sum is more suspicious. Min-max normalized loss and cluster scores are weighted 0.5/0.5 for `normalized_suspicion_score` (HIGH ≥ .60, MEDIUM ≥ .30). The trigger response is normalized but **explicitly excluded** from the final score.

Triage is therefore suspicion ranking, not proof. It conditions on knowledge of the exact trigger and compares contributors’ data using the already trained model. No triage output JSON/report exists, so observed real-model rankings are **not verifiable from the current repository**. Tests use a dummy model and only verify contributor coverage, field presence, and repeatability, not discriminatory performance.

## Collusion/group attribution

The engine group test retrains collusion baseline and without `[2,3]`, then `GroupVerdictEngine` implicates when group ΔASR is at least 50% of baseline; it has no standard-deviation or clean-accuracy guard. Saved `engine/group_attribution_engine.json` (seed 42) reports ASR all `0.993222`, ASR without C2+C3 `0.021444`, ΔASR `0.971778`, clean drop `0.032700`, hence `forensic_report.json` says **COLLUDING GROUP IMPLICATED**. This establishes only a seed-42 group-removal result under a scenario where each collaborator received a full backdoor.

`multi_seed_group_test.py` is implemented to run seeds 42/123/2024 and write `multi_seed_group_attribution.json`, but that file is absent. `collusion_individual.py` can remove C2 and C3 singly but saves nothing and no such output exists. Consequently, individual-removal behavior and the claimed joint-versus-individual collusion distinction are not verifiable from artifacts.

## Result inventory and authority

| File | Content / status |
|---|---|
| `results/multi_seed_leave_one_out.json` | Legacy raw 3-seed S1/S2 metrics; authoritative raw legacy evidence. |
| `results/leave_one_out_results.json` | Rounded/partial seed-42 derivative; matches engine seed 42. |
| `results/final_verdicts.json`, `results/forensic_report.md` | Derived rounded S1/S2 summaries. |
| `results/engine/multi_seed_engine.json` | Engine raw three-seed S1/S2 LOO metrics plus flawed single-seed placeholder verdicts; best detailed engine raw evidence. |
| `results/engine/multi_seed_forensic_verdict.json` | Best final multi-seed engine verdict artifact. |
| `results/engine/multi_seed_analysis.json`, `final_forensic_report.md` | Derived matching engine summaries. |
| `results/engine/leave_one_out_engine.json` | Seed-42 engine result, superseded for inference by multi-seed output. |
| `results/engine/group_attribution_engine.json`, `forensic_report.json` | Single-seed S3/group evidence; `forensic_report.json` combines this with S1/S2 seed-42 individual result, so it is not one coherent scenario experiment. |
| `results/engine/engine_test.json` | Serialization smoke artifact, not an ML result. |

## Test and notebook audit

`pytest -q` was run safely under `ml/.venv`: **12 passed in 27.45 s**, with one Pytest cache warning because `.pytest_cache/v/cache` could not be created (WinError 183). `test_attacks.py` has three meaningful backdoor unit tests. `test_detector.py` has two numeric fixture verdict tests. `test_statistical_analysis.py` has three duplicate-local rule tests including the clean guard. `test_triage.py` has three dummy-model/data-dependent integration-style tests. `test_environment.py` only checks installed package version strings. There is no test for actual training, generated scenario contents, raw/processed data cardinality, ASR implementation, saved experiment reproducibility, checkpoint loading, group verdict, or multi-seed group output.

The notebook imports `RAW_DATA_DIR`, loads Fashion-MNIST, and appears to inspect data (`2` executed code cells; one has output). It neither trains nor emits artifacts; cell outputs may be stale because no source-version provenance is recorded.

## Empty, broken, incomplete, and consistency findings

| Severity | File(s) | Problem / evidence | Impact |
|---|---|---|---|
| HIGH | `training/train.py`, `engine/trainer/trainer.py` | Any non-`collusion` scenario silently means poisoned C3+C5; default is named `original`. | Easy to run/analyze a different experiment than the caller believes. |
| HIGH | `engine/prove/leave_one_out.py` | “pooled std” is SD across five contributors’ effects, not repeat uncertainty; saved single-seed verdict uses it. | Single-seed statistical threshold is invalid/misleading. |
| HIGH | `engine/prove/multi_seed_group_test.py` | Implemented expected output `multi_seed_group_attribution.json` is absent. | No multi-seed group evidence. |
| HIGH | `engine/report.py`, `forensic_report.json` | Combines S1/S2 individual seed-42 output with S3 group output as one report. | Cross-scenario report can be misread as one unified run. |
| MEDIUM | `engine/verdict/verdict.py`, `group_verdict.py` | Single/group verdicts omit the 2% clean-accuracy guard present in multi-seed method. Saved group clean drop is 3.27%. | Group implication is not governed by stated multi-seed safeguards. |
| MEDIUM | `experiments/configs/j2_multiseed.yaml` | Present but not loaded by any source. | Configuration cannot reproduce runs; parameters are hard-coded. |
| MEDIUM | `data/poisoned/*.npz` / metadata | Poison/noise indices are not saved. | Exact selected membership is not independently auditable from metadata. |
| MEDIUM | `models/scenario_all_contributors.pt` | No seed/scenario/config/hash/metric provenance. | Checkpoint cannot be linked confidently to a result. |
| MEDIUM | `attribution/`, `training/` vs `engine/` | Substantial duplicate implementations and verdict paths; `MultiSeedAttributionEngine` returns results but does not save them. | Drift and ambiguous authority risk. |
| LOW | `data/verify_shards.py` | Creates unused `all_indices`; prints shapes/distributions but performs no overlap test. | Name overstates verification. |
| LOW | `data/preprocessing.py`, `data/splitting.py`, `utils/metrics.py` | Populated helpers not invoked by pipeline. | Dead/experimental-looking code. |
| LOW | `training/evaluate.py` | Imports `BaselineCNN` but does not use it. | Stale import. |
| LOW | `.gitignore` | Empty. | Generated environments/caches may pollute version control. |
| INFO | `__pycache__`, `.pytest_cache`, `.venv` | Generated files/directories exist in ML tree. | Audit/noise and reproducibility hygiene concern. |

No `TODO`, `FIXME`, `NotImplementedError`, or literal `pass` placeholder was found in source text. One compiled cache references a nonexistent `multi_seed_group.py` artifact; this is generated-bytecode evidence only and should not be treated as an active import failure.

## Reproducibility rating: PARTIAL

Positive evidence: raw data, all processed/poisoned shards, scenario parameters, three seeds, CPU hyperparameters, test set, code, requirements list, and raw S1/S2 results are present. The experiments genuinely retrain for each saved seed. Constraints: requirements are unpinned; YAML is unused; no command runner/readme; no git revision, package versions, data hashes, selected indices, model provenance, training logs, or multi-seed S3 output; result writers use relative paths in several scripts. Results can plausibly be replicated in the same environment, but not established as bitwise or fully independently reproducible.

## End-to-end status

| Component | Implemented | Tested | Experiment run | Result saved | Confidence |
|---|---|---|---|---|---|
| Dataset/download | Yes | environment only | local files exist | raw files | Medium |
| Sharding | Yes | No | artifacts exist | five NPZ | High |
| Backdoor | Yes | Yes | S1/S3 artifacts | NPZ+metadata | High |
| Noisy decoy | Yes | No direct test | S2 artifact | NPZ+metadata | High |
| CNN/training | Yes | No real training test | yes | one checkpoint/metrics | Medium |
| Evaluation | Yes | No direct test | yes | metrics JSON | Medium |
| Triage | Yes | Dummy-model only | not verifiable | none | Low |
| LOO causal proof | Yes | Fixture rules | three seeds S1/S2 | JSON/reports | High |
| Multi-seed verdict | Yes | Fixture rules | three seeds S1/S2 | JSON/report | High for narrow scenario |
| Group attribution | Yes | No | one S3 seed | JSON | Medium/low |
| Multi-seed group proof | Yes | No | not verifiable | missing | Low |

## Scientifically justified conclusions

**Demonstrated:** under the stored Fashion-MNIST S1+S2 construction, with one fully backdoored C3 shard and one noisy C5 shard, the saved three deterministic CPU seeds consistently show C3 removal nearly eliminates the specific known-trigger ASR while preserving clean accuracy within the configured 2-point guard. The saved multi-seed engine therefore supports C3’s causal implication and C5’s clearance under its own rule.

**Strongly supported with limitations:** the S3 artifact construction means removing both fully backdoored contributors C2+C3 produced a large ASR reduction in one saved seed. This is evidence that the pair's data accounts for the outcome, but not robust multi-seed proof of collusion.

**Not yet demonstrated:** reliable triage identification on actual results; multi-seed group attribution; individual-versus-joint removal behavior in S3; generalization to unknown triggers, different targets/poison rates, non-IID client data, other datasets/models, or real distributed/federated training.

**Unsupported claims:** that the group result has passed the same statistical/clean-accuracy criteria as individual multi-seed proof; that the pipeline runs from the YAML; that all documentation/report artifacts describe a single same-scenario experiment; or that this establishes real-world malicious intent.

## Final technical summary and next verification targets

PoisonTrace is a well-contained synthetic forensic prototype, not a production attribution system. It uses balanced Fashion-MNIST contributor shards, direct pixel-patch/label-flip backdoors, CPU CNN retraining, targeted ASR, a heuristic triage ranker, and the more meaningful leave-one-out counterfactual experiment. The principal evidence is in `experiments/results/engine/multi_seed_engine.json` and `multi_seed_forensic_verdict.json`; S1/S2 is the current defensible result. Prioritize independent rerun with version capture, generate/persist multi-seed group results and individual S3 removals, save poison indices/checkpoint provenance, make scenario semantics explicit, and avoid using the single-seed pooled-SD or cross-scenario forensic report as scientific evidence.
