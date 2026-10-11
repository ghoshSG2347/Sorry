# PoisonTrace Final ML Audit

**Scope:** Read-only audit of the current `ml/` tree. Source code, generated data/model/result artifacts, configuration, notebook, tests, and dependency/ignore files were inspected. The saved S1/S2 and S3 experiment outputs were treated as frozen evidence; no training, scenario generation, or result generation was run. The only write associated with this audit is this required report.

## Audit conclusion

### Overall ML Status

**READY WITH MINOR FIXES**

The core controlled experiments are complete: the repository contains multi-seed S1/S2 individual attribution and S3 individual-plus-joint-removal collusion evidence, and their saved results agree with the applicable code. It is ready to integrate as a synthetic Fashion-MNIST forensic prototype if backend integration uses the saved final verdict artifacts and preserves scenario scope. The remaining issues are primarily guard/provenance/documentation issues; they do not invalidate the demonstrated S1/S2 or S3 ASR effects.

## 1. Evidence classification

The audit uses the following distinction throughout:

| Label | Meaning |
|---|---|
| **Code exists** | An implementation is present in `src/`. |
| **Test exists** | A unit/integration-style test covers some behavior. |
| **Real evidence** | A saved artifact contains output from a real trained-model experiment. |
| **Multi-seed evidence** | Saved output aggregates/reports more than one training seed. |

Code existence is not treated as real or multi-seed validation.

## 2. Data pipeline

### Status

| Item | Status | Evidence |
|---|---|---|
| Fashion-MNIST source | **IMPLEMENTED** | `src/data/download_dataset.py` uses torchvision FashionMNIST; local IDX/gzip train/test files exist under `data/raw/FashionMNIST/raw/`. |
| Contributor shard creation | **IMPLEMENTED** | `src/data/create_shards.py` takes 2,000 examples/class (20,000 total), shuffles seed 42, and writes five 4,000-sample shards. |
| Local shard artifacts | **IMPLEMENTED / evidenced** | `data/processed/train/contributors/C1.npz` through `C5.npz` exist. Each is 4,000×28×28 `uint8`, labels length 4,000. |
| Class balance | **IMPLEMENTED / evidenced** | Selection starts class-balanced globally; per-contributor assignment is shuffled, so individual shards are approximately, not exactly, balanced. Saved shard label counts support this. |
| Input preprocessing | **IMPLEMENTED** | `ContributorDataset.__getitem__()` converts `uint8` to `float32/255.0` and adds a single channel. No mean/std normalization or augmentation is used. |
| Contributor disjointness | **PARTIALLY IMPLEMENTED** | The construction partitions selected indices, so disjointness follows by algorithm. Direct artifact inspection found no exact image-byte overlap. Source indices are not saved and `verify_shards.py` only prints shape/class counts; no persistent overlap/duplicate validation exists. |
| Train/validation split | **NOT REQUIRED for the saved protocol** | The experiments use the official test set for evaluation. `splitting.py` exists but is not used. There is no validation artifact. |
| Poison selection provenance | **PARTIALLY IMPLEMENTED** | Parameters/counts are stored in metadata, but selected poison/noise indices are not persisted. |

`data/processed/clean`, `metadata`, `poisoned`, `test`, and `validation` are empty. They are not on the active data path. The active clean shards are stored in `data/processed/train/contributors/`; scenario replacements are stored in `data/poisoned/`.

### S1/S2/S3 data definitions and artifact cross-check

| Scenario | Implementation | Saved artifact evidence |
|---|---|---|
| S1 | `create_scenario.py` backdoors C3 with 8% (`320/4,000`) poison rate, target 0, lower-right 3×3 patch, seed 42. | `C3_poisoned.npz` and `scenario_metadata.json` exist; direct comparison to clean C3 finds 320 changed images and labels. |
| S2 | Same script selects 10% (`400/4,000`) C5 examples with RNG seed 42, changes each to a different random label, and applies `clip(image*1.10, 0, 255)`. | `C5_noisy.npz` and metadata exist; comparison finds 400 changed images/labels. |
| S3 | `create_collusion_scenario.py` independently applies the same full 8% backdoor to C2 (seed 102) and C3 (seed 103). | `C2_collusion.npz`, `C3_collusion.npz`, and `collusion_metadata.json` exist; each has 320 changed images/labels. |

S3 is a valid **joint-necessity collusion experiment** for this synthetic construction: either participant’s full backdoor can sustain high ASR after the other is removed, whereas removing both eliminates the signal. It is not a split-trigger or individually incomplete-payload construction; each participant independently receives a complete backdoor.

## 3. Attack implementation

`src/attacks/backdoor.py::add_backdoor()` copies input arrays, creates `np.random.default_rng(seed)`, chooses `int(len(images)*poison_fraction)` candidates among labels not equal to target 0 without replacement, writes `255` into `[-3:, -3:]`, and relabels selected items to 0. `src/attacks/trigger.py::apply_trigger()` applies the corresponding patch value `1.0` to normalized Torch images at inference.

The functional attack is tested for count, target-class exclusion, non-mutation, patch behavior, and deterministic identical-seed output in `tests/test_attacks.py`. A duplicate class implementation, `src/engine/attacks/backdoor.py::BackdoorAttack`, has the same behavior but is not imported by the current scenario/train pipeline.

## 4. Training and model verification

`src/models/baseline.py::BaselineCNN` is a PyTorch CNN with three 3×3 convolution layers (1→32→64→128), ReLUs, two 2×2 max pools, flattening from 128×7×7, a 256-unit ReLU layer, dropout 0.2, and a 10-logit output. The implementation has **1,701,130 trainable parameters**.

`src/training/train.py::train_model()` and `src/engine/trainer/trainer.py::EngineTrainer.train()` are near-duplicate CPU training loops: fresh model initialization, Adam at 0.001, cross-entropy, 8 epochs, batch size 128, shuffled `DataLoader(num_workers=0)`. Both seed Python, NumPy, Torch CPU and CUDA RNGs and set cuDNN deterministic/benchmark flags. No GPU training is selected; device is hard-coded as CPU.

### Important scenario behavior

**`scenario="original"` is not a clean dataset.** In `build_training_dataset()`, every scenario other than exactly `"collusion"` loads C3 and C5 as `scenario="poisoned"`, while C1/C2/C4 use clean shards. The default `scenario="original"` therefore means the S1+S2 mixture. In collusion mode, C2/C3 use collusion artifacts and the other contributors use clean artifacts. This is actual code behavior, not a naming interpretation.

The standalone `train.py` main can save `models/scenario_all_contributors.pt`. That checkpoint contains a state dict but no saved seed, scenario, metric, package-version, or dataset hash provenance.

## 5. Evaluation metrics

Both `src/training/evaluate.py` and `src/engine/probes/metrics.py` implement the same metrics:

* **Clean accuracy:** correct class predictions divided by all 10,000 official Fashion-MNIST test examples.
* **ASR:** exclude true class-0 test items, apply the known lower-right 3×3 trigger, and calculate predictions equal to target 0 divided by the 9,000 eligible non-target items.
* **ΔASR:** `ASR(all) - ASR(without contributor or group)`.
* **Clean-accuracy drop:** `clean_accuracy(all) - clean_accuracy(without contributor or group)`.

No material definition discrepancy was found between legacy and engine clean-accuracy/ASR probes. Their saved S1/S2 raw results agree. ASR is targeted response to a **known** trigger, not broad backdoor discovery or triggered classification accuracy.

## 6. Triage verification

`src/engine/triage/triage.py::TriageEngine` exists and computes, for each contributor:

1. mean and population-standard-deviation cross-entropy on that contributor’s selected scenario data;
2. two-cluster KMeans (`random_state=42`, `n_init=10`) over `BaselineCNN.extract_features()` 256-D representations, scored as minority-cluster fraction;
3. mean change in probability of target 0 after applying the known trigger;
4. ranks loss and cluster score descending, sums ranks, and min-max normalizes signals;
5. final `normalized_suspicion_score = 0.5*normalized_loss + 0.5*normalized_cluster`, with HIGH/MEDIUM/LOW thresholds of 0.60/0.30.

The trigger-response score is calculated and normalized but **explicitly excluded** from the final suspicion score. Triage is therefore a suspicion ordering rather than causal proof.

| Aspect | Code | Tests | Real result | Multi-seed result |
|---|---|---|---|---|
| Triage | Yes | Yes, dummy-model/data-backed structural tests | **No saved real-model triage output** | No |

The suite checks contributor coverage, required fields, and deterministic ranking with a `DummyModel`; it does not establish real-model triage discrimination. No claim that triage works experimentally is justified by the current artifacts.

## 7. Individual attribution: S1/S2 verification

`src/engine/prove/leave_one_out.py::LeaveOneOutEngine` trains a baseline with all contributors then independently retrains five models each missing one contributor. `src/engine/multiseed.py` repeats that for seeds 42, 123, and 2024 and writes `experiments/results/engine/multi_seed_engine.json`. `src/engine/verdict/analyze_multiseed.py` aggregates values using NumPy population SD (`ddof=0`) and writes `multi_seed_forensic_verdict.json`.

The saved results support the frozen figures exactly (shown to six decimals):

| Contributor | ΔASR mean ± SD | Clean drop mean ± SD | Saved verdict |
|---|---:|---:|---|
| Baseline | ASR `0.994000 ± 0.001485`; clean accuracy `0.904700 ± 0.001485` | — | — |
| C1 | `-0.001259 ± 0.003039` | `0.010600 ± 0.003612` | CLEARED |
| C2 | `0.002000 ± 0.001996` | `0.010600 ± 0.002142` | CLEARED |
| C3 | `0.986815 ± 0.000962` | `0.008633 ± 0.001812` | CAUSALLY IMPLICATED |
| C4 | `-0.000296 ± 0.001588` | `0.011033 ± 0.003574` | CLEARED |
| C5 | `0.005926 ± 0.003738` | `0.007967 ± 0.004196` | CLEARED |

Note: individual-result clean-accuracy SD values come from `multi_seed_forensic_verdict.json`. Baseline clean accuracy is `0.904700 ± 0.001577` there; the supplied frozen `±0.001485` is the ASR SD. The source/final JSON therefore supports the stated mean but not that exact baseline clean-accuracy SD. The legacy Markdown reports a separately rounded `±0.19%`; the detailed engine JSON is the more precise source.

### Individual verdict logic

`MultiSeedVerdictEngine` implicates when all are true:

* mean ΔASR ≥ `0.50 * mean baseline ASR`;
* mean ΔASR > `2 * population_std(ΔASR)`;
* mean clean-accuracy drop ≤ 0.02.

It clears when mean ΔASR < `0.10 * mean baseline ASR`, otherwise returns inconclusive. C3 passes all three conditions; C1/C2/C4/C5 meet the cleared condition. The `2×SD` term is a **heuristic stability rule**, not a formal hypothesis test, p-value, confidence interval, or conventional pooled-standard-error analysis.

The single-seed `VerdictEngine` differs: it lacks the clean-accuracy guard. `LeaveOneOutEngine.apply_verdicts()` computes its so-called pooled SD across the five contributors’ ΔASRs, not repeated runs. The code itself calls this a placeholder; its single-seed statistical field should not be interpreted as formal uncertainty evidence. The saved multi-seed verdict artifact is the appropriate final S1/S2 evidence.

## 8. S3 collusion verification

The required frozen artifacts exist under `experiments/results/engine/`:

* `multi_seed_collusion_individual.json` contains seeds 42, 123, 2024 and the four contributor configurations.
* `multi_seed_collusion_verdict.json` computes paired seed-by-seed ASR differences and the verdict.
* `final_collusion_forensic_report.md` renders the saved verdict.

The final collusion JSON matches the frozen evidence:

| Condition | ASR mean ± population SD | Paired ΔASR mean ± population SD |
|---|---:|---:|
| Baseline | `0.9940 ± 0.0014` | — |
| Without C2 | `0.9900 ± 0.0029` | `0.0041 ± 0.0027` |
| Without C3 | `0.9922 ± 0.0032` | `0.0018 ± 0.0035` |
| Without C2+C3 | `0.0141 ± 0.0060` | `0.9799 ± 0.0060` |

`src/engine/prove/multi_seed_collusion_individual.py` retrains each of the four configurations for all three seeds. `src/engine/verdict/multi_seed_collusion_verdict.py` preserves seed pairing and declares collusion if average ASR without C2 > 0.90, average ASR without C3 > 0.90, and average ASR without both < 0.10. All saved booleans are true, and its resulting verdict is **COLLUSIVE GROUP IMPLICATED**.

This verdict code does **not** use the individual causal threshold, cleared threshold, `2×SD` stability rule, or any clean-accuracy safeguard. It is a separate deterministic collusion criterion, not the same evidentiary rule as S1/S2 individual attribution. It does calculate group clean-accuracy drop (`0.021500 ± 0.008392`), but does not use it in its verdict. This is the principal readiness limitation: the group criterion establishes synthetic joint dependence, but it does not guard against a verdict obtained only alongside an unacceptable clean-accuracy change. The observed mean group drop is slightly above the individual 0.02 guard; the individual guard is not formally specified as a group requirement, so this is a consistency/credibility issue rather than a falsification of the saved S3 ASR result.

`multi_seed_group_attribution.json` is an additional three-seed group-removal aggregate produced by a separate path. It agrees materially with the individual-collusion artifact but uses sample SD (`statistics.stdev`) rather than population SD; this accounts for different displayed SDs. It is supplementary, not the final S3 verdict source.

## 9. Verdict and report audit

| Verdict/report component | Code | Test | Real evidence | Multi-seed evidence | Audit result |
|---|---|---|---|---|---|
| `VerdictEngine` single individual | Yes | Fixture tests | Seed-42 artifact | No | Uses threshold heuristic; no clean guard. |
| `MultiSeedVerdictEngine` individual | Yes | Equivalent local fixture rule test | Final S1/S2 JSON | Yes | Correctly produces saved S1/S2 verdicts. |
| `GroupVerdictEngine` single group | Yes | No direct test | Seed-42 group artifact | No | No statistical/clean guard. |
| S3 collusion verdict script | Yes | No direct test | Final collusion JSON/report | Yes | Correctly reproduces its own >.90/.10 logic; no clean guard/stability rule. |
| Individual forensic Markdown report | Yes | No direct test | Exists | Yes | Derived from final individual JSON. |
| S3 forensic Markdown report | Yes | No direct test | Exists | Yes | Derived from final collusion JSON. |

`engine/forensic_report.json` combines a seed-42 S1/S2 individual result with a seed-42 S3 group result. It is not a single coherent scenario experiment and should not supersede the separate final multi-seed reports.

## 10. Result provenance and reproducibility

### Present in saved multi-seed evidence

* seeds 42, 123, 2024;
* contributor configurations and per-seed S1/S2 or S3 metrics;
* S1/S2 engine hyperparameters in `multi_seed_engine.json` (8 epochs, batch size 128, learning rate 0.001, scenario `original`);
* S3 individual per-seed contributor configurations; fixed code constants identify identical training settings;
* baseline/leave-out ASR and clean accuracy metrics, deltas, standard deviations, and final verdicts;
* scenario metadata naming contributor(s), rates, target class, patch size, and scenario category.

### Material reproducibility gaps

* No persisted selected poison/noise index lists, source-dataset index lists, input artifact hashes, package/version lock, hardware/OS record, or source revision.
* The standalone checkpoint has no configuration/result provenance.
* `requirements.txt` is unpinned; `j2_multiseed.yaml` is not read by the implementation, so it is not the executed configuration source.
* Engine result timestamps are present only for serializer-wrapped files; the final multi-seed JSONs lack an immutable provenance envelope.

These gaps do not contradict the frozen result numbers but limit independent bitwise reproduction and external judging confidence.

## 11. Tests

The lightweight suite was run with `python -m pytest -q` from `ml/`.

* **12 passed, 0 failed** in 13.59 seconds.
* **1 warning:** pytest could not create `.pytest_cache/v/cache/nodeids` because a path already exists (`WinError 183`). This did not affect test outcomes.

Coverage includes backdoor mechanics, fixture verdict behavior, environment import availability, and structural/determinism checks for dummy-model triage. Coverage gaps include full training, evaluation/ASR calculations, actual triage effectiveness, data shard/poison generation artifacts, S3 collusion verdict behavior, report generation, and persisted-result provenance.

## 12. Conceptual pipeline completeness

| Stage | Status | Code | Tests | Real evidence | Multi-seed evidence |
|---|---|---|---|---|---|
| Dataset | [COMPLETE] | Yes | Environment only | Local raw files/shards | N/A |
| Contributor shards | [COMPLETE] | Yes | No direct test | Five NPZ shards | N/A |
| Attack generation | [COMPLETE] | Yes | Backdoor only | S1/S2/S3 NPZ artifacts | N/A |
| Training | [COMPLETE] | Yes | No full-train test | Checkpoint/result metrics | Yes, S1/S2/S3 |
| Evaluation | [COMPLETE] | Yes | No direct metric test | Saved metrics | Yes, S1/S2/S3 |
| Triage | [PARTIAL] | Yes | Dummy-model structural tests | No saved real-model output | No |
| Individual attribution | [COMPLETE] | Yes | Fixture logic | Saved LOO records | Yes, S1/S2 |
| Multi-seed verification | [COMPLETE] | Yes | Fixture logic | Final individual JSON | Yes, S1/S2 |
| Collusion testing | [COMPLETE] | Yes | No direct test | Saved individual/joint conditions | Yes, S3 |
| Verdict | [COMPLETE] | Yes | Partial fixture coverage | Saved final JSONs | Yes, separate S1/S2 and S3 rules |
| Forensic reports | [COMPLETE] | Yes | No direct test | Markdown reports | Yes, individual and S3 reports |

## 13. What is proven

* In the stored Fashion-MNIST S1/S2 data construction, C3’s 8% targeted patch-and-label backdoor produces high known-trigger ASR, and removing C3 reduces mean ASR by `0.986815` across the three saved seeds while meeting the individual clean-drop guard.
* In the same S1/S2 experiment, saved multi-seed individual verdict logic clears C1, C2, C4, and the noisy C5 decoy and causally implicates C3.
* In the stored S3 construction, C2-only removal and C3-only removal preserve high ASR, while joint C2+C3 removal reduces ASR to `0.0141` on average across the three saved seeds.
* The final S3 script correctly derives `COLLUSIVE GROUP IMPLICATED` from its implemented >0.90, >0.90, and <0.10 criterion.

## 14. What is not proven

* Real-model triage ranks/identifies the attacker effectively; no saved real-model triage run exists.
* Generalization to unknown triggers, targets, rates, datasets, model families, non-IID/federated aggregation, adaptive attackers, or real-world malicious intent.
* A formal statistical significance claim: `2×SD` is not a formal test, and the S3 verdict does not use it.
* That S3 satisfies an individual-style clean-accuracy safety criterion; no group clean guard is implemented or enforced.
* Bitwise independent reproduction from the repository alone, due to missing index/hash/version/source-revision provenance.

## 15. Issues and smallest next actions

### Critical issues (0)

No defect was found that invalidates the frozen S1/S2 or S3 numerical evidence under their stated synthetic protocols.

### Minor issues (5)

1. **S3 clean-performance guard is absent:** group clean drop is recorded but ignored by its verdict; the observed mean is `0.0215`, slightly above the individual 0.02 guard.
2. **Triage lacks empirical evidence:** only dummy-model tests exist; no trained-model output is saved.
3. **Scenario naming ambiguity:** `scenario="original"` actually loads S1/S2 poisoned contributor artifacts.
4. **Result provenance is incomplete:** poison/source indices, hashes, version lock/source revision, and checkpoint metadata are absent.
5. **Parallel/duplicate paths and reports:** legacy and engine implementations coexist; the combined single-seed `forensic_report.json` mixes different scenarios; `multi_seed_group_attribution.json` and final S3 analysis use different SD conventions.

### Recommended next actions

Before backend integration, the smallest justified actions are:

1. Ensure backend presentation consumes the designated final multi-seed JSON/report artifacts separately for S1/S2 and S3, rather than the mixed `engine/forensic_report.json`.
2. Surface the S3 clean-accuracy drop and the fact that the group verdict has no clean-performance guard; do not represent it as the same criterion as the individual verdict.
3. Treat triage as an optional suspicion feature until a saved real-model run exists.

These are integration/communication safeguards; no re-training is required to preserve the existing experimental claims.

## 16. Final status table

| Component | Implementation | Tests | Real Evidence | Multi-Seed Evidence | Status |
|---|---|---|---|---|---|
| Dataset | Yes | Environment only | Raw Fashion-MNIST present | N/A | COMPLETE |
| Sharding | Yes | No direct test | Five shards present | N/A | COMPLETE |
| S1 Backdoor | Yes | Yes | C3 artifact/results | Via S1/S2 | COMPLETE |
| S2 Decoy | Yes | No direct test | C5 artifact/results | Via S1/S2 | COMPLETE |
| S3 Collusion | Yes | No direct test | Saved S3 artifacts/results | Yes | COMPLETE |
| Training | Yes | No full-train test | Metrics/checkpoint | Yes | COMPLETE |
| Evaluation | Yes | No direct test | Saved metrics | Yes | COMPLETE |
| Triage | Yes | Dummy-model only | No | No | PARTIAL |
| LOO Attribution | Yes | Fixture logic | Saved records | Yes | COMPLETE |
| Multi-Seed Verdict | Yes | Fixture-equivalent logic | Final JSON/report | Yes | COMPLETE |
| Collusion Verdict | Yes | No direct test | Final S3 JSON/report | Yes | COMPLETE WITH LIMITATION |
| Forensic Report | Yes | No direct test | Individual/S3 Markdown | Yes | COMPLETE |
| Tests | Yes | 12 passed | Current execution | N/A | COMPLETE |
