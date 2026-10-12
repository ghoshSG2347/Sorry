# PoisonTrace — ML Subsystem Handoff & Operation Manual

## 1. Provenance and Source Repository

- **ML Source Repository**: `https://github.com/ghoshSG2347/Sorry` (and upstream `https://github.com/ghoshSG2347/Kuchbhi_banado_yaar`)
- **Author & Ownership**: Saptarshi Ghosh
- **Source Commit Hash**: `6c620eb31d2297a731ce670f0389a0f8092ea5ed`
- **Import Strategy into Integrated Monorepo**:
  The ML subsystem was imported directly into `ml/` preserving its internal package hierarchy (`ml/src/`, `ml/data/`, `ml/models/`, `ml/experiments/`, `ml/tests/`). 
  Raw and processed data files (`ml/data/raw/` and `ml/data/processed/train/contributors/`), poisoned scenario shards (`ml/data/poisoned/`), trained PyTorch checkpoint (`ml/models/scenario_all_contributors.pt`), and multi-seed experiment results (`ml/experiments/results/engine/`) have been verified and placed under version control.

---

## 2. Environment and Installation

- **Verified Python Version**: `Python 3.14.6` (compatible with Python 3.10+)
- **OS Platform Verified**: Windows 11 (x86_64 / win32)
- **Dependencies** (from `ml/requirements.txt`):
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
  - `torch`, `torchvision` (CPU-only execution by design)
  - `pyyaml`, `tqdm`, `pytest`, `jsonschema`

### Installation Commands:

```bash
# 1. Create and activate a clean virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r ml/requirements.txt
```

---

## 3. Smoke Test & Verification

To verify the ML environment, unit tests, backdoor injection mechanics, statistical verdict rules, and backend adapter integration without waiting for long retraining loops:

```bash
python -m pytest ml/tests -q
```

### Observed Execution Result (Verified 2026-10-12):
```text
.........................                                                [100%]
25 passed in 18.70s
```
All 25 tests pass (12 original tests covering backdoor injection, dummy-model triage, and single-seed verdicts + 13 integration adapter and JSON schema validation tests).

---

## 4. Supported Scenarios and Execution

PoisonTrace supports four canonical scenario IDs:

| Canonical Scenario ID | Description | Contributor Role | Attack Payload / Injection | Supported Modes |
|---|---|---|---|---|
| `S1_BACKDOOR_C3` | Single-contributor targeted backdoor | C3 (Malicious) | 8% poison (320 items), target class 0, bottom-right 3×3 patch, seed 42 | `replay`, `live` |
| `S2_NOISY_DECOY_C5` | Honest-but-noisy decoy contributor | C5 (Decoy) | 10% label noise (400 items), 1.10 brightness scaling, seed 42 | `replay`, `live` |
| `S1_S2_COMBINED` | Default forensic experiment (S1 + S2 co-present) | C3 (Malicious), C5 (Decoy), C1, C2, C4 (Clean) | Co-evaluates leave-one-out attribution on attacker vs decoy | `replay`, `live` |
| `S3_COLLUSION_C2_C3` | Multi-contributor joint collusion | C2 & C3 (Colluding) | Each has independent 8% backdoor (seed 102 & 103), target class 0, 3×3 patch | `replay`, `live` |

### How to Run via Backend Adapter (Recommended):

```python
from backend.app.services.ml_runner import run_scenario

# 1. Instant Replay Mode (Uses verified frozen results, 0.0s latency)
result = run_scenario(scenario_id="S1_S2_COMBINED", mode="replay")
print(result["contributors"]["C3"]["verdict"])  # -> CAUSALLY IMPLICATED

# 2. Live Retraining Mode (Re-trains BaselineCNN across seeds [42, 123, 2024])
# Note: Takes ~5-7 minutes on CPU
live_result = run_scenario(scenario_id="S1_S2_COMBINED", mode="live")
```

### How to Run Standalone Scripts:

```bash
# S1 + S2 Scenario Generation:
python ml/src/attacks/create_scenario.py

# S3 Collusion Scenario Generation:
python ml/src/attacks/create_collusion_scenario.py

# Execute Multi-Seed Leave-One-Out Retraining (seeds 42, 123, 2024):
python ml/src/engine/multiseed.py

# Aggregate Multi-Seed Results & Generate Final Forensic Verdicts:
python ml/src/engine/verdict/analyze_multiseed.py

# Execute Multi-Seed Collusion Evaluation:
python ml/src/engine/prove/multi_seed_collusion_individual.py
python ml/src/engine/verdict/multi_seed_collusion_verdict.py
```

---

## 5. Artifact Inventory & Hashes

All critical artifacts have been hashed with SHA-256 for audit immutability:

| Category | Artifact Path | SHA-256 Digest |
|---|---|---|
| **Protocol** | `ml/replay/protocol.json` | `133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9` |
| **Replay Fixture** | `ml/replay/s1_s2_combined_fixture.json` | Generated from `multi_seed_forensic_verdict.json` |
| **Replay Fixture** | `ml/replay/s3_collusion_fixture.json` | Generated from `multi_seed_collusion_verdict.json` |
| **Model** | `ml/models/scenario_all_contributors.pt` | `c6d57a738c49d473d14caa81efedb0cb65e0dc43e2c610099244da2aca303d5e` |
| **Clean Shard C1** | `ml/data/processed/train/contributors/C1.npz` | `8f8586faa76f86c44e43af70f6bea69b9f06acc39e5d60ac2ad91a38bbe21798` |
| **Clean Shard C2** | `ml/data/processed/train/contributors/C2.npz` | `8622c9dc9c667ba0c9cb0344eb1c0a87093a61e420f1bbcf023b658a4077c246` |
| **Clean Shard C3** | `ml/data/processed/train/contributors/C3.npz` | `d27130235ddef6fcf3a8bccf24c9839931946bec95625ebd1229f2346fe40a7a` |
| **Clean Shard C4** | `ml/data/processed/train/contributors/C4.npz` | `6368115d82c072d15083bc231f43978435ffb6967b749ff3e1ecc3e4762954d3` |
| **Clean Shard C5** | `ml/data/processed/train/contributors/C5.npz` | `736a17c3cd3aad6db2988d0be5f7311dd580fcd7410ab5cb5dd090290e6d108b` |
| **S1 Poison C3** | `ml/data/poisoned/C3_poisoned.npz` | `2a87c2f38fc3f235e3ba7913d46906b3d171cb005dd2549ff243c96fa1835ca4` |
| **S2 Decoy C5** | `ml/data/poisoned/C5_noisy.npz` | `b6b1789a909c970e31f0c81564eaf05ea07cc7617675267dcd42f30f2ab86631` |
| **S3 Collusion C2** | `ml/data/poisoned/C2_collusion.npz` | `627fabea242ed8d41fde9b2a3f500d75d9b1b6935eb610414240448408d39a7a` |
| **S3 Collusion C3** | `ml/data/poisoned/C3_collusion.npz` | `9179e6a39e2ca6c2e35798317aec9d943285537884704f887ac04f895cb47113` |

---

## 6. Limitations & Known Behaviors

1. **Synthetic Controlled Trigger**:
   The attack evaluation tests response to a known 3×3 square trigger placed at `[-3:, -3:]` (bottom-right corner) with target class 0 on Fashion-MNIST. Broad backdoor discovery on unknown triggers is not supported.
2. **Scenario Naming Convention**:
   In `src/training/train.py::build_training_dataset`, `scenario="original"` loads the S1 (C3 poisoned) + S2 (C5 noisy) combined training set. Clean-only training requires explicit clean shard loading.
3. **Statistical Threshold Rule**:
   The `2×SD` stability rule used by `MultiSeedVerdictEngine` is an empirical heuristic stability condition, not a formal Student's t-test or ANOVA p-value.
4. **Collusion Clean-Accuracy Guard**:
   In S3 collusion, removing C2+C3 reduces clean accuracy by `0.0215 ± 0.0084` (mean 2.15%). The collusion verdict checks ASR preservation (>0.90) and joint elimination (<0.10), but does not guard against clean-accuracy degradation.
5. **Triage Evidence**:
   While `TriageEngine` is structurally implemented and tested with mock models, no real-model triage metrics were persisted in historical experiment files. Triage is therefore documented as an experimental heuristic rather than verified causal proof.
