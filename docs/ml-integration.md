# PoisonTrace — ML Subsystem Integration Specification & Contract

**Author**: Saptarshi Ghosh (ML Subsystem Lead)  
**Recipient**: Priyanshu (Backend & Verifier Lead), Mondrita (Frontend Lead)  
**Date**: October 12, 2026  
**Protocol Version**: `POISONTRACE_PROTOCOL_V1` (`1.0.0`)  
**Protocol SHA-256**: `133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9`  
**Result Schema**: `docs/ml-result-schema.json` (and `ml/schema/ml_result_schema.json`)

---

## 1. Overview & Architectural Boundaries

PoisonTrace assesses contributor integrity in decentralized and collaborative model training. The ML subsystem is responsible for:
1. Data sharding and synthetic poisoning injection (Fashion-MNIST baseline).
2. Training execution of the canonical `BaselineCNN` architecture.
3. Evaluation probes: clean accuracy and targeted Attack Success Rate (ASR).
4. Causal leave-one-out (LOO) attribution across multiple deterministic random seeds.
5. Statistical verdict generation under frozen decision thresholds.

### Strict Separation of Concerns:
- **ML Subsystem (`ml/`, adapter `backend/app/services/ml_runner.py`)**: Executes model logic, loads committed replay fixtures, computes raw metrics and verdicts, and returns structured dictionaries complying with `ml_result_schema.json`.
- **Backend Subsystem (`backend/`, owned by Priyanshu)**: Manages run lifecycle, database persistence (SQLAlchemy/SQLite/Postgres), ledger sequencing, canonical JSON serialization, cryptographic signing (Ed25519/ECDSA), bundle export, and REST/WebSocket API endpoints.
- **Rule**: The ML subsystem **does not** perform cryptographic signing, database writes, or canonical byte formatting. Canonical evidence signing belongs solely to Priyanshu's backend and verification tools.

---

## 2. Module Entry Points & Adapter Interface

The adapter resides at `backend/app/services/ml_runner.py` (with mirror copy at `ml/handoff/backend/app/services/ml_runner.py`).

### Primary Python Interface:

```python
from backend.app.services.ml_runner import MLRunner, run_scenario

# Standalone function invocation
result = run_scenario(
    scenario_id="S1_S2_COMBINED",
    mode="replay",                     # 'replay' | 'live'
    run_id="run-uuid-1234",             # Optional; auto-generated if omitted
    config={"seeds": [42, 123, 2024]},  # Optional execution parameters
)

# Object-oriented invocation
runner = MLRunner()
supported_scenarios = runner.get_supported_scenarios()
protocol_data = runner.load_protocol()
result = runner.run_scenario(
    scenario_id="S3_COLLUSION_C2_C3",
    mode="replay",
    raise_on_failure=False,             # Returns status='FAILED' if False
)
```

### Exceptions:
- `MLRunnerError`: Base exception for runner failures.
- `MLScenarioNotSupportedError`: Raised when `scenario_id` is not in the allowlist.
- `MLInvalidConfigurationError`: Raised when configuration parameters fail type or value checks.

---

## 3. Supported Scenarios & Execution Modes

| Canonical Scenario ID | Display Name | Live Supported | Replay Supported | Primary Forensic Finding |
|---|---|---|---|---|
| `S1_BACKDOOR_C3` | Single-Contributor Backdoor | Yes | Yes (Instant) | C3 is **CAUSALLY IMPLICATED** ($\Delta\text{ASR} \approx 98.7\%$) |
| `S2_NOISY_DECOY_C5` | Honest-but-Noisy Decoy | Yes | Yes (Instant) | C5 is **CLEARED** ($\Delta\text{ASR} \approx 0.6\%$) |
| `S1_S2_COMBINED` | Default Forensic Audit | Yes | Yes (Instant) | C3 **CAUSALLY IMPLICATED**, C1, C2, C4, C5 **CLEARED** |
| `S3_COLLUSION_C2_C3` | Collusive Sybil Backdoor | Yes | Yes (Instant) | C2+C3 **COLLUSIVE GROUP IMPLICATED** (Joint $\Delta\text{ASR} \approx 98.0\%$) |

### Live vs. Replay Behavior:
- **`replay` mode**:
  - **Latency**: $< 1$ millisecond (instant memory/disk load).
  - **Dependencies**: Pure Python standard library (no GPU, no PyTorch required).
  - **Guarantee**: Loads verified multi-seed experiment outputs directly from `ml/replay/`.
  - **Execution Time Field**: Strictly `0.0`.
- **`live` mode**:
  - **Latency**: Measured at 5 to 7 minutes on modern 8-core CPU for full 3-seed LOO (18 models $\times$ 8 epochs).
  - **Dependencies**: Requires `torch`, `torchvision`, and NumPy shards present in `ml/data/`.
  - **Resource Usage**: ~1.5 GB RAM, 100% CPU on single socket.
  - **Execution Time Field**: Actual measured wall-clock seconds.

---

## 4. Input Configuration & Output Schema

### Valid Input Configuration Dictionary (`config`):
```json
{
  "seeds": [42, 123, 2024],
  "epochs": 8,
  "batch_size": 128,
  "learning_rate": 0.001
}
```
*Validation*: `seeds` must be a non-empty list of integers; `epochs` must be integer $1 \le \text{epochs} \le 20$; `batch_size` must be power-of-two between 16 and 512.

### Standard Output JSON Contract (`ml_result_schema.json`):
```json
{
  "schema_version": "1.0.0",
  "run_id": "run-b02135c9-9eb1-4a56-8344-934c9f13e710",
  "scenario_id": "S1_S2_COMBINED",
  "mode": "replay",
  "status": "SUCCESS",
  "timestamp": "2026-10-12T16:50:00Z",
  "execution_time_seconds": 0.0,
  "seeds": [42, 123, 2024],
  "protocol_ref": {
    "protocol_id": "POISONTRACE_PROTOCOL_V1",
    "protocol_version": "1.0.0",
    "protocol_hash": "133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9"
  },
  "baseline": {
    "clean_accuracy": { "mean": 0.9047, "std": 0.0016 },
    "asr": { "mean": 0.9940, "std": 0.0015 }
  },
  "contributors": {
    "C1": { "contributor_id": "C1", "delta_asr": { "mean": -0.0013, "std": 0.0030 }, "clean_accuracy_drop": { "mean": 0.0106, "std": 0.0036 }, "verdict": "CLEARED" },
    "C2": { "contributor_id": "C2", "delta_asr": { "mean": 0.0020, "std": 0.0020 }, "clean_accuracy_drop": { "mean": 0.0106, "std": 0.0021 }, "verdict": "CLEARED" },
    "C3": { "contributor_id": "C3", "delta_asr": { "mean": 0.9868, "std": 0.0010 }, "clean_accuracy_drop": { "mean": 0.0086, "std": 0.0018 }, "verdict": "CAUSALLY IMPLICATED" },
    "C4": { "contributor_id": "C4", "delta_asr": { "mean": -0.0003, "std": 0.0016 }, "clean_accuracy_drop": { "mean": 0.0110, "std": 0.0036 }, "verdict": "CLEARED" },
    "C5": { "contributor_id": "C5", "delta_asr": { "mean": 0.0059, "std": 0.0037 }, "clean_accuracy_drop": { "mean": 0.0080, "std": 0.0042 }, "verdict": "CLEARED" }
  },
  "artifact_hashes": {
    "C1_clean_shard": "8f8586faa76f86c44e43af70f6bea69b9f06acc39e5d60ac2ad91a38bbe21798",
    "C3_poisoned_shard": "2a87c2f38fc3f235e3ba7913d46906b3d171cb005dd2549ff243c96fa1835ca4",
    "C5_noisy_decoy_shard": "b6b1789a909c970e31f0c81564eaf05ea07cc7617675267dcd42f30f2ab86631"
  },
  "limitations": [
    "Evaluated specifically on synthetic Fashion-MNIST with fixed 3x3 patch trigger in bottom-right corner.",
    "Targeted ASR evaluates response to a known trigger; does not prove absence of arbitrary backdoors."
  ],
  "error_message": null
}
```

---

## 5. Metric Definitions, Units & Frozen Verdict Thresholds

### 1. Metric Definitions:
- **Clean Accuracy**: Proportion of correct predictions over the 10,000 standard Fashion-MNIST test images.
  $$\text{Clean Accuracy} = \frac{\sum_{i=1}^{10000} \mathbb{I}(\hat{y}_i = y_i)}{10000} \quad \in [0.0, 1.0]$$
- **Attack Success Rate (ASR)**: Proportion of non-target class test samples (9,000 samples where $y_i \neq 0$) that predict target class 0 after applying the bottom-right 3×3 patch trigger.
  $$\text{ASR} = \frac{\sum_{i: y_i \neq 0} \mathbb{I}(\hat{y}(\tilde{x}_i) = 0)}{9000} \quad \in [0.0, 1.0]$$
- **$\Delta\text{ASR}$ (Delta ASR)**: Drop in ASR when a contributor (or group) is omitted during retraining.
  $$\Delta\text{ASR} = \text{ASR}_{\text{all}} - \text{ASR}_{\text{without}}$$
- **Clean Accuracy Drop**: Drop in clean test accuracy when a contributor (or group) is omitted.
  $$\Delta\text{Acc} = \text{CleanAcc}_{\text{all}} - \text{CleanAcc}_{\text{without}}$$

### 2. Frozen Individual Verdict Thresholds (`MultiSeedVerdictEngine`):
- **Causal Threshold Multiplier**: $0.50 \times \overline{\text{ASR}}_{\text{baseline}}$ (e.g. $0.50 \times 0.9940 = 0.4970$).
- **Cleared Threshold Multiplier**: $0.10 \times \overline{\text{ASR}}_{\text{baseline}}$ (e.g. $0.10 \times 0.9940 = 0.0994$).
- **Statistical Stability Threshold**: $2.0 \times \sigma(\Delta\text{ASR})$.
- **Clean Accuracy Guard**: $\Delta\text{Acc} \le 0.02$ (maximum permissible clean degradation is 2.0%).

$$\text{Verdict}(C) = \begin{cases} 
\textbf{CAUSALLY IMPLICATED} & \text{if } \overline{\Delta\text{ASR}} \ge \text{CausalThresh} \land \overline{\Delta\text{ASR}} > 2\sigma \land \overline{\Delta\text{Acc}} \le 0.02 \\
\textbf{CLEARED} & \text{if } \overline{\Delta\text{ASR}} < \text{ClearedThresh} \\
\textbf{INCONCLUSIVE} & \text{otherwise}
\end{cases}$$

### 3. Frozen Collusion Verdict Thresholds:
For group $G = \{C2, C3\}$:
- Individual removal preservation: $\overline{\text{ASR}}_{\setminus C2} > 0.90$ AND $\overline{\text{ASR}}_{\setminus C3} > 0.90$.
- Joint removal elimination: $\overline{\text{ASR}}_{\setminus \{C2, C3\}} < 0.10$.
- **Verdict**:
  - $\textbf{COLLUSIVE GROUP IMPLICATED}$ if all three conditions are satisfied.
  - $\textbf{COLLUSION NOT CONFIRMED}$ otherwise.
- *Caveat for Priyanshu*: The collusion rule does **not** enforce a clean accuracy guard. The recorded clean drop without C2+C3 is $0.0215 \pm 0.0084$ (exceeds the 0.02 individual guard). This must be surfaced in audit reports as a documented characteristic.

---

## 6. Priyanshu Integration Checklist & Evidence Ledger Mapping

### 1. Database Persistence:
In the backend runs database table (`runs`), Priyanshu should persist:
- `run_id` (Primary Key, UUID string)
- `scenario_id` (String, enum-validated)
- `mode` (`replay` | `live`)
- `status` (`SUCCESS` | `FAILED`)
- `protocol_version` (`1.0.0`)
- `protocol_hash` (`133f229e...`)
- `baseline_clean_accuracy_mean` (Float)
- `baseline_asr_mean` (Float)
- `execution_time_seconds` (Float)
- `created_at` (DateTime ISO-8601)
- `raw_result_json` (JSON / Text blob storing full compliant payload)

### 2. Evidence Records for Ledger & Signing:
Priyanshu's backend generates canonical signed evidence records. Map ML fields as follows:

- **`TRAINING_RUN` Record**:
  - `model_architecture`: `"BaselineCNN"`
  - `parameter_count`: `1701130` (Integer)
  - `dataset_name`: `"Fashion-MNIST"`
  - `total_samples`: `20000` (Integer)
  - `epochs`: `8` (Integer)
  - `seeds`: `[42, 123, 2024]`
  - `artifact_hashes`: Map of dataset shards and baseline checkpoint SHA-256.

- **`EVALUATION` Record**:
  - `test_samples_count`: `10000` (Integer)
  - `eligible_asr_samples_count`: `9000` (Integer)
  - `clean_accuracy_mean`: Float or canonical scaled integer (see below).
  - `asr_mean`: Float or canonical scaled integer.

- **`FINDING` Record**:
  - For each contributor $C_i$:
    - `contributor_id`: e.g. `"C3"`
    - `verdict`: `"CAUSALLY IMPLICATED"` | `"CLEARED"` | `"INCONCLUSIVE"`
    - `delta_asr_mean`: Float or scaled integer
    - `delta_asr_std`: Float or scaled integer
    - `clean_accuracy_drop_mean`: Float or scaled integer

### 3. Canonical Serialization & Integer Scaling Rules:
> [!IMPORTANT]
> Floating-point representations can vary across operating systems, CPU architectures, and JSON serializers.
> If the project signing protocol mandates integer fixed-point serialization for canonical signing:
> - Scale floating-point ratios ($[0.0, 1.0]$) by $10^6$ (`round(value * 1_000_000)`).
> - Example: ASR `0.994037` $\rightarrow$ `994037`.
> - Example: $\Delta\text{ASR}$ `0.986815` $\rightarrow$ `986815`.
> Priyanshu's backend is responsible for applying this conversion immediately before canonical signing.

### 4. Code Ownership Boundaries:
- **Files Owned by ML Subsystem (Do NOT edit without coordination)**:
  - `ml/src/**` (All model, attack, dataset, and training code)
  - `ml/replay/protocol.json` (Frozen protocol)
  - `ml/replay/*_fixture.json` (Frozen replay fixtures)
  - `ml/schema/ml_result_schema.json` (Contract schema)
- **Shared Integration Boundary**:
  - `backend/app/services/ml_runner.py` (Maintained by Saptarshi to match ML; imported by Priyanshu into FastAPI run router).

---

## 7. Error Handling & Timeout Behavior

- **Unknown Scenario**: Adapter returns `status: "FAILED"`, `error_message: "Unsupported scenario_id ..."` (or raises `MLScenarioNotSupportedError` if `raise_on_failure=True`).
- **Missing Data or PyTorch**: In live mode, if PyTorch is not installed or shard `.npz` files are absent, returns clean `status: "FAILED"` with descriptive error guidance ("Run in replay mode or install dependencies").
- **Timeout**: For live executions, if a run exceeds `timeout_seconds` (default 600s), process terminates cleanly and returns `status: "FAILED"`, `error_message: "Execution timed out after 600s"`.
- **Integrity**: A failed run **never** masquerades as a success.
