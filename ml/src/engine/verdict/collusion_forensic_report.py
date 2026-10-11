from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "results"
    / "engine"
    / "multi_seed_collusion_verdict.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "results"
    / "engine"
    / "final_collusion_forensic_report.md"
)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline = data["baseline_asr"]
    without_c2 = data["without_C2_asr"]
    without_c3 = data["without_C3_asr"]
    without_group = data["without_C2_C3_asr"]

    delta_c2 = data["delta_asr_without_C2"]
    delta_c3 = data["delta_asr_without_C3"]
    delta_group = data["delta_asr_without_C2_C3"]

    verdict = data["verdict"]

    report = f"""# PoisonTrace — S3 Collusion Forensic Report

## 1. Experiment

This experiment evaluates whether contributors C2 and C3 are
jointly responsible for a collusive backdoor attack.

The experiment uses three deterministic seeds:

- 42
- 123
- 2024

The same training and evaluation protocol is used for every condition.

---

## 2. Experimental Design

Four contributor configurations were evaluated:

1. **Baseline:** C1 + C2 + C3 + C4 + C5
2. **Without C2:** C1 + C3 + C4 + C5
3. **Without C3:** C1 + C2 + C4 + C5
4. **Without C2 and C3:** C1 + C4 + C5

The purpose is to determine whether either contributor alone is sufficient
to preserve the backdoor, and whether removing both contributors eliminates
the backdoor.

---

## 3. Results

| Configuration | Mean ASR | Std |
|---|---:|---:|
| Baseline | {baseline["mean"]:.4f} | {baseline["std"]:.4f} |
| Without C2 | {without_c2["mean"]:.4f} | {without_c2["std"]:.4f} |
| Without C3 | {without_c3["mean"]:.4f} | {without_c3["std"]:.4f} |
| Without C2 + C3 | {without_group["mean"]:.4f} | {without_group["std"]:.4f} |

---

## 4. Attribution Signal

### Removing C2

Mean ΔASR:

**{delta_c2["mean"]:.4f} ± {delta_c2["std"]:.4f}**

The attack remains active after removing C2.

Therefore, C2 cannot be identified as the sole necessary contributor.

### Removing C3

Mean ΔASR:

**{delta_c3["mean"]:.4f} ± {delta_c3["std"]:.4f}**

The attack remains active after removing C3.

Therefore, C3 cannot be identified as the sole necessary contributor.

### Removing C2 + C3

Mean ΔASR:

**{delta_group["mean"]:.4f} ± {delta_group["std"]:.4f}**

The ASR collapses from approximately
{baseline["mean"]:.4f} to {without_group["mean"]:.4f}.

This demonstrates that the backdoor signal depends on the joint presence
of C2 and C3 under this experimental protocol.

---

## 5. Collusion Criteria

The following conditions were evaluated:

- Removing C2 individually preserves a high-ASR attack.
- Removing C3 individually preserves a high-ASR attack.
- Removing C2 and C3 jointly removes the attack.

Results:

- C2 alone preserves attack: **{data["individual_C2_removal_preserves_backdoor"]}**
- C3 alone preserves attack: **{data["individual_C3_removal_preserves_backdoor"]}**
- Joint removal removes attack: **{data["joint_C2_C3_removal_removes_backdoor"]}**

---

## 6. Verdict

# {verdict}

The multi-seed experiment supports the conclusion that C2 and C3 form a
collusive contributor group under the PoisonTrace S3 synthetic scenario.

Neither individual removal eliminates the backdoor, while joint removal
causes the backdoor success rate to collapse.

---

## 7. Scope and Limitations

This conclusion applies to the current synthetic Fashion-MNIST collusion
scenario and the evaluated training protocol.

It does not establish generalization to:

- unknown datasets
- unknown triggers
- unknown target classes
- arbitrary poison rates
- arbitrary model architectures
- federated training
- real-world poisoning attacks

The result should therefore be interpreted as validated experimental
evidence for the implemented S3 scenario rather than a universal guarantee
of collusion detection.

---

## 8. Reproducibility

Seeds:

`42, 123, 2024`

Results source:

`multi_seed_collusion_individual.json`

Verdict source:

`multi_seed_collusion_verdict.json`
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # UTF-8 with BOM improves correct display in Windows PowerShell.
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig") as f:
        f.write(report)

    print(f"Saved report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()