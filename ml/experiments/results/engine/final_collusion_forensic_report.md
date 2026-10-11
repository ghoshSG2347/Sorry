# PoisonTrace — S3 Collusion Forensic Report

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
| Baseline | 0.9940 | 0.0014 |
| Without C2 | 0.9900 | 0.0029 |
| Without C3 | 0.9922 | 0.0032 |
| Without C2 + C3 | 0.0141 | 0.0060 |

---

## 4. Attribution Signal

### Removing C2

Mean ΔASR:

**0.0041 ± 0.0027**

The attack remains active after removing C2.

Therefore, C2 cannot be identified as the sole necessary contributor.

### Removing C3

Mean ΔASR:

**0.0018 ± 0.0035**

The attack remains active after removing C3.

Therefore, C3 cannot be identified as the sole necessary contributor.

### Removing C2 + C3

Mean ΔASR:

**0.9799 ± 0.0060**

The ASR collapses from approximately
0.9940 to 0.0141.

This demonstrates that the backdoor signal depends on the joint presence
of C2 and C3 under this experimental protocol.

---

## 5. Collusion Criteria

The following conditions were evaluated:

- Removing C2 individually preserves a high-ASR attack.
- Removing C3 individually preserves a high-ASR attack.
- Removing C2 and C3 jointly removes the attack.

Results:

- C2 alone preserves attack: **True**
- C3 alone preserves attack: **True**
- Joint removal removes attack: **True**

---

## 6. Verdict

# COLLUSIVE GROUP IMPLICATED

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
