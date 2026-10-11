# PoisonTrace — Forensic Attribution Report

## Experiment

Scenario:
- S1: Backdoor attack by C3
- S2: Honest-but-noisy decoy by C5

Three random seeds were evaluated:
- 42
- 123
- 2024

## Baseline

Clean accuracy:

90.47% ± 0.19%

Attack Success Rate:

99.40% ± 0.18%

## Contributor Attribution

| Contributor | ΔASR | Verdict |
|---|---:|---|
| C1 | -0.13% | CLEARED |
| C2 | +0.20% | CLEARED |
| C3 | +98.68% | CAUSALLY IMPLICATED |
| C4 | -0.03% | CLEARED |
| C5 | +0.59% | CLEARED |

## Primary Finding

C3 is causally implicated.

Removing C3 reduced ASR from approximately 99.40%
to approximately 0.72%.

This produces:

ΔASR = 98.68 percentage points.

The result substantially exceeds the predefined
causal attribution threshold.

## Decoy Finding

C5 was intentionally constructed as an
honest-but-noisy decoy.

Despite producing a measurable change in ASR,
its ΔASR is far below the causal threshold.

Therefore C5 is cleared.

## Conclusion

PoisonTrace successfully distinguishes:

1. Contributors associated with suspicious data.
2. Contributors that are causally responsible for the
   observed backdoor behavior.

Final verdict:

C3 → CAUSALLY IMPLICATED

C1 → CLEARED
C2 → CLEARED
C4 → CLEARED
C5 → CLEARED