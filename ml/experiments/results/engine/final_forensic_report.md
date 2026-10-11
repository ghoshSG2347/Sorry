# PoisonTrace Forensic Report

## Experiment

Multi-seed leave-one-contributor-out causal attribution.

Seeds: [42, 123, 2024]

## Baseline

- ASR: 0.9940 +/- 0.0015
- Clean accuracy: 0.9047 +/- 0.0016

## Contributor Verdicts

| Contributor | ΔASR | ΔASR Std | Clean Drop | Verdict |
|---|---:|---:|---:|---|
| C1 | -0.001259 | 0.003039 | 0.010600 | **CLEARED** |
| C2 | 0.002000 | 0.001996 | 0.010600 | **CLEARED** |
| C3 | 0.986815 | 0.000962 | 0.008633 | **CAUSALLY IMPLICATED** |
| C4 | -0.000296 | 0.001588 | 0.011033 | **CLEARED** |
| C5 | 0.005926 | 0.003738 | 0.007967 | **CLEARED** |

## Interpretation

- Causally implicated: C3
- Cleared: C1, C2, C4, C5
- Inconclusive: None

## Decision Rule

A contributor is considered causally implicated only when:

1. Mean ΔASR is at least 50% of baseline ASR.
2. Mean ΔASR exceeds twice its multi-seed standard deviation.
3. Mean clean-accuracy degradation does not exceed 2 percentage points.

A contributor is cleared when its mean ΔASR is below 10% of baseline ASR.

## Conclusion

The experiment identifies C3 as causally responsible for the observed backdoor behavior.
