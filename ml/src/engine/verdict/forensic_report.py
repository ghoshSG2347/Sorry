import json
from pathlib import Path


INPUT_PATH = Path(
    "experiments/results/engine/"
    "multi_seed_forensic_verdict.json"
)

OUTPUT_PATH = Path(
    "experiments/results/engine/"
    "final_forensic_report.md"
)


def build_report():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    baseline = data["baseline"]

    lines = []

    lines.append(
        "# PoisonTrace Forensic Report"
    )

    lines.append("")

    lines.append(
        "## Experiment"
    )

    lines.append("")

    lines.append(
        "Multi-seed leave-one-contributor-out "
        "causal attribution."
    )

    lines.append("")

    lines.append(
        f"Seeds: {data['seeds']}"
    )

    lines.append("")

    lines.append(
        "## Baseline"
    )

    lines.append("")

    lines.append(
        f"- ASR: "
        f"{baseline['asr']['mean']:.4f} "
        f"+/- "
        f"{baseline['asr']['std']:.4f}"
    )

    lines.append(
        f"- Clean accuracy: "
        f"{baseline['clean_accuracy']['mean']:.4f} "
        f"+/- "
        f"{baseline['clean_accuracy']['std']:.4f}"
    )

    lines.append("")

    lines.append(
        "## Contributor Verdicts"
    )

    lines.append("")

    lines.append(
        "| Contributor | ΔASR | "
        "ΔASR Std | Clean Drop | Verdict |"
    )

    lines.append(
        "|---|---:|---:|---:|---|"
    )

    for contributor, result in (
        data["contributors"].items()
    ):

        delta = result["delta_asr"]

        clean_drop = (
            result[
                "clean_accuracy_drop"
            ]
        )

        verdict = result[
            "verdict"
        ]["verdict"]

        lines.append(
            f"| {contributor} | "
            f"{delta['mean']:.6f} | "
            f"{delta['std']:.6f} | "
            f"{clean_drop['mean']:.6f} | "
            f"**{verdict}** |"
        )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    implicated = [
        contributor
        for contributor, result
        in data["contributors"].items()
        if result["verdict"][
            "verdict"
        ]
        == "CAUSALLY IMPLICATED"
    ]

    cleared = [
        contributor
        for contributor, result
        in data["contributors"].items()
        if result["verdict"][
            "verdict"
        ]
        == "CLEARED"
    ]

    inconclusive = [
        contributor
        for contributor, result
        in data["contributors"].items()
        if result["verdict"][
            "verdict"
        ]
        == "INCONCLUSIVE"
    ]

    lines.append(
        f"- Causally implicated: "
        f"{', '.join(implicated) or 'None'}"
    )

    lines.append(
        f"- Cleared: "
        f"{', '.join(cleared) or 'None'}"
    )

    lines.append(
        f"- Inconclusive: "
        f"{', '.join(inconclusive) or 'None'}"
    )

    lines.append("")

    lines.append(
        "## Decision Rule"
    )

    lines.append("")

    lines.append(
        "A contributor is considered causally "
        "implicated only when:"
    )

    lines.append("")

    lines.append(
        "1. Mean ΔASR is at least 50% of "
        "baseline ASR."
    )

    lines.append(
        "2. Mean ΔASR exceeds twice its "
        "multi-seed standard deviation."
    )

    lines.append(
        "3. Mean clean-accuracy degradation "
        "does not exceed 2 percentage points."
    )

    lines.append("")

    lines.append(
        "A contributor is cleared when its "
        "mean ΔASR is below 10% of baseline ASR."
    )

    lines.append("")

    lines.append(
        "## Conclusion"
    )

    lines.append("")

    if implicated:

        lines.append(
            f"The experiment identifies "
            f"{', '.join(implicated)} as "
            "causally responsible for the "
            "observed backdoor behavior."
        )

    else:

        lines.append(
            "No contributor satisfied the "
            "causal attribution criteria."
        )

    lines.append("")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )

    print(
        f"Saved forensic report to:\n"
        f"{OUTPUT_PATH}"
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    build_report()