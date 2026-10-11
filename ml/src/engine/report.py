from src.engine.results import load_json, save_json
from src.engine.verdict import GroupVerdictEngine


def build_forensic_report():
    """
    Combine individual attribution and group attribution
    into one unified PoisonTrace forensic report.
    """

    loo = load_json("leave_one_out_engine.json")
    group = load_json("group_attribution_engine.json")

    loo_data = loo["data"]
    group_data = group["data"]

    group_result = group_data["result"]

    group_verdict_engine = GroupVerdictEngine()

    group_verdict = group_verdict_engine.classify(
        asr_all=group_result["asr_all"],
        asr_without_group=group_result["asr_without_group"],
    )

    report = {
        "project": "PoisonTrace",

        "individual_attribution": {
            "experiment": loo_data["experiment"],
            "seed": loo_data["seed"],
            "contributors": loo_data["contributors"],
            "baseline": loo_data["baseline"],
            "results": loo_data["results"],
        },

        "group_attribution": {
            "experiment": group_data["experiment"],
            "group": group_data["group"],
            "result": group_result,
            "verdict": group_verdict,
        },
    }

    save_json(
        report,
        "forensic_report.json",
    )

    return report


def print_forensic_report(report):
    print("\n" + "=" * 60)
    print("POISONTRACE FORENSIC REPORT")
    print("=" * 60)

    print("\nINDIVIDUAL ATTRIBUTION")
    print("-" * 60)

    results = report["individual_attribution"]["results"]

    for contributor_id, result in results.items():
        print(
            f"{contributor_id}: "
            f"ΔASR={result['delta_asr']:.4f} | "
            f"Verdict={result['verdict']}"
        )

    group = report["group_attribution"]

    print("\nGROUP ATTRIBUTION")
    print("-" * 60)

    print(f"Group: {group['group']}")
    print(
        f"ΔASR: "
        f"{group['verdict']['delta_asr']:.4f}"
    )
    print(
        f"Verdict: "
        f"{group['verdict']['verdict']}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    report = build_forensic_report()
    print_forensic_report(report)