from src.engine.report import build_forensic_report


def run_pipeline():
    print("\n" + "=" * 60)
    print("POISONTRACE FORENSIC PIPELINE")
    print("=" * 60)

    report = build_forensic_report()

    print("\nPipeline completed successfully.")

    print("\nIndividual attribution:")
    for contributor_id, result in report[
        "individual_attribution"
    ]["results"].items():
        print(
            f"  {contributor_id}: "
            f"{result['verdict']}"
        )

    group = report["group_attribution"]

    print("\nGroup attribution:")
    print(f"  Group: {group['group']}")
    print(
        f"  Verdict: "
        f"{group['verdict']['verdict']}"
    )

    print("\n" + "=" * 60)

    return report


if __name__ == "__main__":
    run_pipeline()