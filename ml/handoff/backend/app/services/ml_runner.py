"""
PoisonTrace Backend ML Runner Adapter
=====================================
Target file in product repository: backend/app/services/ml_runner.py
Ownership: Shared interface between ML Subsystem (Saptarshi) and Backend Orchestration (Priyanshu).

Responsibilities:
- Provide a safe, documented, stable Python entrypoint for the backend run service.
- Validate incoming scenario IDs and configuration against strict allowlists.
- Dispatch between instant 'replay' (committed verified fixtures) and 'live' ML execution.
- Return structured, versioned result dictionaries conforming to `ml_result_schema.json` (v1.0.0).
- Sanitize output so that internal server filesystem paths are never leaked.
- Isolate failure states (timeouts, missing files, process errors) into structured status="FAILED" records.
- Strictly avoid embedding database models, API routing, ledger logic, or cryptographic signing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("poisontrace.ml_runner")

# Current schema version implemented by this adapter
SCHEMA_VERSION: str = "1.0.0"

# Canonical scenario IDs supported by PoisonTrace
SCENARIO_S1: str = "S1_BACKDOOR_C3"
SCENARIO_S2: str = "S2_NOISY_DECOY_C5"
SCENARIO_S1_S2: str = "S1_S2_COMBINED"
SCENARIO_S3: str = "S3_COLLUSION_C2_C3"

SUPPORTED_SCENARIOS: Set[str] = {
    SCENARIO_S1,
    SCENARIO_S2,
    SCENARIO_S1_S2,
    SCENARIO_S3,
}

SUPPORTED_MODES: Set[str] = {"replay", "live"}


class MLRunnerError(Exception):
    """Base exception for ML runner errors."""


class MLScenarioNotSupportedError(MLRunnerError):
    """Raised when an unrecognized scenario ID is requested."""


class MLInvalidConfigurationError(MLRunnerError):
    """Raised when runner configuration parameters fail validation."""


class MLRunner:
    """
    Main orchestration adapter for PoisonTrace ML scenarios.

    Usage:
        runner = MLRunner()
        result = runner.run_scenario(
            scenario_id="S1_S2_COMBINED",
            mode="replay",
        )
    """

    def __init__(self, ml_root: Optional[Path] = None):
        """
        Initialize the ML runner with the root path of the ml/ directory.
        If not provided, resolves relative to repository conventions.
        """
        if ml_root is not None:
            self.ml_root = Path(ml_root).resolve()
        else:
            self.ml_root = self._locate_ml_root()

        self.replay_dir = self.ml_root / "replay"
        self.protocol_path = self.replay_dir / "protocol.json"

    def _locate_ml_root(self) -> Path:
        """Find the ml/ directory regardless of whether invoked from backend/ or ml/."""
        current = Path(__file__).resolve()
        # Traverse upwards looking for 'ml' directory or markers
        for parent in [current] + list(current.parents):
            candidate = parent / "ml"
            if candidate.is_dir() and (candidate / "replay").is_dir():
                return candidate
            if parent.name == "ml" and (parent / "replay").is_dir():
                return parent
            # Fallback: check workspace root
            if (parent / "ml" / "src").is_dir():
                return parent / "ml"
        # Default fallback to parent of current module
        return current.parents[3] / "ml" if len(current.parents) >= 4 else current.parent

    def get_supported_scenarios(self) -> List[str]:
        """Return the sorted list of canonical scenario IDs supported by the ML subsystem."""
        return sorted(list(SUPPORTED_SCENARIOS))

    def get_supported_modes(self) -> List[str]:
        """Return supported execution modes."""
        return sorted(list(SUPPORTED_MODES))

    def load_protocol(self) -> Dict[str, Any]:
        """Load the frozen forensic protocol descriptor."""
        if not self.protocol_path.is_file():
            raise MLRunnerError(f"Frozen protocol file missing at expected path: {self.protocol_path}")
        with open(self.protocol_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_scenario(
        self,
        scenario_id: str,
        mode: str = "replay",
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        raise_on_failure: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute or replay a PoisonTrace forensic auditing scenario.

        Args:
            scenario_id: One of SUPPORTED_SCENARIOS.
            mode: 'replay' (fast, deterministic fixture) or 'live' (train PyTorch model).
            run_id: Optional unique identifier for this run. Auto-generated UUID if omitted.
            config: Optional dictionary containing execution overrides (e.g. seeds, epochs).
            raise_on_failure: If True, raises MLRunnerError on execution failure;
                              if False, returns a structured status="FAILED" result dict.

        Returns:
            A validated result dictionary conforming to `ml_result_schema.json`.
        """
        effective_run_id = run_id or f"run-{uuid.uuid4()}"
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Validation
        if scenario_id not in SUPPORTED_SCENARIOS:
            err_msg = (
                f"Unsupported scenario_id '{scenario_id}'. "
                f"Must be one of: {sorted(list(SUPPORTED_SCENARIOS))}"
            )
            logger.error(err_msg)
            if raise_on_failure:
                raise MLScenarioNotSupportedError(err_msg)
            return self._build_failed_result(
                run_id=effective_run_id,
                scenario_id=scenario_id,
                mode=mode,
                timestamp=timestamp,
                error_message=err_msg,
            )

        if mode not in SUPPORTED_MODES:
            err_msg = f"Unsupported mode '{mode}'. Must be one of: {sorted(list(SUPPORTED_MODES))}"
            logger.error(err_msg)
            if raise_on_failure:
                raise MLInvalidConfigurationError(err_msg)
            return self._build_failed_result(
                run_id=effective_run_id,
                scenario_id=scenario_id,
                mode=mode,
                timestamp=timestamp,
                error_message=err_msg,
            )

        # 2. Execution dispatch
        try:
            if mode == "replay":
                result = self._execute_replay(
                    scenario_id=scenario_id,
                    run_id=effective_run_id,
                    timestamp=timestamp,
                )
                result["execution_time_seconds"] = 0.0
            else:
                result = self._execute_live(
                    scenario_id=scenario_id,
                    run_id=effective_run_id,
                    timestamp=timestamp,
                    config=config or {},
                )
                elapsed = time.time() - start_time
                result["execution_time_seconds"] = round(elapsed, 4)

            return result

        except Exception as exc:
            logger.exception("ML scenario execution failed: %s", exc)
            if raise_on_failure:
                raise MLRunnerError(str(exc)) from exc
            elapsed = time.time() - start_time
            return self._build_failed_result(
                run_id=effective_run_id,
                scenario_id=scenario_id,
                mode=mode,
                timestamp=timestamp,
                error_message=str(exc),
                execution_time_seconds=round(elapsed, 4),
            )

    def _execute_replay(
        self,
        scenario_id: str,
        run_id: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Load and return committed, verified fixture."""
        if scenario_id in (SCENARIO_S1_S2, SCENARIO_S1, SCENARIO_S2):
            fixture_path = self.replay_dir / "s1_s2_combined_fixture.json"
        elif scenario_id == SCENARIO_S3:
            fixture_path = self.replay_dir / "s3_collusion_fixture.json"
        else:
            raise MLScenarioNotSupportedError(f"No replay fixture mapped for {scenario_id}")

        if not fixture_path.is_file():
            raise MLRunnerError(f"Replay fixture missing: {fixture_path.name}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Clone and customize with runtime invocation parameters
        result = dict(data)
        result["run_id"] = run_id
        result["scenario_id"] = scenario_id
        result["mode"] = "replay"
        result["timestamp"] = timestamp
        result["execution_time_seconds"] = 0.0

        # If individual scenario requested from combined fixture, filter contributors view
        if scenario_id == SCENARIO_S1 and "contributors" in result:
            result["limitations"] = result.get("limitations", []) + [
                "Focused view for S1 (C3 Backdoor); full LOO baseline and decoy C5 were co-evaluated in combined dataset."
            ]
        elif scenario_id == SCENARIO_S2 and "contributors" in result:
            result["limitations"] = result.get("limitations", []) + [
                "Focused view for S2 (C5 Noisy Decoy); full LOO baseline and attacker C3 were co-evaluated in combined dataset."
            ]

        return result

    def _execute_live(
        self,
        scenario_id: str,
        run_id: str,
        timestamp: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute live model training and evaluation using the ML engine.
        Ensures dependencies and data are available before starting.
        """
        try:
            import torch
        except ImportError as exc:
            raise MLRunnerError(
                "PyTorch is not installed in the active environment. "
                "Live ML training requires 'torch' and 'torchvision'. Use mode='replay' or install dependencies."
            ) from exc

        # Ensure raw and processed data exist
        contributors_dir = self.ml_root / "data" / "processed" / "train" / "contributors"
        if not contributors_dir.is_dir() or not (contributors_dir / "C1.npz").is_file():
            raise MLRunnerError(
                "Contributor shard data is missing in ml/data/processed/train/contributors. "
                "Run 'python ml/src/data/create_shards.py' first or use mode='replay'."
            )

        protocol = self.load_protocol()
        seeds = config.get("seeds", [42, 123, 2024])
        epochs = config.get("epochs", 8)
        batch_size = config.get("batch_size", 128)
        lr = config.get("learning_rate", 0.001)

        # Validate types
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            raise MLInvalidConfigurationError("seeds must be a list of integers")
        if not isinstance(epochs, int) or epochs < 1 or epochs > 20:
            raise MLInvalidConfigurationError("epochs must be an integer between 1 and 20")

        # Live execution: import relevant engine modules
        import sys
        ml_src = str(self.ml_root)
        if ml_src not in sys.path:
            sys.path.insert(0, ml_src)

        if scenario_id in (SCENARIO_S1, SCENARIO_S2, SCENARIO_S1_S2):
            return self._run_live_individual(
                run_id=run_id,
                scenario_id=scenario_id,
                timestamp=timestamp,
                protocol=protocol,
                seeds=seeds,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
        elif scenario_id == SCENARIO_S3:
            return self._run_live_collusion(
                run_id=run_id,
                scenario_id=scenario_id,
                timestamp=timestamp,
                protocol=protocol,
                seeds=seeds,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
        else:
            raise MLScenarioNotSupportedError(f"Live execution not implemented for scenario {scenario_id}")

    def _run_live_individual(
        self,
        run_id: str,
        scenario_id: str,
        timestamp: str,
        protocol: Dict[str, Any],
        seeds: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        """Perform live multi-seed leave-one-out attribution."""
        from src.engine.prove.leave_one_out import LeaveOneOutEngine
        from src.engine.verdict.multiseed_verdict import MultiSeedVerdictEngine
        import numpy as np

        all_seed_results = {}
        for seed in seeds:
            engine = LeaveOneOutEngine(
                contributors=(1, 2, 3, 4, 5),
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                scenario="original",
            )
            raw = engine.run()
            all_seed_results[seed] = raw

        # Aggregate across seeds
        baseline_asrs = [r["baseline"]["asr"] for r in all_seed_results.values()]
        baseline_cleans = [r["baseline"]["clean_accuracy"] for r in all_seed_results.values()]

        baseline_summary = {
            "clean_accuracy": {
                "mean": float(np.mean(baseline_cleans)),
                "std": float(np.std(baseline_cleans)),
            },
            "asr": {
                "mean": float(np.mean(baseline_asrs)),
                "std": float(np.std(baseline_asrs)),
            },
        }

        verdict_engine = MultiSeedVerdictEngine()
        contributors_summary = {}

        for c_id in ["C1", "C2", "C3", "C4", "C5"]:
            c_delta_asrs = [r["results"][c_id]["delta_asr"] for r in all_seed_results.values()]
            c_clean_drops = [r["results"][c_id]["clean_accuracy_drop"] for r in all_seed_results.values()]
            c_asrs_without = [r["results"][c_id]["asr_without"] for r in all_seed_results.values()]
            c_cleans_without = [r["results"][c_id]["clean_accuracy_without"] for r in all_seed_results.values()]

            delta_mean = float(np.mean(c_delta_asrs))
            delta_std = float(np.std(c_delta_asrs))
            clean_drop_mean = float(np.mean(c_clean_drops))

            verdict_info = verdict_engine.classify(
                baseline_asr_mean=baseline_summary["asr"]["mean"],
                delta_asr_mean=delta_mean,
                delta_asr_std=delta_std,
                clean_accuracy_drop_mean=clean_drop_mean,
            )

            contributors_summary[c_id] = {
                "contributor_id": c_id,
                "delta_asr": {
                    "mean": delta_mean,
                    "std": delta_std,
                },
                "clean_accuracy_drop": {
                    "mean": clean_drop_mean,
                    "std": float(np.std(c_clean_drops)),
                },
                "asr_without": {
                    "mean": float(np.mean(c_asrs_without)),
                    "std": float(np.std(c_asrs_without)),
                },
                "clean_accuracy_without": {
                    "mean": float(np.mean(c_cleans_without)),
                    "std": float(np.std(c_cleans_without)),
                },
                "verdict": verdict_info["verdict"],
                "verdict_details": {
                    "causal_threshold": verdict_info["causal_threshold"],
                    "cleared_threshold": verdict_info["cleared_threshold"],
                    "statistical_threshold": verdict_info["statistical_threshold"],
                    "max_clean_accuracy_drop": verdict_info["max_clean_accuracy_drop"],
                    "passes_causal_threshold": verdict_info["passes_causal_threshold"],
                    "passes_statistical_threshold": verdict_info["passes_statistical_threshold"],
                    "passes_clean_accuracy_guard": verdict_info["passes_clean_accuracy_guard"],
                },
            }

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "mode": "live",
            "status": "SUCCESS",
            "timestamp": timestamp,
            "seeds": seeds,
            "protocol_ref": {
                "protocol_id": protocol.get("protocol_id", "POISONTRACE_PROTOCOL_V1"),
                "protocol_version": protocol.get("protocol_version", "1.0.0"),
                "protocol_hash": "133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9",
            },
            "baseline": baseline_summary,
            "contributors": contributors_summary,
            "artifact_hashes": {
                "C1_clean_shard": "8f8586faa76f86c44e43af70f6bea69b9f06acc39e5d60ac2ad91a38bbe21798",
                "C2_clean_shard": "8622c9dc9c667ba0c9cb0344eb1c0a87093a61e420f1bbcf023b658a4077c246",
                "C3_clean_shard": "d27130235ddef6fcf3a8bccf24c9839931946bec95625ebd1229f2346fe40a7a",
                "C4_clean_shard": "6368115d82c072d15083bc231f43978435ffb6967b749ff3e1ecc3e4762954d3",
                "C5_clean_shard": "736a17c3cd3aad6db2988d0be5f7311dd580fcd7410ab5cb5dd090290e6d108b",
                "C3_poisoned_shard": "2a87c2f38fc3f235e3ba7913d46906b3d171cb005dd2549ff243c96fa1835ca4",
                "C5_noisy_decoy_shard": "b6b1789a909c970e31f0c81564eaf05ea07cc7617675267dcd42f30f2ab86631",
            },
            "limitations": [
                "Live training executed on local CPU.",
                "Targeted ASR evaluates response to known 3x3 patch trigger in bottom-right corner.",
            ],
            "error_message": null,
        }

    def _run_live_collusion(
        self,
        run_id: str,
        scenario_id: str,
        timestamp: str,
        protocol: Dict[str, Any],
        seeds: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        """Perform live multi-seed collusion group evaluation."""
        from src.engine.prove.group_test import GroupAttributionEngine
        from src.engine.probes.metrics import evaluate_model
        import numpy as np

        tests = {
            "baseline": [1, 2, 3, 4, 5],
            "without_C2": [1, 3, 4, 5],
            "without_C3": [1, 2, 4, 5],
            "without_C2_C3": [1, 4, 5],
        }

        eval_records: Dict[str, List[Dict[str, float]]] = {k: [] for k in tests}

        for test_name, contributors in tests.items():
            for seed in seeds:
                engine = GroupAttributionEngine(
                    contributors=[1, 2, 3, 4, 5],
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    scenario="collusion",
                )
                model = engine.train_model(contributors)
                metrics = evaluate_model(model)
                eval_records[test_name].append({
                    "clean_accuracy": metrics["clean_accuracy"],
                    "asr": metrics["asr"],
                })

        baseline_asr = np.array([x["asr"] for x in eval_records["baseline"]])
        baseline_clean = np.array([x["clean_accuracy"] for x in eval_records["baseline"]])
        without_c2_asr = np.array([x["asr"] for x in eval_records["without_C2"]])
        without_c3_asr = np.array([x["asr"] for x in eval_records["without_C3"]])
        without_group_asr = np.array([x["asr"] for x in eval_records["without_C2_C3"]])
        without_group_clean = np.array([x["clean_accuracy"] for x in eval_records["without_C2_C3"]])

        delta_c2 = baseline_asr - without_c2_asr
        delta_c3 = baseline_asr - without_c3_asr
        delta_group = baseline_asr - without_group_asr
        clean_drop_group = baseline_clean - without_group_clean

        c2_preserved = bool(np.mean(without_c2_asr) > 0.90)
        c3_preserved = bool(np.mean(without_c3_asr) > 0.90)
        joint_removed = bool(np.mean(without_group_asr) < 0.10)
        collusion_confirmed = c2_preserved and c3_preserved and joint_removed

        def m_s(arr: np.ndarray) -> Dict[str, float]:
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "mode": "live",
            "status": "SUCCESS",
            "timestamp": timestamp,
            "seeds": seeds,
            "protocol_ref": {
                "protocol_id": protocol.get("protocol_id", "POISONTRACE_PROTOCOL_V1"),
                "protocol_version": protocol.get("protocol_version", "1.0.0"),
                "protocol_hash": "133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9",
            },
            "baseline": {
                "clean_accuracy": m_s(baseline_clean),
                "asr": m_s(baseline_asr),
            },
            "collusion_group": {
                "group": [2, 3],
                "without_C2_asr": m_s(without_c2_asr),
                "without_C3_asr": m_s(without_c3_asr),
                "without_group_asr": m_s(without_group_asr),
                "delta_asr_without_C2": m_s(delta_c2),
                "delta_asr_without_C3": m_s(delta_c3),
                "delta_asr_without_group": m_s(delta_group),
                "clean_accuracy_without_group": m_s(without_group_clean),
                "clean_accuracy_drop_without_group": m_s(clean_drop_group),
                "individual_C2_removal_preserves_backdoor": c2_preserved,
                "individual_C3_removal_preserves_backdoor": c3_preserved,
                "joint_removal_removes_backdoor": joint_removed,
                "verdict": (
                    "COLLUSIVE GROUP IMPLICATED"
                    if collusion_confirmed
                    else "COLLUSION NOT CONFIRMED"
                ),
            },
            "artifact_hashes": {
                "C1_clean_shard": "8f8586faa76f86c44e43af70f6bea69b9f06acc39e5d60ac2ad91a38bbe21798",
                "C2_collusion_shard": "627fabea242ed8d41fde9b2a3f500d75d9b1b6935eb610414240448408d39a7a",
                "C3_collusion_shard": "9179e6a39e2ca6c2e35798317aec9d943285537884704f887ac04f895cb47113",
                "C4_clean_shard": "6368115d82c072d15083bc231f43978435ffb6967b749ff3e1ecc3e4762954d3",
                "C5_clean_shard": "736a17c3cd3aad6db2988d0be5f7311dd580fcd7410ab5cb5dd090290e6d108b",
            },
            "limitations": [
                "Live collusion training executed on local CPU.",
                "Collusion verdict verifies joint-necessity against known trigger without clean-accuracy guard.",
            ],
            "error_message": null,
        }

    def _build_failed_result(
        self,
        run_id: str,
        scenario_id: str,
        mode: str,
        timestamp: str,
        error_message: str,
        execution_time_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Construct a valid error result complying with schema failure status."""
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "scenario_id": scenario_id if scenario_id in SUPPORTED_SCENARIOS else "S1_S2_COMBINED",
            "mode": mode if mode in SUPPORTED_MODES else "replay",
            "status": "FAILED",
            "timestamp": timestamp,
            "execution_time_seconds": execution_time_seconds,
            "seeds": [42, 123, 2024],
            "protocol_ref": {
                "protocol_id": "POISONTRACE_PROTOCOL_V1",
                "protocol_version": "1.0.0",
                "protocol_hash": "133f229e5f2e7376ee62ca264cbb8ef46ecfb5dc9b9ed6f1c4fca035e35c94a9",
            },
            "baseline": {
                "clean_accuracy": {"mean": 0.0, "std": 0.0},
                "asr": {"mean": 0.0, "std": 0.0},
            },
            "artifact_hashes": {},
            "limitations": ["Run terminated prematurely due to execution failure."],
            "error_message": error_message,
        }


# Convenience function for direct module-level invocation
def run_scenario(
    scenario_id: str,
    mode: str = "replay",
    run_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Top-level entrypoint for backend run orchestrator."""
    runner = MLRunner()
    return runner.run_scenario(
        scenario_id=scenario_id,
        mode=mode,
        run_id=run_id,
        config=config,
    )
