import json
from pathlib import Path
from datetime import datetime

from src.utils.config import RESULTS_DIR


ENGINE_RESULTS_DIR = RESULTS_DIR / "engine"


def save_json(data, filename):
    """
    Save experiment results as a JSON file.
    """
    ENGINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ENGINE_RESULTS_DIR / filename

    payload = {
        "created_at": datetime.now().isoformat(),
        "data": data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to: {output_path}")

    return output_path


def load_json(filename):
    """
    Load a previously saved engine result.
    """
    input_path = ENGINE_RESULTS_DIR / filename

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    test_data = {
        "experiment": "engine_test",
        "status": "success",
    }

    save_json(test_data, "engine_test.json")