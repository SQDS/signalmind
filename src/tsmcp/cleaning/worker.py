"""Credential-free recipe worker; its only inputs are observations and parent candidates."""

import importlib.util
import random
import resource
import sys
from pathlib import Path

sys.path.insert(0, "/opt/signalmind/src")
from tsmcp.cleaning.common import read, write
from tsmcp.cleaning.engine import process_one


def main():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    import numpy as np

    random.seed(111101)
    np.random.seed(111101)
    job = read("/output/job.json")
    spec = importlib.util.spec_from_file_location("model_recipe", "/output/program.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results, errors = {}, []
    for case in job["cases"]:
        try:
            results[case["id"]] = process_one(
                Path("/output"), case, job["recipe"], job["implementation"], job["parents"].get(case["id"]), module
            )
        except Exception as exc:
            errors.append(
                {
                    "case_id": case["id"],
                    "task": case["task"],
                    "reason": f"{type(exc).__name__}: {exc}",
                    "status": "execution_failed",
                }
            )
    write("/output/result.json", {"results": results, "errors": errors})


if __name__ == "__main__":
    main()
