# runner.py
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, List, Dict

from .experiment import Experiment
from .utils.metadata import get_metadata, get_git_diff


class Runner:
    """Runs multiple experiments."""

    def __init__(self, suite_name: str, base_dir: str = "results", suite_id: str | Path | None = None):
        self.suite_name = suite_name
        self.base_dir = Path(base_dir)
        
        meta = get_metadata()
        timestamp = meta["timestamp"]
        short_hash = meta["git_head"][:7] if meta["git_head"] != "untracked" else "untracked"

        # if a suite id is not provided, create a new suite
        if suite_id is None:
            self.suite_id = f"{timestamp}_{short_hash}"
            self.suite_dir = self.base_dir / self.suite_name / self.suite_id
        else:
            # otherwise, resume the progress
            suite_path = Path(suite_id)
            if suite_path.is_absolute() or "/" in str(suite_id) or suite_path.exists():
                self.suite_dir = suite_path
            else:
                self.suite_dir = self.base_dir / self.suite_name / str(suite_id)
            self.suite_id = self.suite_dir.name

    def _hash_params(self, params: Dict) -> str:
        """Create a hash for a parameter dictionary."""
        encoded = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.md5(encoded).hexdigest()[:8]

    def _is_run_completed(self, run_id: str) -> bool:
        """Check if a run exists and finished successfully by existence of metrics.json file."""
        metrics_file = self.suite_dir / run_id / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("status") == "SUCCESS"
            except json.JSONDecodeError:
                return False
        return False

    def run_grid(self, func: Callable, param_grid: List[Dict], stop_on_error: bool = False) -> List[Dict]:
        """Execute a batch of configurations sequentially."""
        self.suite_dir.mkdir(parents=True, exist_ok=True)
        
        # extract metadata once for the entire batch
        meta = get_metadata()
        timestamp = meta["timestamp"]
        
        # save uncommited changes only once for all experiments and replace existing patch in resume case
        existing_patches = sorted(self.suite_dir.glob("suite_uncommitted_*.patch"), reverse=True)
        if existing_patches:
            patch_file = existing_patches[0]
            timestamp = patch_file.name.replace("suite_uncommitted_", "").replace(".patch", "")
        else:
            timestamp = meta["timestamp"]
            if meta.get("has_uncommitted_changes"):
                diff_text = get_git_diff()
                if diff_text:
                    patch_name = f"suite_uncommitted_{timestamp}.patch"
                    with open(self.suite_dir / patch_name, "w", encoding="utf-8") as f:
                        f.write(diff_text)

        suite_start_time = time.perf_counter()
        start_date_str = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[{start_date_str}] Starting suite: '{self.suite_name}' | Total configs: {len(param_grid)}\n")
        results = []

        # iterate over configurations
        for index, params in enumerate(param_grid):
            param_hash = self._hash_params(params)
            run_id = f"run_{index:03d}_{param_hash}"

            # skip finished configurations in case of crash
            if self._is_run_completed(run_id):
                print(f"[SKIP] {run_id} completed in a previous session.")
                results.append({"run_id": run_id, "status": "SKIPPED"})
                continue

            print(f"[RUN] {run_id} | Params: {params}")

            # run experiments
            exp = Experiment(name=self.suite_name, base_dir=self.base_dir, run_id=f"{self.suite_id}/{run_id}", save_git_patch=False)
            wrapped_func = exp(func)

            exp_start_time = time.perf_counter()
            try:
                wrapped_func(**params)

                run_duration = time.perf_counter() - exp_start_time
                print(f"Finished in {run_duration:.4f}s")

                results.append({"run_id": run_id, "status": "SUCCESS", "duration_s": round(run_duration, 4)})
            except Exception as _:
                print(f"[ERROR] {run_id} failed.")

                run_duration = time.perf_counter() - exp_start_time
                print(f"Failed in {run_duration:.4f}s")

                results.append({"run_id": run_id, "status": "FAILED", "duration_s": round(run_duration, 4)})
                if stop_on_error:
                    print("Aborting suite due to stop_on_error=True.")
                    break

        suite_duration = time.perf_counter() - suite_start_time
        suite_duration_fmt = time.strftime("%H:%M:%S", time.gmtime(suite_duration))

        print(f"\nSuite '{self.suite_name}' finished in {suite_duration_fmt} ({suite_duration:.2f}s).")

        # save summary file about the run
        summary_data = {
            "suite_name": self.suite_name,
            "suite_id": self.suite_id,
            "start_date": start_date_str,
            "total_duration_seconds": round(suite_duration, 4),
            "total_duration_formatted": suite_duration_fmt,
            "total_configs": len(param_grid),
            "runs": results
        }
        
        summary_file = self.suite_dir / f"suite_summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        return results
