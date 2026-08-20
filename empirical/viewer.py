import numpy as np
import pandas as pd

from datetime import datetime
import json
from pathlib import Path
import pickle
import subprocess
import shutil


class ExperimentViewer:
    """Experiment loader."""

    def __init__(self, experiment_path: str | Path, patch_path: Path | None = None):
        self.path = Path(experiment_path)

        # if just the experiment name is given, select the latest run
        if not (self.path / "params.json").exists():
            runs = sorted([p for p in self.path.iterdir() if p.is_dir()], reverse=True)
            if runs:
                self.path = runs[0]
            else:
                raise FileNotFoundError(f"No runs found in: {self.path}")

        self.params = self._load_json("params.json")
        self.metrics = self._load_json("metrics.json")
        self._history = self._load_json("history.json")

        self.patch_path = patch_path or (self.path / "uncommitted_changes.patch")

    def _load_json(self, filename: str):
        filepath = self.path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @property
    def history(self) -> pd.DataFrame:
        """Return iteration history as a dataframe."""
        if self._history:
            return pd.DataFrame(self._history)

        return pd.DataFrame()

    @property
    def result(self):
        """Return tje result of the experiment."""
        npy_path = self.path / "result.npy"
        pkl_path = self.path / "result.pkl"
        txt_path = self.path / "result.txt"

        if npy_path.exists():
            return np.load(npy_path, allow_pickle=True)
        elif pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        elif txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
        return None

    def summary(self):
        """Show a clear summary."""
        meta = self.params.get("metadata", {})
        params = self.params.get("parameters", {})

        raw_start = meta.get("timestamp", "")
        try:
            start_date = datetime.strptime(raw_start, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            start_date = raw_start or "N/A"

        end_date = self.metrics.get("end_timestamp", "N/A")
        status = self.metrics.get("status", "N/A")
        duration = self.metrics.get("duration_formatted", "N/A")
        duration_sec = self.metrics.get("duration_seconds", 0)
        peak_ram = self.metrics.get("peak_ram_mb", "N/A")
        git_head = meta.get("git_head", "N/A")
        has_diff = meta.get("has_uncommitted_changes", False)

        print(f"\n\n ---------------------------- EXPERIMENT: {self.path.parent.name} / {self.path.name} -----------------------------------------")
        print(f" -> Start date: {start_date} - End date: {end_date}  |  Status: {status}")
        print(f" -> Time: {duration} ({duration_sec}s)")
        print(f" -> Peak RAM Usage: {peak_ram} MB")
        print(f" -> Git Commit Hash: {git_head} (Uncommitted changes: {has_diff})")

        print(f"\nParameters:")
        for k, v in params.get("kwargs", {}).items():
            print(f" -> {k} = {v}")
        if params.get("args"):
            print(f" -> args = {params.get('args')}")

    def show_log(self, lines: int = 30):
        """Show the last N lines of log file."""
        log_file = self.path / "output.log"
        if not log_file.exists():
            print("File output.log not foundd.")
            return

        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            print("".join(all_lines[-lines:]))
                
    def restore_code_state(self, force: bool = False, destination_path: Path | None = None) -> None:
        """Checkout the exact git commit and apply uncommitted patch changes to restore the exact code state of the experiment in a new folder."""
        git_head = self.params.get("metadata", {}).get("git_head")

        if not git_head or git_head == "untracked":
            print("No Git commit associated with this run.")
            return

        restore_dir = destination_path or (self.path / "restored_code")
        
        if restore_dir.exists():
            if not force:
                print(f"Folder '{restore_dir.name}' already exists. Use force = True to overwrite it.")
                return
            shutil.rmtree(restore_dir)

        try:
            repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()

            print(f"Cloning repository to: {restore_dir.resolve()}...")
            
            subprocess.run(["git", "clone", repo_root, str(restore_dir)], check=True, capture_output=True)

            print(f"Checking out commit: {git_head}...")
            
            subprocess.run(["git", "-C", str(restore_dir), "checkout", git_head], check=True, capture_output=True)

            if self.patch_path and self.patch_path.exists():
                print(f"Applying patch: {self.patch_path.name}...")
                subprocess.run(["git", "-C", str(restore_dir), "apply", str(self.patch_path.resolve())], check=True, capture_output=True)
                print("Commit and patch applied successfully.")
            else:
                print("Restored to commit state without patch.")
                
            git_folder = restore_dir / ".git"
            if git_folder.exists():
                shutil.rmtree(git_folder)

            print(f"\n Code restored in:\n{restore_dir.resolve()}")

        except subprocess.CalledProcessError as e:
            print(f"Git operation failed: {e}")
            if e.stderr:
                print(e.stderr.decode("utf-8"))


class RunnerViewer:
    """Runner loader for multiple Experiments"""

    def __init__(self, suite_path: str | Path):
        self.path = Path(suite_path)
        if not self.path.exists() or not self.path.is_dir():
            raise FileNotFoundError(f"Suite directory not found: {self.path}")

        # detect the most recent folder if the base folder is provided
        if not list(self.path.glob("suite_summary_*.json")):
            subdirs = sorted([d for d in self.path.iterdir() if d.is_dir()], reverse=True)
            if subdirs:
                self.path = subdirs[0]

        # load Runner summary
        summary_files = sorted(self.path.glob("suite_summary_*.json"), reverse=True)
        self.summary_data = json.loads(summary_files[0].read_text(encoding="utf-8")) if summary_files else {}

        # find the one patch file
        patch_files = sorted(self.path.glob("suite_uncommitted_*.patch"), reverse=True)
        suite_patch = patch_files[0] if patch_files else None

        # create a list of ExperimentViewer instantiations with the same patch file
        run_dirs = sorted([d for d in self.path.iterdir() if d.is_dir() and d.name.startswith("run_")])
        self.runs = [ExperimentViewer(d, patch_path=suite_patch) for d in run_dirs]

    def summary(self):
        """Show Runner summary."""

        runner_name = self.path.name
        start_date = self.summary_data.get('start_date', 'N/A')
        total_duration = self.summary_data.get('total_duration_formatted', 'N/A')

        success_count = sum(1 for r in self.runs if r.metrics.get("status") == "SUCCESS")
        total = len(self.runs)

        print(f"\n{'-' * 30}")
        print(f"SUITE: {self.path.name}")
        print(f" -> start date: {self.summary_data.get('start_date', 'N/A')}")
        print(f" -> total duration: {self.summary_data.get('total_duration_formatted', 'N/A')}")
        
        print(f" -> status breakdown: {success_count} SUCCESS / {total - success_count} FAILED")
        print(f"{'-' * 30}\n")

    def full_summary(self):
        """Show Runner summary and each Experiment summary."""

        self.summary()

        for i, run in enumerate(self.runs):
            print(f"\n + [Experiment #{i}]")
            run.summary()

    def restore_code_state(self, force: bool = False) -> None:
        """Restore code state globally"""
        if not self.runs:
            print("No runs available to extract Git metadata from.")
            return
            
        runner_restore_dir = self.path / "restored_code"
        # take the restore_code_state func from any run
        self.runs[0].restore_code_state(force=force, destination_path=runner_restore_dir)
