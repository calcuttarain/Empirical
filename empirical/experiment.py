import numpy as np

import pickle
import inspect
import time
import json
import sys
import tracemalloc
import functools
import contextlib
import traceback
from pathlib import Path

from .utils.metadata import get_metadata, get_git_diff 
from .utils.duallogger import DualLogger
from utils.io import ArtifactSaver

class Experiment:
    def __init__(self, name, base_dir = "results", run_id = None, save_git_patch = True, save_results = True, result_ext=None):
        self.name = name 
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.save_git_patch = save_git_patch
        self.save_results = save_results
        self.result_ext = result_ext

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # just for console printing
            if not self.save_results:
                returned_obj = func(*args, **kwargs)
                
                print(f"Starting experiment '{self.name}'.")
                
                if inspect.isgenerator(returned_obj):
                    step = 0
                    while True:
                        step_start = time.perf_counter()
                        try:
                            yielded_val = next(returned_obj)
                            step_time = time.perf_counter() - step_start
                            msg = f"[Wrapper] Iteration {step} -> Time {step_time:.4f}s"
                            if yielded_val is not None:
                                if isinstance(yielded_val, dict):
                                    msg += f" | {self._format_yielded_dict(yielded_val)}"
                                else:
                                    msg += f" | {yielded_val}"
                            print(msg)
                            step += 1
                        except StopIteration as e:
                            return e.value
                return returned_obj

            meta = get_metadata()

            run_dir = self._create_run_dir(meta)

            self._save_params_metadata(meta, args, kwargs, run_dir)

            log_file_path = run_dir / "output.log"

            # define dual loggers for both log file and console output
            dual_stdout = DualLogger(log_file_path, sys.stdout)
            dual_stderr = DualLogger(log_file_path, sys.stderr)

            print(f"Starting experiment '{self.name}'. Saved in '{run_dir}'")

            result = None
            status = "SUCCESS"
            history_data = []

            start_time = time.perf_counter()

            # start memory tracing
            tracemalloc.start()

            try:
                with (contextlib.redirect_stdout(dual_stdout), contextlib.redirect_stderr(dual_stderr)):
                    
                    returned_obj = func(*args, **kwargs)
                    
                    # generator
                    if inspect.isgenerator(returned_obj):
                        step = 0
                        while True:
                            step_start = time.perf_counter()
                            try:
                                yielded_val = next(returned_obj)
                                
                                step_time = time.perf_counter() - step_start
                                
                                msg = f"[Wrapper] [Iteration {step}] -> Time {step_time:.4f}s"
                                if yielded_val is not None:

                                    # save iteration time along with iteration data
                                    if isinstance(yielded_val, dict):
                                        yielded_val["step_duration_s"] = step_time
                                    
                                    # save iteration data
                                    history_data.append(yielded_val)

                                    # append iteration data to console message
                                    if isinstance(yielded_val, dict):
                                        msg += f" | {self._format_yielded_dict(yielded_val)}"
                                    else:
                                        msg += f" | {yielded_val}"
                                print(msg)
                                
                                step += 1
                                
                            except StopIteration as e:
                                result = e.value
                                break
                    else:
                        # function is not a generator
                        result = returned_obj
                        
            # save error and stacktrace 
            except Exception as e:
                status = "FAILED"
                error_msg = f"\n[ERROR]: {str(e)}\n{traceback.format_exc()}"
                ArtifactSaver.append_text(error_msg, log_file_path)
                raise
                
            # save experiment metrics
            finally:
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                end_time = time.perf_counter()
                duration_seconds = end_time - start_time
                duration_formatted = time.strftime('%H:%M:%S', time.gmtime(duration_seconds))

                end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                metrics = {
                    "status": status,
                    "end_timestamp": end_timestamp,
                    "duration_seconds": round(duration_seconds, 4),
                    "duration_formatted": duration_formatted,
                    "peak_ram_mb": round(peak_mem / (1024 * 1024), 2)
                }

                ArtifactSaver.save_json(metrics, run_dir / "metrics.json")

                if history_data:
                    ArtifactSaver.save_json(history_data, run_dir / "history.json")

            if result is not None:
                ArtifactSaver.save_result(result, run_dir, ext=self.result_ext)
                
            return result
            
        return wrapper

    def _save_params_metadata(self, meta, args, kwargs, run_dir):
        # save uncommited changes in a .patch
        if self.save_git_patch and meta.get("has_uncommitted_changes"):
            diff_text = get_git_diff()
            if diff_text:
                ArtifactSaver.save_text(diff_text, run_dir / "uncommitted_changes.patch")
        
        # save function parameters
        run_params = {
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()}
        }
        
        ArtifactSaver.save_json({"metadata": meta, "parameters": run_params}, run_dir / "params.json")

             
    def _create_run_dir(self, meta):
        if self.run_id:
            run_dir = self.base_dir / self.name / self.run_id 
        else:
            # get a short hash for readable id
            short_hash = meta["git_head"][:7] if meta["git_head"] != "untracked" else "untracked"
            run_id = f"{meta['timestamp']}_{short_hash}"

            # define running folder
            run_dir = self.base_dir / self.name / run_id

        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir
    
    @staticmethod
    def _format_yielded_dict(data: dict) -> str:
        formatted_items = []
        for k, v in data.items():
            if k == "step_duration_s":
                continue
            
            # extract values from numpy
            val = v.item() if hasattr(v, "item") else v

            
            if isinstance(val, (float, np.floating)):
                formatted_items.append(f"{k}: {val:.4f}")
            else:
                formatted_items.append(f"{k}: {val}")
                
        return " | ".join(formatted_items)
