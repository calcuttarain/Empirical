import json
import pickle
import numpy as np
from pathlib import Path

class ArtifactSaver:
    """Utilitary class for saving files."""

    @staticmethod
    def save_json(data: dict | list, file_path: Path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def save_text(text: str, file_path: Path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def append_text(text: str, file_path: Path):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(text)

    @classmethod
    def save_result(cls, result, run_dir: Path, ext: str | None = None, filename: str = "result"):
        """Save result using the specified extension or fallback to automatic saving."""
        if ext is None:
            cls._save_auto_fallback(result, run_dir, filename)
            return

        ext = ext.lower()
        if not ext.startswith('.'):
            ext = f".{ext}"
            
        file_path = run_dir / f"{filename}{ext}"

        try:
            if ext == '.npy':
                np.save(file_path, result)
            elif ext == '.npz':
                if isinstance(result, dict):
                    np.savez(file_path, **result)
                else:
                    np.savez(file_path, data=result)
            elif ext == '.parquet':
                import pandas as pd
                df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
                df.to_parquet(file_path)
            elif ext == '.json':
                cls.save_json(result, file_path)
            elif ext in ['.pkl', '.pickle']:
                with open(file_path, "wb") as f:
                    pickle.dump(result, f)
            else:
                raise ValueError(f"Extension '{ext}' is not supported. Fallback to automatic format.")
                cls._save_auto_fallback(result, run_dir, filename)
                
        except Exception as e:
            cls._save_auto_fallback(result, run_dir, filename)

    @staticmethod
    def _save_auto_fallback(result, run_dir: Path, filename: str):
        """Duck typing method for authomatic fallback"""
        if isinstance(result, np.ndarray):
            np.save(run_dir / f"{filename}.npy", result)
        elif isinstance(result, dict):
            with open(run_dir / f"{filename}.pkl", "wb") as f:
                pickle.dump(result, f)
        else:
            ArtifactSaver.save_text(str(result), run_dir / f"{filename}.txt")


class ArtifactLoader:
    """Utilitary class for loading saved artifacts."""

    @staticmethod
    def load_json(file_path: Path) -> dict | list:
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_text(file_path: Path) -> str:
        if not file_path.exists():
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def load_result(cls, dir_path: Path, filename: str = "result"):
        """
        Load result files by checking the extension. 
        """
        if (dir_path / f"{filename}.npy").exists():
            return np.load(dir_path / f"{filename}.npy", allow_pickle=True)
            
        elif (dir_path / f"{filename}.npz").exists():
            data = np.load(dir_path / f"{filename}.npz", allow_pickle=True)
            return {k: data[k] for k in data.files}
            
        elif (dir_path / f"{filename}.parquet").exists():
            import pandas as pd
            return pd.read_parquet(dir_path / f"{filename}.parquet")
            
        elif (dir_path / f"{filename}.pkl").exists():
            with open(dir_path / f"{filename}.pkl", "rb") as f:
                return pickle.load(f)
                
        elif (dir_path / f"{filename}.json").exists():
            return cls.load_json(dir_path / f"{filename}.json")
            
        elif (dir_path / f"{filename}.txt").exists():
            return cls.load_text(dir_path / f"{filename}.txt")
            
        return None
