import subprocess
import time
import sys
import platform

def get_git_head():
    """Return the hash of the latest commit."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT).strip().decode('utf-8')
    except Exception:
        return "untracked"

def get_git_diff():
    """Get uncommited changes."""
    try:
        subprocess.run(["git", "add", "-N", "."], stderr=subprocess.DEVNULL)

        return subprocess.check_output(
                ["git", "diff", "HEAD"], stderr=subprocess.STDOUT
                ).decode("utf-8")
    except Exception:
        return ""

def get_metadata():
    """
    Return metadata:
        -> timestamp 
        -> current commit hash 
        -> python version 
        -> platform 
    """
    diff_text = get_git_diff()
    has_changes = len(diff_text.strip()) > 0
    
    return {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "git_head": get_git_head(),
        "has_uncommitted_changes": has_changes,
        "python_version": sys.version.split()[0],
        "system": platform.platform()
    }
