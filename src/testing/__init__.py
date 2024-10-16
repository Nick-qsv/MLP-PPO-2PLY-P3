import subprocess


def build():
    """Runs the build.sh script."""
    result = subprocess.run(["./build.sh"], capture_output=True, text=True)
    return result.stdout  # Return the output of the script


def run_tests():
    """Runs the run_tests.sh script."""
    result = subprocess.run(["./run_tests.sh"], capture_output=True, text=True)
    return result.stdout  # Return the output of the script


__all__ = ["build", "run_tests"]
