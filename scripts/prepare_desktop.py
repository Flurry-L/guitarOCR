"""Compatibility development entry point; packaging is implemented in Node, never Python."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    raise SystemExit(subprocess.call([
        'node', str(Path(__file__).with_name('prepare_native_desktop.mjs')), *sys.argv[1:]
    ]))
