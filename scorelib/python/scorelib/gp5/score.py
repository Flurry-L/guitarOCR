"""Export a complete multi-part recognition result."""
import json
from pathlib import Path
from scorelib import _native


def write_score_gp5(result, output):
    output = Path(output)
    _native.write_gp5(json.dumps(result), str(output))
    return output
