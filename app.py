"""GFlowNet-Prover Command-Line Entrypoint (`app.py`).

Alias for `main.py` allowing running `python app.py <COMMAND>` in CMD.
"""

from __future__ import annotations

import sys
from cli.runner import main

if __name__ == "__main__":
    sys.exit(main())
