"""GFlowNet-Prover Command-Line Entrypoint (`main.py`).

Provides direct CMD execution for training, formal verification,
evaluation benchmarks, and model visualization.
"""

from __future__ import annotations

import sys
from cli.runner import main

if __name__ == "__main__":
    sys.exit(main())
