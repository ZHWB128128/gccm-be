"""Package entry point: ``python -m gccm_be`` starts the Web dashboard + API."""
from __future__ import annotations

import sys

from .app.launcher import main

if __name__ == "__main__":
    sys.exit(main())
