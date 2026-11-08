from __future__ import annotations

import platform

from edgeconbce import __version__


def system_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "edgeconbce_version": __version__,
    }
