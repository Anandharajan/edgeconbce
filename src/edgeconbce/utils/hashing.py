from __future__ import annotations

import hashlib


def sample_fingerprint(sample_ids: list[str]) -> str:
    joined = "".join(sample_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

