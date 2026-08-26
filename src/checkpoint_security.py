import hashlib
import hmac
import re
from pathlib import Path


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def verify_sha256(file_path, expected_sha256):
    if not expected_sha256 or not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(
            "A trusted 64-character SHA-256 digest is required before loading "
            f"the legacy checkpoint: {file_path}"
        )

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)

    if not hmac.compare_digest(digest.hexdigest(), expected_sha256.lower()):
        raise ValueError(f"Checkpoint SHA-256 verification failed: {file_path}")
