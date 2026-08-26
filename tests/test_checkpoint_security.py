import hashlib
import tempfile
import unittest
from pathlib import Path

from checkpoint_security import verify_sha256


class VerifySha256Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp_dir.name) / "model.ckpt"
        self.checkpoint.write_bytes(b"trusted checkpoint")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_matching_digest(self):
        expected = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

        verify_sha256(self.checkpoint, expected)

    def test_rejects_mismatched_digest(self):
        with self.assertRaisesRegex(ValueError, "verification failed"):
            verify_sha256(self.checkpoint, "0" * 64)

    def test_requires_a_valid_digest(self):
        with self.assertRaisesRegex(ValueError, "trusted 64-character"):
            verify_sha256(self.checkpoint, None)


if __name__ == "__main__":
    unittest.main()
