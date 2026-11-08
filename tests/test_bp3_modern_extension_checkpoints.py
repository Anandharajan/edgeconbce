from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

import torch

from edgeconbce.pipeline import bp3_modern_extension as modern


class BP3ModernExtensionCheckpointTests(unittest.TestCase):
    def test_missing_contact_regression_checkpoint_uses_contact_free_fallback(self) -> None:
        fake_root = Path("outputs") / "_test_tmp" / "missing_esm_contact"
        checkpoint_dir = fake_root / ".torchcache" / "hub" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}, "cfg": {"model": {}}}, checkpoint_dir / "esm2_t6_8M_UR50D.pt")
        modern._ESM_CACHE.clear()
        try:
            model = torch.nn.Linear(1, 1)
            alphabet = Mock()
            alphabet.get_batch_converter.return_value = object()
            with patch.object(modern, "_project_root", return_value=fake_root.resolve()):
                with patch.object(modern.esm.pretrained, "load_model_and_alphabet_core", return_value=(model, alphabet)):
                    with self.assertWarnsRegex(RuntimeWarning, "contact-regression"):
                        loaded_model, loaded_alphabet, batch_converter = modern._ensure_esm_loaded()
            self.assertIs(loaded_model, model)
            self.assertIs(loaded_alphabet, alphabet)
            self.assertIs(batch_converter, alphabet.get_batch_converter.return_value)
            self.assertFalse(modern._ESM_CACHE["contacts_available"])
        finally:
            modern._ESM_CACHE.clear()
            if fake_root.exists():
                shutil.rmtree(fake_root)


if __name__ == "__main__":
    unittest.main()
