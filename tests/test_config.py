"""Configuration contract regression tests."""
import os
import unittest
from unittest.mock import patch

from code.config import Settings


class SettingsTests(unittest.TestCase):
    def test_rejects_unknown_media_mode(self):
        with patch.dict(os.environ, {"ROUTER_LLM_PROVIDER": "auto",
                                    "ROUTER_MEDIA_MODE": "unsupported"}):
            with self.assertRaisesRegex(ValueError, "ROUTER_MEDIA_MODE"):
                Settings.from_environment()

    def test_accepts_disabled_media_mode(self):
        with patch.dict(os.environ, {"ROUTER_LLM_PROVIDER": "auto",
                                    "ROUTER_MEDIA_MODE": "off"}):
            self.assertEqual("off", Settings.from_environment().media_mode)


if __name__ == "__main__":
    unittest.main()
